#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pluggable USM allocator hook for :mod:`dpctl.memory`.

This module implements a process-wide registry that lets users (or
upstream libraries such as ``dpnp`` and ``gpu4pyscf``) install a custom
allocator for one or more ``(usm_type, sycl_device)`` combinations. When
a hook is installed, every subsequent construction of
:class:`dpctl.memory.MemoryUSMShared`, :class:`MemoryUSMHost`, or
:class:`MemoryUSMDevice` of the matching type and device routes its
allocation through the user callable.

When no hook is installed (the default), allocations go directly to
``sycl::malloc_*`` exactly as in earlier dpctl releases; the cost of the
registry lookup on the legacy path is a single dictionary access.

A typical use mirrors the CuPy pattern::

    import dpctl
    import dpctl.memory as dpm

    q = dpctl.SyclQueue()
    # Use the process-wide default pool so cache is shared with dpnp
    # and any other SYCL-using library in the same process.
    pool = dpm.MemoryPool.get_default(sycl_queue=q, usm_type="device")
    dpm.set_allocator(pool.malloc, usm_type="device",
                      sycl_device=q.sycl_device)

    # All subsequent USM-device allocations go through ``pool``:
    m = dpm.MemoryUSMDevice(1 << 20, queue=q)  # served from pool

    # gpu4pyscf-style hybrid: pool below a threshold, direct above.
    THRESHOLD = 64 << 20
    direct_malloc = dpm.malloc_device
    pool_malloc = pool.malloc
    def hybrid(nbytes, queue):
        if nbytes >= THRESHOLD:
            return direct_malloc(nbytes, queue=queue)
        return pool_malloc(nbytes, queue)
    dpm.set_allocator(hybrid, usm_type="device",
                      sycl_device=q.sycl_device)
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

import dpctl

__all__ = [
    "get_allocator",
    "reset_allocator",
    "set_allocator",
]

# Module-level state ------------------------------------------------------

_VALID_USM_TYPES = frozenset({"device", "shared", "host"})

# Registry: (usm_type_str, device_key_or_None) -> callable
# ``device_key_or_None`` is the device's hash() result (an int) when a
# specific device was requested, or ``None`` when the hook applies to
# "any device of this USM type that doesn't have a more-specific hook
# installed."
_registry: dict = {}
_lock = threading.RLock()

# Thread-local flag used by ``malloc_device`` / ``malloc_shared`` /
# ``malloc_host`` (and any user code that wants a one-shot bypass) to
# tell ``_lookup_allocator`` to ignore the registry on this thread.
# Using thread-local state avoids the race that a global registry
# mutation would introduce (where a concurrent allocation on another
# thread would also miss its hook). Implemented as a counter so that
# nested bypasses compose correctly.
_bypass_tls = threading.local()


def _bypass_active() -> bool:
    return getattr(_bypass_tls, "depth", 0) > 0


class _bypass_hooks:
    """Context manager that suspends allocator-hook lookup on the
    current thread for the duration of a ``with`` block. Used by
    :func:`dpctl.memory.malloc_device` and friends; also available for
    user code that needs a one-shot bypass.
    """

    def __enter__(self):
        _bypass_tls.depth = getattr(_bypass_tls, "depth", 0) + 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _bypass_tls.depth = getattr(_bypass_tls, "depth", 1) - 1
        return False


def _device_key(sycl_device: Optional[dpctl.SyclDevice]) -> Optional[int]:
    """Convert a SyclDevice (or None) to a registry key."""
    if sycl_device is None:
        return None
    if not isinstance(sycl_device, dpctl.SyclDevice):
        raise TypeError(
            "sycl_device must be a dpctl.SyclDevice instance or None; "
            f"got {type(sycl_device).__name__}"
        )
    return hash(sycl_device)


def _normalize_usm_type(usm_type: str) -> str:
    if not isinstance(usm_type, str):
        raise TypeError(
            f"usm_type must be a string, got {type(usm_type).__name__}"
        )
    usm_type_norm = usm_type.lower()
    if usm_type_norm not in _VALID_USM_TYPES:
        raise ValueError(
            f"usm_type must be one of {sorted(_VALID_USM_TYPES)}; "
            f"got {usm_type!r}"
        )
    return usm_type_norm


def set_allocator(
    allocator: Optional[Callable],
    *,
    usm_type: str = "device",
    sycl_device: Optional[dpctl.SyclDevice] = None,
) -> None:
    """Install a USM allocator hook.

    Args:
        allocator: A callable with signature
            ``allocator(nbytes: int, sycl_queue: dpctl.SyclQueue)
            -> dpctl.memory._Memory``. It must return a ``_Memory``
            instance (typically a ``MemoryUSM{Device,Shared,Host}``) bound
            to a USM allocation of size at least ``nbytes`` in the context
            of the provided queue.

            If ``None``, removes any previously installed hook for the
            given ``(usm_type, sycl_device)`` combination, restoring the
            legacy direct-allocation behavior for that combination.

        usm_type: One of ``"device"``, ``"shared"``, or ``"host"``.
            Selects which kind of USM allocation this hook serves.

        sycl_device: An optional :class:`dpctl.SyclDevice` restricting
            the hook to allocations whose target queue is bound to this
            device. ``None`` (the default) installs the hook as the
            fallback for all devices that do not have a device-specific
            hook installed.

    Notes:
        Lookups prefer the most specific match: a device-specific hook
        wins over a device-``None`` fallback. If neither is installed,
        the allocation goes directly to ``sycl::malloc_*``.
    """
    usm_type_norm = _normalize_usm_type(usm_type)
    key = (usm_type_norm, _device_key(sycl_device))
    with _lock:
        if allocator is None:
            _registry.pop(key, None)
            return
        if not callable(allocator):
            raise TypeError(
                "allocator must be callable or None; "
                f"got {type(allocator).__name__}"
            )
        _registry[key] = allocator


def get_allocator(
    *,
    usm_type: str = "device",
    sycl_device: Optional[dpctl.SyclDevice] = None,
) -> Optional[Callable]:
    """Return the allocator hook installed for a given
    ``(usm_type, sycl_device)`` combination, or ``None`` if no hook
    matches.

    Lookup precedence is identical to :func:`_lookup_allocator`: a
    device-specific hook overrides a device-``None`` fallback.
    """
    usm_type_norm = _normalize_usm_type(usm_type)
    dev_key = _device_key(sycl_device)
    with _lock:
        if dev_key is not None:
            specific = _registry.get((usm_type_norm, dev_key))
            if specific is not None:
                return specific
        return _registry.get((usm_type_norm, None))


def reset_allocator(
    *,
    usm_type: Optional[str] = None,
    sycl_device: Optional[dpctl.SyclDevice] = None,
) -> None:
    """Remove allocator hooks.

    With no arguments, clears the entire registry, restoring legacy
    behavior for every ``(usm_type, sycl_device)`` combination.

    When ``usm_type`` is given, clears hooks only for that USM type;
    ``sycl_device`` further narrows the reset.
    """
    with _lock:
        if usm_type is None and sycl_device is None:
            _registry.clear()
            return
        if usm_type is not None:
            usm_type_norm = _normalize_usm_type(usm_type)
            if sycl_device is None:
                # Remove all hooks for this usm_type, any device.
                keys = [k for k in _registry if k[0] == usm_type_norm]
                for k in keys:
                    _registry.pop(k, None)
            else:
                _registry.pop(
                    (usm_type_norm, _device_key(sycl_device)), None
                )
        else:
            # usm_type is None but sycl_device is set.
            dev_key = _device_key(sycl_device)
            keys = [k for k in _registry if k[1] == dev_key]
            for k in keys:
                _registry.pop(k, None)


def _lookup_allocator(
    usm_type: str, sycl_device: Optional[dpctl.SyclDevice]
) -> Optional[Callable]:
    """Internal fast-path lookup used by ``_Memory._cinit_alloc``.

    Returns ``None`` when no hook applies — this is the common case and
    is intentionally cheap (one or two dict.get calls under a lock).
    Honors the per-thread bypass flag set by
    :class:`_bypass_hooks`.
    """
    # ``usm_type`` is supplied by the Cython caller as the already-decoded
    # string ("shared"/"host"/"device") so we skip ``_normalize_usm_type``
    # to keep the common-no-hook path branchless.
    if _bypass_active():
        return None
    dev_key = _device_key(sycl_device) if sycl_device is not None else None
    with _lock:
        if dev_key is not None:
            specific = _registry.get((usm_type, dev_key))
            if specific is not None:
                return specific
        return _registry.get((usm_type, None))
