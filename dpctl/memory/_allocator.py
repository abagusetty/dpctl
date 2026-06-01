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

"""Pluggable USM allocator hook for :mod:`dpctl.memory`."""

from __future__ import annotations

import threading
from typing import Callable, Optional

import dpctl

__all__ = [
    "get_allocator",
    "reset_allocator",
    "set_allocator",
]

_VALID_USM_TYPES = frozenset({"device", "shared", "host"})

# Registry of installed hooks. CPython's GIL makes both ``dict.get`` and
# single-key mutations atomic, so the hot read path in
# ``_Memory._cinit_alloc`` reads this dict *without* taking ``_lock``.
# ``_lock`` is held only by writers (``set_allocator`` /
# ``reset_allocator``) to serialize against each other.
_registry: dict = {}
_lock = threading.RLock()

# Thread-local bypass counter used by ``malloc_device`` and friends to
# suspend hook lookup on the current thread for the duration of a direct
# allocation. The Cython hot path reads ``_bypass_tls.depth`` directly.
_bypass_tls = threading.local()


def _bypass_active() -> bool:
    return getattr(_bypass_tls, "depth", 0) > 0


class _bypass_hooks:
    """Context manager that suspends allocator-hook lookup on the
    current thread for the duration of a ``with`` block."""

    def __enter__(self):
        _bypass_tls.depth = getattr(_bypass_tls, "depth", 0) + 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _bypass_tls.depth = getattr(_bypass_tls, "depth", 1) - 1
        return False


def _device_key(sycl_device: Optional[dpctl.SyclDevice]) -> Optional[int]:
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


_SENTINEL = object()


def _pool_from_allocator(allocator):
    """If ``allocator`` is a :class:`MemoryPool` or a bound method of
    one (e.g. ``pool.malloc``), return the underlying pool. Otherwise
    return ``None``."""
    from ._memory_pool import MemoryPool

    if isinstance(allocator, MemoryPool):
        return allocator
    owner = getattr(allocator, "__self__", None)
    if isinstance(owner, MemoryPool):
        return owner
    return None


def set_allocator(
    allocator: Optional[Callable],
    *,
    usm_type=_SENTINEL,
    sycl_device=_SENTINEL,
) -> None:
    """Install a USM allocator hook.

    Args:
        allocator: One of

            * a :class:`dpctl.memory.MemoryPool` instance,
            * a bound method of a ``MemoryPool`` (e.g. ``pool.malloc``),
            * any callable ``(nbytes, sycl_queue) -> _Memory``,
            * or ``None`` to remove the hook for the given
              ``(usm_type, sycl_device)`` combination.

            When a :class:`MemoryPool` (or its bound method) is passed,
            ``usm_type`` and ``sycl_device`` default to the pool's
            corresponding attributes; explicit kwargs override but must
            be consistent with the pool's attributes (a mismatch is
            rejected).

        usm_type: One of ``"device"``, ``"shared"``, or ``"host"``.
            Required when ``allocator`` is a plain callable.
        sycl_device: An optional :class:`dpctl.SyclDevice`. ``None``
            installs the hook as the fallback for all devices that do
            not have a device-specific hook installed.

    A device-specific hook takes precedence over a device-``None``
    fallback. If neither is installed, the allocation goes directly to
    ``sycl::malloc_*``.
    """
    pool = _pool_from_allocator(allocator) if allocator is not None else None
    if pool is not None:
        resolved_usm = pool.usm_type
        resolved_dev = pool.sycl_device
        if usm_type is not _SENTINEL:
            if _normalize_usm_type(usm_type) != resolved_usm:
                raise ValueError(
                    f"usm_type={usm_type!r} conflicts with the pool's "
                    f"usm_type={resolved_usm!r}"
                )
        if sycl_device is not _SENTINEL and sycl_device is not None:
            if sycl_device != resolved_dev:
                raise ValueError(
                    "sycl_device argument conflicts with the pool's "
                    "sycl_device"
                )
        usm_type_norm = resolved_usm
        dev_key = _device_key(resolved_dev)
    else:
        usm_type_norm = _normalize_usm_type(
            "device" if usm_type is _SENTINEL else usm_type
        )
        dev_key = _device_key(
            None if sycl_device is _SENTINEL else sycl_device
        )

    key = (usm_type_norm, dev_key)
    with _lock:
        if allocator is None:
            removed = _registry.pop(key, None)
            _sync_c_registry_on_remove(key, removed)
            return
        if not callable(allocator):
            raise TypeError(
                "allocator must be callable or None; "
                f"got {type(allocator).__name__}"
            )
        _registry[key] = allocator
        _sync_c_registry_on_install(key, allocator, pool)


def get_allocator(
    *,
    usm_type: str = "device",
    sycl_device: Optional[dpctl.SyclDevice] = None,
) -> Optional[Callable]:
    """Return the allocator hook installed for a given
    ``(usm_type, sycl_device)`` combination, or ``None`` if no hook
    matches."""
    usm_type_norm = _normalize_usm_type(usm_type)
    dev_key = _device_key(sycl_device)
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
    """Remove allocator hooks. With no arguments, clears the entire
    registry."""
    with _lock:
        if usm_type is None and sycl_device is None:
            removed = list(_registry.items())
            _registry.clear()
            for k, v in removed:
                _sync_c_registry_on_remove(k, v)
            return
        if usm_type is not None:
            usm_type_norm = _normalize_usm_type(usm_type)
            if sycl_device is None:
                for k in [k for k in _registry if k[0] == usm_type_norm]:
                    v = _registry.pop(k, None)
                    _sync_c_registry_on_remove(k, v)
            else:
                k = (usm_type_norm, _device_key(sycl_device))
                v = _registry.pop(k, None)
                _sync_c_registry_on_remove(k, v)
        else:
            dev_key = _device_key(sycl_device)
            for k in [k for k in _registry if k[1] == dev_key]:
                v = _registry.pop(k, None)
                _sync_c_registry_on_remove(k, v)


def _sync_c_registry_on_install(key, allocator, pool) -> None:
    """Mirror device-USM pool installs into the C-side registry so
    C++ consumers (dpnp's smart_malloc_*) can find the pool without
    going through Python."""
    if pool is None or key[0] != "device" or key[1] is None:
        return
    from ._memory_pool import _install_device_pool

    _install_device_pool(pool)


def _sync_c_registry_on_remove(key, removed_value) -> None:
    """Clear the matching C-side registry entry when a Python-side
    device-USM hook for a specific device is removed."""
    if key[0] != "device" or key[1] is None or removed_value is None:
        return
    pool = _pool_from_allocator(removed_value)
    if pool is None:
        return
    from ._memory_pool import _uninstall_device_pool

    _uninstall_device_pool(pool.sycl_context, pool.sycl_device)
