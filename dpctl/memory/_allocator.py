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

"""Pluggable USM-device allocator hook for :mod:`dpctl.memory`.

The registry maps a :class:`dpctl.SyclDevice` (or ``None`` for the
global fallback) to an allocator callable. Only USM-device
allocations consult the registry; USM-shared and USM-host
allocations always go directly to ``sycl::malloc_*`` because the
underlying ``sycl_ext_oneapi_async_memory_alloc`` extension does not
support pooled shared / host allocations.
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

# Registry of installed device-USM hooks. CPython's GIL makes both
# ``dict.get`` and single-key mutations atomic, so the hot read path
# in ``_Memory._cinit_alloc`` reads this dict *without* taking
# ``_lock``. ``_lock`` is held only by writers (``set_allocator`` /
# ``reset_allocator``) to serialize against each other.
#
# Key: hash(SyclDevice) for a per-device hook, or None for the
# global fallback (applies to any device that has no specific hook).
_registry: dict = {}
_lock = threading.RLock()

# Thread-local bypass counter used by ``malloc_device`` and friends to
# suspend hook lookup on the current thread for the duration of a
# direct allocation. The Cython hot path reads ``_bypass_tls.depth``
# directly.
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
    sycl_device=_SENTINEL,
) -> None:
    """Install a USM-device allocator hook.

    Args:
        allocator: One of

            * a :class:`dpctl.memory.MemoryPool` instance,
            * a bound method of a ``MemoryPool`` (e.g. ``pool.malloc``),
            * any callable ``(nbytes, sycl_queue) -> MemoryUSMDevice``,
            * or ``None`` to remove the hook for the given
              ``sycl_device`` (or the global fallback if
              ``sycl_device`` is not provided).

            When a :class:`MemoryPool` (or its bound method) is passed,
            ``sycl_device`` defaults to the pool's
            :attr:`MemoryPool.sycl_device`; if explicitly provided it
            must agree with the pool.

        sycl_device: An optional :class:`dpctl.SyclDevice`. ``None``
            installs the hook as the fallback for all devices that do
            not have a device-specific hook installed.

    A device-specific hook takes precedence over a device-``None``
    fallback. If neither is installed, the device allocation goes
    directly to ``sycl::malloc_device``.

    Only USM-device allocations are affected; USM-shared and USM-host
    allocations always go directly to ``sycl::malloc_*``.
    """
    pool = _pool_from_allocator(allocator) if allocator is not None else None
    if pool is not None:
        resolved_dev = pool.sycl_device
        if sycl_device is not _SENTINEL and sycl_device is not None:
            if sycl_device != resolved_dev:
                raise ValueError(
                    "sycl_device argument conflicts with the pool's "
                    "sycl_device"
                )
        dev_key = _device_key(resolved_dev)
    else:
        dev_key = _device_key(
            None if sycl_device is _SENTINEL else sycl_device
        )

    with _lock:
        if allocator is None:
            removed = _registry.pop(dev_key, None)
            _sync_c_registry_on_remove(dev_key, removed)
            return
        if not callable(allocator):
            raise TypeError(
                "allocator must be callable or None; "
                f"got {type(allocator).__name__}"
            )
        _registry[dev_key] = allocator
        _sync_c_registry_on_install(dev_key, allocator, pool)


def get_allocator(
    *,
    sycl_device: Optional[dpctl.SyclDevice] = None,
) -> Optional[Callable]:
    """Return the USM-device allocator hook installed for the given
    ``sycl_device``, falling back to the device-``None`` global hook
    if no per-device hook is registered. Returns ``None`` if no hook
    matches."""
    dev_key = _device_key(sycl_device)
    if dev_key is not None:
        specific = _registry.get(dev_key)
        if specific is not None:
            return specific
    return _registry.get(None)


def reset_allocator(
    *,
    sycl_device: Optional[dpctl.SyclDevice] = None,
) -> None:
    """Remove USM-device allocator hooks.

    With no arguments, clears the entire registry. With
    ``sycl_device``, clears only that device's hook. To clear the
    global fallback specifically, pass ``sycl_device=None``
    explicitly (which is the default when no argument is given, so
    the no-arg call clears everything by being unambiguous).
    """
    with _lock:
        if sycl_device is None:
            removed = list(_registry.items())
            _registry.clear()
            for k, v in removed:
                _sync_c_registry_on_remove(k, v)
            return
        dev_key = _device_key(sycl_device)
        v = _registry.pop(dev_key, None)
        _sync_c_registry_on_remove(dev_key, v)


def _sync_c_registry_on_install(dev_key, allocator, pool) -> None:
    """Mirror device-USM pool installs into the C-side registry so
    C++ consumers (dpnp's smart_malloc_device) can find the pool
    without going through Python."""
    if pool is None or dev_key is None:
        return
    from ._memory_pool import _install_device_pool

    _install_device_pool(pool)


def _sync_c_registry_on_remove(dev_key, removed_value) -> None:
    """Clear the matching C-side registry entry when a Python-side
    device-USM hook for a specific device is removed."""
    if dev_key is None or removed_value is None:
        return
    pool = _pool_from_allocator(removed_value)
    if pool is None:
        return
    from ._memory_pool import _uninstall_device_pool

    _uninstall_device_pool(pool.sycl_context, pool.sycl_device)
