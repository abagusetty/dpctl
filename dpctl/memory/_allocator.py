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

_registry: dict = {}
_lock = threading.RLock()

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


def set_allocator(
    allocator: Optional[Callable],
    *,
    usm_type: str = "device",
    sycl_device: Optional[dpctl.SyclDevice] = None,
) -> None:
    """Install a USM allocator hook.

    Args:
        allocator: A callable ``(nbytes, sycl_queue) -> _Memory``, or
            ``None`` to remove the hook for the given
            ``(usm_type, sycl_device)`` combination.
        usm_type: One of ``"device"``, ``"shared"``, or ``"host"``.
        sycl_device: An optional :class:`dpctl.SyclDevice`. ``None``
            installs the hook as the fallback for all devices that do
            not have a device-specific hook installed.

    A device-specific hook takes precedence over a device-``None``
    fallback. If neither is installed, the allocation goes directly to
    ``sycl::malloc_*``.
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
    matches."""
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
    """Remove allocator hooks. With no arguments, clears the entire
    registry."""
    with _lock:
        if usm_type is None and sycl_device is None:
            _registry.clear()
            return
        if usm_type is not None:
            usm_type_norm = _normalize_usm_type(usm_type)
            if sycl_device is None:
                keys = [k for k in _registry if k[0] == usm_type_norm]
                for k in keys:
                    _registry.pop(k, None)
            else:
                _registry.pop(
                    (usm_type_norm, _device_key(sycl_device)), None
                )
        else:
            dev_key = _device_key(sycl_device)
            keys = [k for k in _registry if k[1] == dev_key]
            for k in keys:
                _registry.pop(k, None)


def _lookup_allocator(
    usm_type: str, sycl_device: Optional[dpctl.SyclDevice]
) -> Optional[Callable]:
    if _bypass_active():
        return None
    dev_key = _device_key(sycl_device) if sycl_device is not None else None
    with _lock:
        if dev_key is not None:
            specific = _registry.get((usm_type, dev_key))
            if specific is not None:
                return specific
        return _registry.get((usm_type, None))
