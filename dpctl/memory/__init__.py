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

"""
**Data Parallel Control Memory** provides Python objects for untyped USM
memory container of bytes for each kind of USM pointers: shared pointers,
device pointers and host pointers.

Shared and host pointers are accessible from both host and a device,
while device pointers are only accessible from device.

Python objects corresponding to shared and host pointers implement
Python simple buffer protocol. It is therefore possible to use these
objects to maniputalate USM memory using NumPy or `bytearray`,
`memoryview`, or `array.array` classes.

This module also exposes a pluggable allocator hook
(:func:`set_allocator`) together with a stream-ordered
:class:`MemoryPool` implementation. When no allocator is installed
(the default), every allocation calls ``sycl::malloc_*`` and every
deallocation calls ``sycl::free``. For the common opt-in case use
:func:`use_default_pool` to install the SYCL runtime's default
pool for every supported USM kind in one call.
"""

from ._allocator import get_allocator, reset_allocator, set_allocator
from ._memory import (
    MemoryUSMDevice,
    MemoryUSMHost,
    MemoryUSMShared,
    USMAllocationError,
    as_usm_memory,
)
from ._memory_pool import MemoryPool, is_memory_pool_available


def malloc_device(nbytes, queue=None):
    """Allocate ``nbytes`` bytes of USM-device memory bypassing any
    installed allocator hook."""
    return _direct_alloc(MemoryUSMDevice, nbytes, queue)


def malloc_shared(nbytes, queue=None):
    """Allocate ``nbytes`` bytes of USM-shared memory bypassing any
    installed allocator hook."""
    return _direct_alloc(MemoryUSMShared, nbytes, queue)


def malloc_host(nbytes, queue=None):
    """Allocate ``nbytes`` bytes of USM-host memory bypassing any
    installed allocator hook."""
    return _direct_alloc(MemoryUSMHost, nbytes, queue)


def _direct_alloc(cls, nbytes, queue):
    import dpctl

    from ._allocator import _bypass_hooks

    if queue is None:
        queue = dpctl.SyclQueue()
    with _bypass_hooks():
        return cls(nbytes, queue=queue)


def use_default_pool(
    *, sycl_queue=None, usm_types=("device", "shared", "host")
):
    """Install the SYCL runtime's default :class:`MemoryPool` as the
    USM allocator for every USM kind in ``usm_types``.

    Convenience wrapper for the common opt-in pattern::

        for kind in usm_types:
            pool = MemoryPool.get_default(sycl_queue=q, usm_type=kind)
            set_allocator(pool)

    Args:
        sycl_queue (Optional[:class:`dpctl.SyclQueue`]):
            Queue whose ``(context, device)`` selects which pool to
            install. ``None`` uses dpctl's cached default queue.
        usm_types (Iterable[str]):
            USM kinds to install pools for. Defaults to all three.
            Kinds that the underlying SYCL extension rejects (today
            only ``"device"`` is mandated by
            ``sycl_ext_oneapi_async_memory_alloc``) are silently
            skipped; a per-kind ``RuntimeWarning`` is emitted so the
            user can see which kinds were not installed.

    Returns:
        Dict[str, MemoryPool]: mapping from installed USM kind to its
        pool wrapper. Kinds that were skipped are absent.

    A subsequent :func:`reset_allocator` call (with no kwargs) clears
    every hook installed by this function.
    """
    import warnings

    installed = {}
    for kind in usm_types:
        try:
            pool = MemoryPool.get_default(sycl_queue=sycl_queue, usm_type=kind)
        except RuntimeError as e:
            warnings.warn(
                f"Could not install default pool for usm_type={kind!r}: "
                f"{e}. The SYCL runtime may not support pooled "
                f"allocations for this kind yet.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        set_allocator(pool)
        installed[kind] = pool
    return installed


__all__ = [
    "MemoryPool",
    "MemoryUSMDevice",
    "MemoryUSMHost",
    "MemoryUSMShared",
    "USMAllocationError",
    "as_usm_memory",
    "get_allocator",
    "is_memory_pool_available",
    "malloc_device",
    "malloc_host",
    "malloc_shared",
    "reset_allocator",
    "set_allocator",
    "use_default_pool",
]
