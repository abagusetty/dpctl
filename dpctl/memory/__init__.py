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
:class:`MemoryPool` implementation, allowing users to opt into pooled
USM allocation in the style of CuPy's memory pool. When no allocator is
installed (the default), behavior is bit-for-bit identical to earlier
dpctl releases: every allocation calls ``sycl::malloc_*`` and every
deallocation calls ``sycl::free``.
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
    installed allocator hook.

    This is the explicit "go directly to ``sycl::malloc_device``" entry
    point, intended for cases like::

        threshold = 64 << 20
        def hybrid(nbytes, queue):
            if nbytes >= threshold:
                return dpm.malloc_device(nbytes, queue=queue)
            return pool.malloc(nbytes, queue)
        dpm.set_allocator(hybrid, usm_type="device", sycl_device=dev)

    where the user wants to route most allocations through a pool but
    keep huge ones out of it.

    The returned :class:`MemoryUSMDevice` does **not** participate in
    any pool; its destruction calls ``sycl::free`` directly, exactly as
    if no allocator were installed.
    """
    return _direct_alloc(MemoryUSMDevice, nbytes, queue)


def malloc_shared(nbytes, queue=None):
    """Allocate ``nbytes`` bytes of USM-shared memory bypassing any
    installed allocator hook. See :func:`malloc_device` for rationale.
    """
    return _direct_alloc(MemoryUSMShared, nbytes, queue)


def malloc_host(nbytes, queue=None):
    """Allocate ``nbytes`` bytes of USM-host memory bypassing any
    installed allocator hook. See :func:`malloc_device` for rationale.
    """
    return _direct_alloc(MemoryUSMHost, nbytes, queue)


def _direct_alloc(cls, nbytes, queue):
    """Construct a ``MemoryUSM*`` instance while temporarily suspending
    the allocator-hook lookup on the current thread, ensuring the
    allocation goes straight to ``sycl::malloc_*``.

    Implemented via a thread-local bypass flag (see
    :class:`dpctl.memory._allocator._bypass_hooks`) so that concurrent
    allocations on other threads continue to use their installed hooks.
    """
    import dpctl

    from ._allocator import _bypass_hooks

    if queue is None:
        queue = dpctl.SyclQueue()
    with _bypass_hooks():
        return cls(nbytes, queue=queue)


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
]
