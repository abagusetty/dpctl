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
