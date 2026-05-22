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

# distutils: language = c++
# cython: language_level=3

"""Pool-backed USM allocator for :mod:`dpctl.memory`.

The :class:`MemoryPool` class is the recommended way to install a
stream-ordered, reuse-friendly USM allocator via
:func:`dpctl.memory.set_allocator`. When the underlying SYCL
``sycl_ext_oneapi_memory_pool`` extension is available, allocations and
frees are stream-ordered against the queue the pool was constructed
with, providing the same lifetime-safety guarantees as CuPy's
``MemoryAsyncPool``. When the extension is not available the pool falls
back to a transparent ``sycl::malloc_* / sycl::free`` implementation,
preserving correctness at the cost of the stream-ordering optimization.
"""

import dpctl

from dpctl._backend cimport (  # noqa: E211
    DPCTLMemoryPool_AsyncFree,
    DPCTLMemoryPool_Available,
    DPCTLMemoryPool_Create,
    DPCTLMemoryPool_Delete,
    DPCTLMemoryPool_Malloc,
    DPCTLMemoryPool_TrimTo,
    DPCTLSyclMemoryPoolRef,
    DPCTLSyclUSMRef,
    _usm_type,
)
from dpctl._sycl_queue cimport SyclQueue
from dpctl._sycl_queue_manager cimport get_device_cached_queue
from dpctl.memory._memory cimport (
    MemoryUSMDevice,
    MemoryUSMHost,
    MemoryUSMShared,
    _Memory,
)


__all__ = ["MemoryPool", "is_memory_pool_available"]


def is_memory_pool_available():
    """Return ``True`` if libsyclinterface was built against a DPC++
    that exposes the ``sycl_ext_oneapi_memory_pool`` and
    ``sycl_ext_oneapi_async_alloc`` extensions, ``False`` otherwise.

    When this returns ``False`` :class:`MemoryPool` still functions, but
    its allocations go through plain ``sycl::malloc_*`` and its frees
    through synchronous ``sycl::free`` on the pool's context. The pool
    therefore still provides a single, opt-in allocator surface for
    :func:`dpctl.memory.set_allocator`, but no stream-ordered free or
    cache-reuse optimization.
    """
    return bool(DPCTLMemoryPool_Available())


cdef class MemoryPool:
    """MemoryPool(sycl_queue=None, usm_type='device')

    A SYCL memory pool bound to a queue's device/context.

    The pool is the recommended building block for installing a custom
    allocator via :func:`dpctl.memory.set_allocator`. Its
    :meth:`MemoryPool.malloc` method has the exact signature expected
    by the allocator hook (``(nbytes, sycl_queue) -> _Memory``).

    Args:
        sycl_queue (Optional[:class:`dpctl.SyclQueue`]):
            The queue whose device and context the pool is bound to.
            All allocations served by this pool are valid in that
            context and may be touched by that device. If ``None``, a
            cached default-constructed queue is used.
        usm_type (str):
            One of ``"device"`` (default), ``"shared"``, or ``"host"``.
            Selects which kind of USM allocation this pool serves.

    Example::

        import dpctl
        import dpctl.memory as dpm

        q = dpctl.SyclQueue()
        pool = dpm.MemoryPool(sycl_queue=q, usm_type="device")
        dpm.set_allocator(pool.malloc, usm_type="device",
                          sycl_device=q.sycl_device)

        # All subsequent USM-device allocations on this device are
        # served from ``pool``:
        m = dpm.MemoryUSMDevice(1 << 20, queue=q)
    """

    def __cinit__(self, *, SyclQueue sycl_queue=None, str usm_type="device"):
        cdef _usm_type kind
        cdef DPCTLSyclMemoryPoolRef pref = NULL
        self._pool_ref = NULL
        self._queue = None
        self._usm_type = None

        if usm_type not in ("device", "shared", "host"):
            raise ValueError(
                "usm_type must be one of 'device', 'shared', 'host'; "
                f"got {usm_type!r}"
            )

        if sycl_queue is None:
            sycl_queue = get_device_cached_queue(dpctl.SyclDevice())

        if usm_type == "device":
            kind = _usm_type._USM_DEVICE
        elif usm_type == "shared":
            kind = _usm_type._USM_SHARED
        else:
            kind = _usm_type._USM_HOST

        pref = DPCTLMemoryPool_Create(sycl_queue.get_queue_ref(), kind)
        if pref is NULL:
            raise RuntimeError(
                f"Failed to create SYCL memory pool for usm_type={usm_type!r}"
            )
        self._pool_ref = pref
        self._queue = sycl_queue
        self._usm_type = usm_type

    def __dealloc__(self):
        if self._pool_ref is not NULL:
            DPCTLMemoryPool_Delete(self._pool_ref)
            self._pool_ref = NULL

    @property
    def sycl_queue(self):
        """The :class:`dpctl.SyclQueue` this pool is bound to."""
        return self._queue

    @property
    def sycl_device(self):
        """The :class:`dpctl.SyclDevice` of the pool's queue."""
        return self._queue.sycl_device

    @property
    def sycl_context(self):
        """The :class:`dpctl.SyclContext` of the pool's queue."""
        return self._queue.sycl_context

    @property
    def usm_type(self):
        """The USM allocation kind this pool serves (``'device'``,
        ``'shared'``, or ``'host'``)."""
        return self._usm_type

    def malloc(self, Py_ssize_t nbytes, SyclQueue sycl_queue=None):
        """Allocate ``nbytes`` bytes from the pool and return a
        :class:`_Memory` instance (concrete subclass matching
        :attr:`usm_type`) that owns the allocation.

        When the returned ``_Memory`` object is destroyed, the allocation
        is returned to the pool (stream-ordered against ``sycl_queue``
        when the SYCL extension is available, synchronous on the pool's
        context otherwise).

        Args:
            nbytes: Number of bytes to allocate. Must be positive.
            sycl_queue: Queue used for ordering of the allocation/free
                operations relative to other work. If ``None``, defaults
                to the queue the pool was constructed with. Must share
                the same SYCL context as the pool's queue.

        Returns:
            One of :class:`MemoryUSMDevice`, :class:`MemoryUSMShared`,
            or :class:`MemoryUSMHost` depending on :attr:`usm_type`.
        """
        cdef DPCTLSyclUSMRef p = NULL
        cdef _Memory base

        if nbytes <= 0:
            raise ValueError(
                "Number of bytes for pool allocation must be positive"
            )
        if sycl_queue is None:
            sycl_queue = self._queue

        with nogil:
            p = DPCTLMemoryPool_Malloc(self._pool_ref, <size_t>nbytes)
        if p is NULL:
            from dpctl.memory._memory import USMAllocationError
            raise USMAllocationError(
                f"Pool allocation of {nbytes} bytes failed"
            )

        # Construct a bare _Memory and stamp it with the pool-return
        # bookkeeping; then wrap it in the appropriate USM-typed subclass
        # via the (bare _Memory) -> subclass constructor path. The
        # subclass's ``__cinit__`` -> ``_cinit_other`` branch recognizes
        # the pool-owned source and keeps it alive via ``refobj``,
        # ensuring the eventual ``__dealloc__`` of the returned object
        # invokes the pool-return callback.
        base = _Memory.__new__(_Memory)
        try:
            base._cinit_from_pool(
                nbytes,
                sycl_queue,
                self,  # pool_owner: keeps the pool alive while allocation lives
                self._pool_ref,
                p,
            )
        except Exception:
            # Bookkeeping failed before ownership of ``p`` was handed to
            # the callback; release manually to avoid leaking the USM
            # allocation.
            DPCTLMemoryPool_AsyncFree(self._pool_ref, p)
            raise

        if self._usm_type == "device":
            return MemoryUSMDevice(base)
        elif self._usm_type == "shared":
            return MemoryUSMShared(base)
        else:
            return MemoryUSMHost(base)

    def trim_to(self, size_t min_bytes_to_keep):
        """Hint the pool to release cached blocks back to the driver,
        leaving at most ``min_bytes_to_keep`` bytes in its internal
        cache. No-op when :func:`is_memory_pool_available` returns
        ``False``.
        """
        DPCTLMemoryPool_TrimTo(self._pool_ref, min_bytes_to_keep)
