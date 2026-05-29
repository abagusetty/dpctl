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

"""Pool-backed USM allocator for :mod:`dpctl.memory`."""

import weakref

import dpctl

from dpctl._backend cimport (  # noqa: E211
    DPCTLMemoryPool_AsyncFree,
    DPCTLMemoryPool_Available,
    DPCTLMemoryPool_Create,
    DPCTLMemoryPool_CreateDefault,
    DPCTLMemoryPool_Delete,
    DPCTLMemoryPool_GetReservedBytes,
    DPCTLMemoryPool_GetUsedBytes,
    DPCTLMemoryPool_IsDefault,
    DPCTLMemoryPool_Malloc,
    DPCTLMemoryPool_ResetMemory,
    DPCTLMemoryPool_SetReleaseThreshold,
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


# Key: (context_hash, device_hash, usm_kind_str).
_default_pool_cache = weakref.WeakValueDictionary()


def is_memory_pool_available():
    """Return ``True`` if libsyclinterface was built against a DPC++
    that exposes the ``sycl_ext_oneapi_memory_pool`` and
    ``sycl_ext_oneapi_async_alloc`` extensions, ``False`` otherwise.

    When ``False`` :class:`MemoryPool` still works, but allocations
    go through plain ``sycl::malloc_*`` and frees through synchronous
    ``sycl::free``.
    """
    return bool(DPCTLMemoryPool_Available())


cdef inline _usm_type _usm_type_str_to_enum(str usm_type):
    if usm_type == "device":
        return _usm_type._USM_DEVICE
    elif usm_type == "shared":
        return _usm_type._USM_SHARED
    elif usm_type == "host":
        return _usm_type._USM_HOST
    raise ValueError(
        "usm_type must be one of 'device', 'shared', 'host'; "
        f"got {usm_type!r}"
    )


cdef class MemoryPool:
    """MemoryPool(sycl_queue=None, usm_type='device')

    A SYCL memory pool bound to a queue's device/context.

    Construction modes:

    * ``MemoryPool(sycl_queue=q, usm_type=...)`` constructs a private
      pool with an isolated cache.
    * :meth:`MemoryPool.get_default` returns a wrapper around the SYCL
      runtime's default pool for the given ``(context, device,
      usm_type)``; all callers within the process share its cache.

    :meth:`malloc` has the signature expected by the allocator hook
    (``(nbytes, sycl_queue) -> _Memory``) and is the intended argument
    to :func:`dpctl.memory.set_allocator`.

    Args:
        sycl_queue (Optional[:class:`dpctl.SyclQueue`]):
            Queue whose device and context the pool is bound to. If
            ``None``, a cached default-constructed queue is used.
        usm_type (str):
            One of ``"device"`` (default), ``"shared"``, or ``"host"``.
    """

    def __cinit__(self, *, SyclQueue sycl_queue=None, str usm_type="device"):
        cdef _usm_type kind
        cdef DPCTLSyclMemoryPoolRef pref = NULL
        self._pool_ref = NULL
        self._queue = None
        self._usm_type = None

        if sycl_queue is None:
            sycl_queue = get_device_cached_queue(dpctl.SyclDevice())

        kind = _usm_type_str_to_enum(usm_type)

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

    @staticmethod
    def get_default(*, SyclQueue sycl_queue=None, str usm_type="device"):
        """Return a :class:`MemoryPool` wrapper around the SYCL
        runtime's default memory pool for the given
        ``(context, device, usm_type)``.

        Repeated calls with equivalent arguments return the same
        Python wrapper object (compared with ``is``).
        """
        cdef _usm_type kind
        cdef DPCTLSyclMemoryPoolRef pref = NULL
        cdef MemoryPool obj

        if sycl_queue is None:
            sycl_queue = get_device_cached_queue(dpctl.SyclDevice())

        kind = _usm_type_str_to_enum(usm_type)

        cache_key = (
            hash(sycl_queue.sycl_context),
            hash(sycl_queue.sycl_device),
            usm_type,
        )
        # Benignly racy: two concurrent misses produce two wrappers
        # both pointing at the same runtime-owned default pool.
        existing = _default_pool_cache.get(cache_key)
        if existing is not None:
            return existing

        pref = DPCTLMemoryPool_CreateDefault(
            sycl_queue.get_queue_ref(), kind
        )
        if pref is NULL:
            raise RuntimeError(
                "Failed to obtain SYCL default memory pool for "
                f"usm_type={usm_type!r}"
            )

        # __cinit__ always builds a private pool; swap in the
        # default-pool handle. The throwaway private pool is built at
        # most once per (context, device, usm_type) cache miss.
        obj = MemoryPool(sycl_queue=sycl_queue, usm_type=usm_type)
        if obj._pool_ref is not NULL:
            DPCTLMemoryPool_Delete(obj._pool_ref)
        obj._pool_ref = pref
        _default_pool_cache[cache_key] = obj
        return obj

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
        """The USM allocation kind this pool serves."""
        return self._usm_type

    @property
    def is_default(self):
        """``True`` when this wrapper refers to the SYCL runtime's
        default pool (constructed via :meth:`get_default`)."""
        if self._pool_ref is NULL:
            return False
        return bool(DPCTLMemoryPool_IsDefault(self._pool_ref))

    def malloc(self, Py_ssize_t nbytes, SyclQueue sycl_queue=None):
        """Allocate ``nbytes`` bytes from the pool and return a
        :class:`_Memory` instance.

        Args:
            nbytes: Number of bytes to allocate. Must be positive.
            sycl_queue: Queue for ordering of the allocation/free.
                Defaults to the queue the pool was constructed with.
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

        base = _Memory.__new__(_Memory)
        try:
            base._cinit_from_pool(
                nbytes,
                sycl_queue,
                self,
                self._pool_ref,
                p,
            )
        except Exception:
            DPCTLMemoryPool_AsyncFree(self._pool_ref, p)
            raise

        if self._usm_type == "device":
            return MemoryUSMDevice(base)
        elif self._usm_type == "shared":
            return MemoryUSMShared(base)
        else:
            return MemoryUSMHost(base)

    def set_release_threshold(self, size_t threshold):
        """Raise the pool's release threshold in bytes. Below the
        threshold the implementation retains cached blocks; above it,
        excess may be released back to the driver. Monotonic — can
        only be raised, not lowered. No-op when
        :func:`is_memory_pool_available` returns ``False``.
        """
        DPCTLMemoryPool_SetReleaseThreshold(self._pool_ref, threshold)

    def reset_memory(self):
        """Attempt to release all currently-cached, unused blocks
        back to the underlying memory provider. Best-effort: the
        runtime decides when and how much to release. No-op when
        :func:`is_memory_pool_available` returns ``False``.
        """
        DPCTLMemoryPool_ResetMemory(self._pool_ref)

    def used_bytes(self):
        """Number of bytes currently handed out by the pool to live
        allocations. Returns ``0`` when
        :func:`is_memory_pool_available` returns ``False``.
        """
        return int(DPCTLMemoryPool_GetUsedBytes(self._pool_ref))

    def total_bytes(self):
        """Total bytes the pool has reserved (used + free).
        Returns ``0`` when :func:`is_memory_pool_available` returns
        ``False``.
        """
        return int(DPCTLMemoryPool_GetReservedBytes(self._pool_ref))

    def free_bytes(self):
        """Number of bytes currently cached and not handed out.
        Computed as ``total_bytes() - used_bytes()``.
        """
        cdef size_t reserved = DPCTLMemoryPool_GetReservedBytes(
            self._pool_ref
        )
        cdef size_t used = DPCTLMemoryPool_GetUsedBytes(self._pool_ref)
        if reserved < used:
            return 0
        return int(reserved - used)
