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

import weakref

import dpctl

from dpctl._backend cimport (  # noqa: E211
    DPCTLMemoryPool_AsyncFree,
    DPCTLMemoryPool_Available,
    DPCTLMemoryPool_Create,
    DPCTLMemoryPool_CreateDefault,
    DPCTLMemoryPool_Delete,
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


# ---------------------------------------------------------------------------
# Default-pool wrapper cache
# ---------------------------------------------------------------------------
# The SYCL runtime's default memory pool is itself a singleton per
# ``(context, device, usm_kind)`` tuple. We additionally cache the
# *Python* wrapper around it so that two calls to
# ``MemoryPool.get_default`` with equivalent arguments return ``is``-
# identical Python objects. The cache uses weak references so that a
# wrapper can still be collected once no user code holds it (its
# entries do not pin the SyclQueue / SyclContext alive indefinitely).
#
# Key shape: (context_hash, device_hash, usm_kind_str)
#
_default_pool_cache = weakref.WeakValueDictionary()


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

    Two construction modes are supported:

    * ``MemoryPool(sycl_queue=q, usm_type=...)`` constructs a **private**
      pool with an isolated cache. Use this when you need pool-level
      isolation (memory budget enforcement, debugging fragmentation,
      multi-tenancy within a single process).

    * :meth:`MemoryPool.get_default` returns a wrapper around the SYCL
      runtime's **default** pool for the given ``(context, device,
      usm_type)``. The underlying pool is a runtime-managed singleton:
      all callers within the process share its cache. This is the
      recommended mode for libraries that want to amortize allocation
      cost across the whole application.

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

        # Process-wide shared default pool (recommended for most uses):
        pool = dpm.MemoryPool.get_default(sycl_queue=q, usm_type="device")

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

        The underlying pool is a process-wide singleton managed by the
        SYCL runtime: multiple calls with the same context+device+
        usm_type share the same cache, regardless of which library
        within the process initiated them. This is the recommended
        choice for production use, because it allows dpnp,
        ``gpu4pyscf``, and any other SYCL-using component in the same
        process to amortize allocation cost together.

        Within a single dpctl process, repeated calls with equivalent
        arguments return the *same Python wrapper object* (compared
        with ``is``), backed by a process-local
        :class:`weakref.WeakValueDictionary`. The cache entry is
        cleared when the last user reference to the wrapper is
        dropped.

        Multi-process note: each OS process has its own SYCL runtime
        and therefore its own default pool. There is no cross-process
        coordination; pool reservations are per-process.

        Args:
            sycl_queue: Queue selecting the ``(context, device)`` tuple.
            usm_type: USM allocation kind (``'device'``, ``'shared'``,
                or ``'host'``).
        """
        cdef _usm_type kind
        cdef DPCTLSyclMemoryPoolRef pref = NULL
        cdef MemoryPool obj

        if sycl_queue is None:
            sycl_queue = get_device_cached_queue(dpctl.SyclDevice())

        # ``usm_type_str_to_enum`` validates the string before we
        # touch the cache.
        kind = _usm_type_str_to_enum(usm_type)

        cache_key = (
            hash(sycl_queue.sycl_context),
            hash(sycl_queue.sycl_device),
            usm_type,
        )
        # Thread-safety note: this lookup-then-insert is racy, but
        # benignly so. If two threads miss simultaneously they'll both
        # construct a wrapper; the second insert wins. Both wrappers
        # point at the same runtime-owned default pool, so cache state
        # is shared regardless of which Python object the caller ends
        # up with. The cache is best-effort dedup, not exclusion.
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

        # Construct a MemoryPool instance and then replace its private
        # pool with the default-pool handle. Cython runs __cinit__
        # unconditionally on object creation, so the cheapest correct
        # path is to let __cinit__ build a (small, immediately
        # discarded) private pool, then swap in the default-pool
        # handle. The wasted private-pool construction happens at most
        # once per unique (context, device, usm_type) cache miss; cache
        # hits skip this path entirely.
        obj = MemoryPool(sycl_queue=sycl_queue, usm_type=usm_type)
        if obj._pool_ref is not NULL:
            DPCTLMemoryPool_Delete(obj._pool_ref)
        obj._pool_ref = pref
        # ``_queue`` and ``_usm_type`` are already correct from __cinit__.
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
        """The USM allocation kind this pool serves (``'device'``,
        ``'shared'``, or ``'host'``)."""
        return self._usm_type

    @property
    def is_default(self):
        """``True`` when this wrapper refers to the SYCL runtime's
        default pool for its ``(context, device, usm_type)`` tuple
        (i.e. constructed via :meth:`get_default`); ``False`` when it
        wraps a private pool constructed via the regular constructor.
        """
        if self._pool_ref is NULL:
            return False
        return bool(DPCTLMemoryPool_IsDefault(self._pool_ref))

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

    def set_release_threshold(self, size_t threshold):
        """Raise the pool's release threshold — the lower bound (in
        bytes) on the size of the cache below which the implementation
        should not release blocks back to the underlying memory
        provider.

        This is **not** a "release the cache now" operation; for that,
        use :meth:`reset_memory`. The release threshold is a *policy*
        for future release decisions:

        * ``cached_bytes <= threshold``: implementation retains the
          cache for cheap reuse.
        * ``cached_bytes >  threshold``: implementation may release
          the excess back to the driver.

        The CUDA analog is the
        ``cudaMemPoolAttrReleaseThreshold`` attribute.

        Note: per the underlying SYCL extension, this setter is
        *monotonic* — it can only raise the threshold, not lower it. To
        shrink the retained cache, use :meth:`reset_memory` and/or
        construct a fresh pool.

        No-op when :func:`is_memory_pool_available` returns ``False``.
        """
        DPCTLMemoryPool_SetReleaseThreshold(self._pool_ref, threshold)

    def reset_memory(self):
        """Attempt to immediately release all currently-cached, unused
        blocks back to the underlying memory provider. Blocks still in
        use (handed out to live allocations) are untouched.

        Note that ``reset_memory()`` on the *default* pool affects
        every component within the process that shares it. Use with
        care in shared-pool scenarios.

        No-op when :func:`is_memory_pool_available` returns ``False``.
        """
        DPCTLMemoryPool_ResetMemory(self._pool_ref)
