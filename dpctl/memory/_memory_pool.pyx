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

import threading
import weakref

import dpctl

from dpctl._backend cimport (  # noqa: E211
    DPCTLMemoryPool_AsyncFreeOnQueue,
    DPCTLMemoryPool_Available,
    DPCTLMemoryPool_Create,
    DPCTLMemoryPool_CreateDefault,
    DPCTLMemoryPool_Delete,
    DPCTLMemoryPool_GetInstalled,
    DPCTLMemoryPool_GetReservedBytes,
    DPCTLMemoryPool_GetUsedBytes,
    DPCTLMemoryPool_IsDefault,
    DPCTLMemoryPool_MallocOnQueue,
    DPCTLMemoryPool_ResetMemory,
    DPCTLMemoryPool_SetInstalled,
    DPCTLMemoryPool_SetReleaseThreshold,
    DPCTLSyclMemoryPoolRef,
    DPCTLSyclQueueRef,
    DPCTLSyclUSMRef,
    _usm_type,
)
from dpctl._sycl_context cimport SyclContext
from dpctl._sycl_device cimport SyclDevice
from dpctl._sycl_queue cimport SyclQueue
from dpctl._sycl_queue_manager cimport get_device_cached_queue
from dpctl.memory._memory cimport (
    MemoryUSMDevice,
    MemoryUSMHost,
    MemoryUSMShared,
    _Memory,
)


__all__ = ["MemoryPool", "is_memory_pool_available"]


def _install_device_pool(MemoryPool pool not None):
    """Register ``pool`` as the installed device-USM pool for its
    ``(context, device)`` in the C-side registry consumed by C++
    callers (e.g. dpnp's ``smart_malloc_*``). No-op when ``pool`` is
    not a USM-device pool."""
    cdef SyclContext ctx
    cdef SyclDevice dev
    if pool._usm_type != "device":
        return
    ctx = <SyclContext>pool._queue.sycl_context
    dev = <SyclDevice>pool._queue.sycl_device
    DPCTLMemoryPool_SetInstalled(
        ctx.get_context_ref(), dev.get_device_ref(), pool._pool_ref
    )


def _uninstall_device_pool(SyclContext ctx not None, SyclDevice dev not None):
    """Clear the C-side device-USM pool registry entry for
    ``(ctx, dev)``."""
    DPCTLMemoryPool_SetInstalled(
        ctx.get_context_ref(), dev.get_device_ref(), NULL
    )


def _get_installed_device_pool_ptr(SyclContext ctx not None,
                                   SyclDevice dev not None):
    """Return the raw ``DPCTLSyclMemoryPoolRef`` (as a Python int) for
    the installed device-USM pool, or 0 if no pool is installed. Used
    by the tests; user code should not need this."""
    return <size_t>DPCTLMemoryPool_GetInstalled(
        ctx.get_context_ref(), dev.get_device_ref()
    )


# Key: (context_hash, device_hash, usm_kind_str). Lookup is protected
# by ``_default_pool_lock`` to make get_default a strict singleton.
_default_pool_cache = weakref.WeakValueDictionary()
_default_pool_lock = threading.Lock()

# Sentinel passed via ``MemoryPool(sycl_queue=_UNINIT)`` to skip pool
# construction in ``__cinit__``. Used by ``get_default`` to avoid
# wasting a SYCL pool object on the cache-miss path.
_UNINIT = object()


def is_memory_pool_available():
    """Return ``True`` if libsyclinterface was built against a DPC++
    that defines the ``SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC`` feature-
    test macro (``sycl_ext_oneapi_async_memory_alloc``), ``False``
    otherwise.

    When ``False`` :class:`MemoryPool` still works, but allocations
    go through plain ``sycl::malloc_*`` and frees through synchronous
    ``sycl::free`` — no pool-side caching or stream-ordered free.

    Note: a ``True`` here is a compile-time signal only. The device
    must also report the ``ext_oneapi_async_memory_alloc`` aspect at
    runtime for the pool fast path to be exercised; without the
    aspect the underlying ``memory_pool`` constructor raises
    ``errc::feature_not_supported``.
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

    The pool object is itself callable with the allocator-hook
    signature ``(nbytes, sycl_queue) -> _Memory``, so it can be
    handed directly to :func:`dpctl.memory.set_allocator`::

        pool = dpctl.memory.MemoryPool.get_default(usm_type="device")
        dpctl.memory.set_allocator(pool)

    Args:
        sycl_queue (Optional[:class:`dpctl.SyclQueue`]):
            Queue whose device and context the pool is bound to. If
            ``None``, a cached default-constructed queue is used.
        usm_type (str):
            One of ``"device"`` (default), ``"shared"``, or ``"host"``.
    """

    def __cinit__(self, *, sycl_queue=None, str usm_type="device"):
        cdef _usm_type kind
        cdef DPCTLSyclMemoryPoolRef pref = NULL
        self._pool_ref = NULL
        self._queue = None
        self._usm_type = None

        if sycl_queue is _UNINIT:
            # Deferred init: caller (e.g. get_default) populates fields.
            return

        if sycl_queue is None:
            sycl_queue = get_device_cached_queue(dpctl.SyclDevice())
        elif not isinstance(sycl_queue, SyclQueue):
            raise TypeError(
                "sycl_queue must be a dpctl.SyclQueue instance or None; "
                f"got {type(sycl_queue).__name__}"
            )

        kind = _usm_type_str_to_enum(usm_type)

        pref = DPCTLMemoryPool_Create(
            (<SyclQueue>sycl_queue).get_queue_ref(), kind
        )
        if pref is NULL:
            raise RuntimeError(
                f"Failed to create SYCL memory pool for usm_type={usm_type!r}"
            )
        self._pool_ref = pref
        self._queue = <SyclQueue>sycl_queue
        self._usm_type = usm_type

    def __dealloc__(self):
        cdef SyclContext ctx
        cdef SyclDevice dev
        # If this pool is currently the installed device-USM pool for
        # its (context, device), clear the C-side registry entry
        # before deleting our handle. Defensive: the standard contract
        # is that the user calls reset_allocator() first.
        if self._pool_ref is not NULL and self._usm_type == "device" \
                and self._queue is not None:
            ctx = <SyclContext>self._queue.sycl_context
            dev = <SyclDevice>self._queue.sycl_device
            if DPCTLMemoryPool_GetInstalled(
                ctx.get_context_ref(), dev.get_device_ref()
            ) == self._pool_ref:
                DPCTLMemoryPool_SetInstalled(
                    ctx.get_context_ref(), dev.get_device_ref(), NULL
                )
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

        # Validate the USM type string up-front so we never grow the
        # cache on a bad argument.
        kind = _usm_type_str_to_enum(usm_type)

        cache_key = (
            hash(sycl_queue.sycl_context),
            hash(sycl_queue.sycl_device),
            usm_type,
        )
        # Double-checked lookup under the lock so that concurrent
        # callers strictly share one wrapper per cache key.
        with _default_pool_lock:
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

            obj = MemoryPool(sycl_queue=_UNINIT)
            obj._pool_ref = pref
            obj._queue = sycl_queue
            obj._usm_type = usm_type
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

    def __call__(self, Py_ssize_t nbytes, SyclQueue sycl_queue=None):
        """Allocator-hook protocol: equivalent to :meth:`malloc`. Lets
        users write ``dpctl.memory.set_allocator(pool)`` instead of
        ``set_allocator(pool.malloc)``.
        """
        return self.malloc(nbytes, sycl_queue)

    def malloc(self, Py_ssize_t nbytes, SyclQueue sycl_queue=None):
        """Allocate ``nbytes`` bytes from the pool and return a
        :class:`_Memory` instance.

        When ``sycl_queue`` is provided it must share the pool's SYCL
        context. The returned allocation is stream-ordered against
        ``sycl_queue``; the eventual pool-return is also ordered
        against ``sycl_queue``, preserving lifetime safety relative to
        the work that consumed the allocation. When ``sycl_queue`` is
        ``None`` the pool's bound queue is used.
        """
        cdef DPCTLSyclUSMRef p = NULL
        cdef DPCTLSyclQueueRef alloc_qref
        cdef DPCTLSyclMemoryPoolRef pool_ref
        cdef _Memory base
        cdef SyclQueue alloc_q

        if nbytes <= 0:
            raise ValueError(
                "Number of bytes for pool allocation must be positive"
            )
        if sycl_queue is None:
            alloc_q = self._queue
        else:
            if sycl_queue.sycl_context != self._queue.sycl_context:
                raise ValueError(
                    "sycl_queue passed to MemoryPool.malloc must share "
                    "the pool's SYCL context"
                )
            alloc_q = sycl_queue

        # Extract raw refs before releasing the GIL; the underlying
        # SYCL objects remain valid because ``alloc_q`` and ``self``
        # are held as Python locals.
        alloc_qref = alloc_q.get_queue_ref()
        pool_ref = self._pool_ref

        with nogil:
            p = DPCTLMemoryPool_MallocOnQueue(
                pool_ref, alloc_qref, <size_t>nbytes
            )
        if p is NULL:
            from dpctl.memory._memory import USMAllocationError
            raise USMAllocationError(
                f"Pool allocation of {nbytes} bytes failed"
            )

        base = _Memory.__new__(_Memory)
        try:
            base._cinit_from_pool(
                nbytes,
                alloc_q,
                self,
                pool_ref,
                p,
            )
        except Exception:
            DPCTLMemoryPool_AsyncFreeOnQueue(pool_ref, alloc_qref, p)
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
        """Best-effort eviction of unused cached blocks.

        Drives the pool's release threshold to 0 (asking the runtime
        to release the cache at its next opportunity) and synchronizes
        the pool's bound queue so that any in-flight stream-ordered
        frees are flushed before returning. The SYCL runtime decides
        when to actually release blocks back to the driver, and only
        blocks not currently in use can be reclaimed. No-op when
        :func:`is_memory_pool_available` returns ``False``.
        """
        if not is_memory_pool_available():
            return
        DPCTLMemoryPool_ResetMemory(self._pool_ref)
        if self._queue is not None:
            self._queue.wait()

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
