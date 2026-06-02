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

"""USM-device memory pool for :mod:`dpctl.memory`.

A :class:`MemoryPool` is identified by its ``(context, device)`` pair.
Only USM-device allocations are pooled — the underlying
``sycl_ext_oneapi_async_memory_alloc`` extension does not currently
support pooled shared / host allocations.

The queue used at allocation or free time is supplied per-call and
mirrors the contract of ``DPCTLmalloc_device(size, QRef)`` and
``DPCTLfree_with_queue(MRef, QRef)``.
"""

import threading
import weakref

import dpctl

from dpctl._backend cimport (  # noqa: E211
    DPCTLMemoryPool_AsyncFree,
    DPCTLMemoryPool_Available,
    DPCTLMemoryPool_Create,
    DPCTLMemoryPool_CreateDefault,
    DPCTLMemoryPool_Delete,
    DPCTLMemoryPool_GetInstalled,
    DPCTLMemoryPool_GetReservedBytes,
    DPCTLMemoryPool_GetUsedBytes,
    DPCTLMemoryPool_IsDefault,
    DPCTLMemoryPool_Malloc,
    DPCTLMemoryPool_ResetMemory,
    DPCTLMemoryPool_SetInstalled,
    DPCTLMemoryPool_SetReleaseThreshold,
    DPCTLSyclContextRef,
    DPCTLSyclDeviceRef,
    DPCTLSyclMemoryPoolRef,
    DPCTLSyclQueueRef,
    DPCTLSyclUSMRef,
)
from dpctl._sycl_context cimport SyclContext
from dpctl._sycl_device cimport SyclDevice
from dpctl._sycl_queue cimport SyclQueue
from dpctl._sycl_queue_manager cimport get_device_cached_queue
from dpctl.memory._memory cimport MemoryUSMDevice, _Memory


__all__ = ["MemoryPool", "is_memory_pool_available"]


def _install_device_pool(MemoryPool pool not None):
    """Register ``pool`` as the installed USM-device pool for its
    ``(context, device)`` in the C-side registry consumed by C++
    callers (e.g. dpnp's ``smart_malloc_device``)."""
    DPCTLMemoryPool_SetInstalled(
        pool._context.get_context_ref(),
        pool._device.get_device_ref(),
        pool._pool_ref,
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


# Key: (context_hash, device_hash). Lookup protected by
# ``_default_pool_lock`` to make get_default a strict singleton.
_default_pool_cache = weakref.WeakValueDictionary()
_default_pool_lock = threading.Lock()

# Sentinel passed via ``MemoryPool(sycl_device=_UNINIT)`` to skip pool
# construction in ``__cinit__``. Used by ``get_default`` to avoid
# wasting a SYCL pool object on the cache-miss path.
_UNINIT = object()


def is_memory_pool_available():
    """Return ``True`` if libsyclinterface was built against a DPC++
    that defines the ``SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC`` feature-
    test macro (``sycl_ext_oneapi_async_memory_alloc``), ``False``
    otherwise.

    When ``False`` :class:`MemoryPool` still works, but allocations
    go through plain ``sycl::malloc_device`` and frees through
    synchronous ``sycl::free`` — no pool-side caching or stream-
    ordered free.

    Note: a ``True`` here is a compile-time signal only. The device
    must also report the ``ext_oneapi_async_memory_alloc`` aspect at
    runtime for the pool fast path to be exercised; without the
    aspect the underlying ``memory_pool`` constructor raises
    ``errc::feature_not_supported``.
    """
    return bool(DPCTLMemoryPool_Available())


cdef inline SyclContext _default_context_for(SyclDevice dev):
    """Return the dpctl-cached default context for ``dev`` (the one
    every ``dpctl.SyclQueue(dev)`` shares)."""
    cdef SyclQueue q = get_device_cached_queue(dev)
    return <SyclContext>q.sycl_context


cdef class MemoryPool:
    """MemoryPool(sycl_device=None)

    A SYCL USM-device memory pool identified by its ``(context,
    device)`` pair. The queue used for allocation / free is supplied
    per-call (see :meth:`malloc`) and must share the pool's context.

    Construction modes:

    * ``MemoryPool(sycl_device=dev)`` constructs a private pool with
      an isolated cache for ``dev`` (and ``dev``'s dpctl-cached
      default context).
    * :meth:`MemoryPool.get_default` returns a wrapper around the SYCL
      runtime's default pool for the given ``(context, device)``; all
      callers within the process share its cache.

    The pool object is itself callable with the allocator-hook
    signature ``(nbytes, sycl_queue) -> MemoryUSMDevice``, so it can
    be handed directly to :func:`dpctl.memory.set_allocator`::

        pool = dpctl.memory.MemoryPool.get_default()
        dpctl.memory.set_allocator(pool)

    Args:
        sycl_device (Optional[:class:`dpctl.SyclDevice`]):
            Device the pool serves allocations for. If ``None``, a
            default-constructed :class:`dpctl.SyclDevice` is used.
    """

    def __cinit__(self, *, sycl_device=None):
        cdef DPCTLSyclMemoryPoolRef pref = NULL
        cdef SyclDevice dev
        cdef SyclContext ctx
        self._pool_ref = NULL
        self._context = None
        self._device = None

        if sycl_device is _UNINIT:
            # Deferred init: caller (e.g. get_default) populates fields.
            return

        if sycl_device is None:
            dev = dpctl.SyclDevice()
        elif isinstance(sycl_device, SyclDevice):
            dev = <SyclDevice>sycl_device
        else:
            raise TypeError(
                "sycl_device must be a dpctl.SyclDevice instance or None; "
                f"got {type(sycl_device).__name__}"
            )

        ctx = _default_context_for(dev)

        pref = DPCTLMemoryPool_Create(
            ctx.get_context_ref(), dev.get_device_ref()
        )
        if pref is NULL:
            raise RuntimeError(
                "Failed to create SYCL USM-device memory pool"
            )
        self._pool_ref = pref
        self._context = ctx
        self._device = dev

    def __dealloc__(self):
        # If this pool is currently the installed device-USM pool for
        # its (context, device), clear the C-side registry entry
        # before deleting our handle. Defensive: the standard contract
        # is that the user calls reset_allocator() first.
        if self._pool_ref is not NULL \
                and self._context is not None and self._device is not None:
            if DPCTLMemoryPool_GetInstalled(
                self._context.get_context_ref(),
                self._device.get_device_ref(),
            ) == self._pool_ref:
                DPCTLMemoryPool_SetInstalled(
                    self._context.get_context_ref(),
                    self._device.get_device_ref(),
                    NULL,
                )
        if self._pool_ref is not NULL:
            DPCTLMemoryPool_Delete(self._pool_ref)
            self._pool_ref = NULL

    @staticmethod
    def get_default(*, SyclDevice sycl_device=None):
        """Return a :class:`MemoryPool` wrapper around the SYCL
        runtime's default USM-device memory pool for the given
        ``(context, device)``.

        Repeated calls with equivalent arguments return the same
        Python wrapper object (compared with ``is``).
        """
        cdef DPCTLSyclMemoryPoolRef pref = NULL
        cdef MemoryPool obj
        cdef SyclDevice dev
        cdef SyclContext ctx

        if sycl_device is None:
            dev = dpctl.SyclDevice()
        else:
            dev = sycl_device

        ctx = _default_context_for(dev)

        cache_key = (hash(ctx), hash(dev))
        # Double-checked lookup under the lock so that concurrent
        # callers strictly share one wrapper per cache key.
        with _default_pool_lock:
            existing = _default_pool_cache.get(cache_key)
            if existing is not None:
                return existing

            pref = DPCTLMemoryPool_CreateDefault(
                ctx.get_context_ref(), dev.get_device_ref()
            )
            if pref is NULL:
                raise RuntimeError(
                    "Failed to obtain SYCL default USM-device memory pool"
                )

            obj = MemoryPool(sycl_device=_UNINIT)
            obj._pool_ref = pref
            obj._context = ctx
            obj._device = dev
            _default_pool_cache[cache_key] = obj
            return obj

    @property
    def sycl_device(self):
        """The :class:`dpctl.SyclDevice` this pool serves."""
        return self._device

    @property
    def sycl_context(self):
        """The :class:`dpctl.SyclContext` this pool is bound to."""
        return self._context

    @property
    def is_default(self):
        """``True`` when this wrapper refers to the SYCL runtime's
        default pool (constructed via :meth:`get_default`)."""
        if self._pool_ref is NULL:
            return False
        return bool(DPCTLMemoryPool_IsDefault(self._pool_ref))

    def __call__(self, Py_ssize_t nbytes, SyclQueue sycl_queue not None):
        """Allocator-hook protocol: equivalent to :meth:`malloc`. Lets
        users write ``dpctl.memory.set_allocator(pool)`` instead of
        ``set_allocator(pool.malloc)``.
        """
        return self.malloc(nbytes, sycl_queue)

    def malloc(self, Py_ssize_t nbytes, SyclQueue sycl_queue not None):
        """Allocate ``nbytes`` bytes of USM-device memory from the pool
        and return a :class:`MemoryUSMDevice` instance.

        Args:
            nbytes: Number of bytes to allocate. Must be positive.
            sycl_queue: Queue used to stream-order the allocation.
                The eventual pool-return is also ordered against this
                queue, preserving lifetime safety relative to the work
                that consumed the allocation. ``sycl_queue`` must
                share the pool's SYCL context.
        """
        cdef DPCTLSyclUSMRef p = NULL
        cdef DPCTLSyclQueueRef alloc_qref
        cdef DPCTLSyclMemoryPoolRef pool_ref
        cdef _Memory base

        if nbytes <= 0:
            raise ValueError(
                "Number of bytes for pool allocation must be positive"
            )
        if sycl_queue.sycl_context != self._context:
            raise ValueError(
                "sycl_queue passed to MemoryPool.malloc must share "
                "the pool's SYCL context"
            )

        # Extract raw refs before releasing the GIL; the underlying
        # SYCL objects remain valid because ``sycl_queue`` and
        # ``self`` are held as Python locals.
        alloc_qref = sycl_queue.get_queue_ref()
        pool_ref = self._pool_ref

        with nogil:
            p = DPCTLMemoryPool_Malloc(
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
                sycl_queue,
                self,
                pool_ref,
                p,
            )
        except Exception:
            DPCTLMemoryPool_AsyncFree(pool_ref, alloc_qref, p)
            raise

        return MemoryUSMDevice(base)

    def set_release_threshold(self, size_t threshold):
        """Raise the pool's release threshold in bytes. Below the
        threshold the implementation retains cached blocks; above it,
        excess may be released back to the driver. Monotonic — can
        only be raised, not lowered. No-op when
        :func:`is_memory_pool_available` returns ``False``.
        """
        DPCTLMemoryPool_SetReleaseThreshold(self._pool_ref, threshold)

    def reset_memory(self, SyclQueue sycl_queue=None):
        """Best-effort eviction of unused cached blocks.

        Drives the pool's release threshold to 0 (asking the runtime
        to release the cache at its next opportunity). When
        ``sycl_queue`` is provided, also synchronizes that queue so
        any in-flight stream-ordered frees on it are flushed before
        returning. No-op when :func:`is_memory_pool_available`
        returns ``False``.
        """
        if not is_memory_pool_available():
            return
        DPCTLMemoryPool_ResetMemory(self._pool_ref)
        if sycl_queue is not None:
            sycl_queue.wait()

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


# ---------------------------------------------------------------------------
# C-API exports for downstream consumers (e.g. dpnp) that include
# dpctl4pybind11.hpp. These thin ``cdef api`` shims delegate to the
# underlying libDPCTLSyclInterface entry points, avoiding the need for
# the consumer to link against ``libDPCTLSyclInterface.so`` directly --
# Cython's import-time symbol-export mechanism wires them up at the
# first call.
# ---------------------------------------------------------------------------

cdef api DPCTLSyclMemoryPoolRef MemoryPool_GetInstalled(
        DPCTLSyclContextRef cref, DPCTLSyclDeviceRef dref) noexcept nogil:
    """Return the installed USM-device pool for ``(cref, dref)`` or
    NULL if no pool is installed."""
    return DPCTLMemoryPool_GetInstalled(cref, dref)


cdef api DPCTLSyclUSMRef MemoryPool_Malloc(
        DPCTLSyclMemoryPoolRef pref,
        DPCTLSyclQueueRef qref,
        size_t size) noexcept nogil:
    """Allocate ``size`` bytes from ``pref`` stream-ordered against
    ``qref``. ``qref`` must share the pool's SYCL context."""
    return DPCTLMemoryPool_Malloc(pref, qref, size)


cdef api void MemoryPool_AsyncFree(
        DPCTLSyclMemoryPoolRef pref,
        DPCTLSyclQueueRef qref,
        DPCTLSyclUSMRef mref) noexcept nogil:
    """Stream-ordered free of ``mref`` against ``qref``. ``qref``
    must share the pool's SYCL context."""
    DPCTLMemoryPool_AsyncFree(pref, qref, mref)
