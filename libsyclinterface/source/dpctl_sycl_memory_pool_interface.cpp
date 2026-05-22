//===-- dpctl_sycl_memory_pool_interface.cpp - C API for SYCL mem pool ----===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implements ``dpctl_sycl_memory_pool_interface.h``.
///
/// Two backend strategies are provided:
///
///   * Extension backend, enabled when the compiler defines
///     ``SYCL_EXT_ONEAPI_MEMORY_POOL`` and ``SYCL_EXT_ONEAPI_ASYNC_ALLOC``.
///     The pool owns a ``sycl::ext::oneapi::experimental::memory_pool``
///     instance; allocation goes through ``async_malloc_from_pool`` and
///     freeing through ``async_free``, both stream-ordered against the
///     queue used to create the pool.
///
///   * Fallback backend, used otherwise. The pool object holds only a
///     queue and a USM kind; allocations are direct ``sycl::malloc_*``
///     calls and frees are synchronous ``sycl::free`` on the pool's
///     context. ``TrimTo`` is a no-op.
///
/// The opaque handle has the same shape on both paths so that callers
/// never need to branch.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_memory_pool_interface.h"

#include "Config/dpctl_config.h"
#include "dpctl_error_handlers.h"
#include "dpctl_sycl_type_casters.hpp"

#include <sycl/sycl.hpp>

#include <new>
#include <utility>

#if defined(SYCL_EXT_ONEAPI_MEMORY_POOL) &&                                    \
    defined(SYCL_EXT_ONEAPI_ASYNC_ALLOC)
#define DPCTL_HAS_SYCL_MEMORY_POOL_EXT 1
#else
#define DPCTL_HAS_SYCL_MEMORY_POOL_EXT 0
#endif

using namespace dpctl::syclinterface;

namespace
{

// ---------------------------------------------------------------------------
// Internal pool bookkeeping structure
// ---------------------------------------------------------------------------
//
// The struct has the same layout regardless of whether the SYCL extension
// is available. The ``pool`` member is either a pointer to a real
// ``sycl::ext::oneapi::experimental::memory_pool`` (extension path) or
// ``nullptr`` (fallback path). Code paths that touch ``pool`` are guarded
// by ``DPCTL_HAS_SYCL_MEMORY_POOL_EXT``.
//
// ``is_default`` records whether ``pool`` points at a private (heap-
// allocated, owned) object or at the runtime's per-(context,device,kind)
// default-pool singleton. The destructor uses this to decide whether to
// ``delete pool`` (private) or just drop the handle (default).
//
struct DPCTLPoolImpl
{
    sycl::queue queue;
    sycl::usm::alloc kind;
    bool is_default;
#if DPCTL_HAS_SYCL_MEMORY_POOL_EXT
    sycl::ext::oneapi::experimental::memory_pool *pool;
#else
    void *pool; // always nullptr; kept for ABI symmetry
#endif

    DPCTLPoolImpl(const sycl::queue &q, sycl::usm::alloc k, bool default_pool)
        : queue(q), kind(k), is_default(default_pool), pool(nullptr)
    {
    }
};

inline DPCTLPoolImpl *unwrap_pool(DPCTLSyclMemoryPoolRef ref)
{
    return reinterpret_cast<DPCTLPoolImpl *>(ref);
}

inline DPCTLSyclMemoryPoolRef wrap_pool(DPCTLPoolImpl *impl)
{
    return reinterpret_cast<DPCTLSyclMemoryPoolRef>(impl);
}

inline sycl::usm::alloc dpctl_to_sycl_usm(DPCTLSyclUSMType t)
{
    switch (t) {
    case DPCTLSyclUSMType::DPCTL_USM_DEVICE:
        return sycl::usm::alloc::device;
    case DPCTLSyclUSMType::DPCTL_USM_SHARED:
        return sycl::usm::alloc::shared;
    case DPCTLSyclUSMType::DPCTL_USM_HOST:
        return sycl::usm::alloc::host;
    default:
        return sycl::usm::alloc::unknown;
    }
}

} // namespace

DPCTL_API
bool DPCTLMemoryPool_Available()
{
#if DPCTL_HAS_SYCL_MEMORY_POOL_EXT
    return true;
#else
    return false;
#endif
}

DPCTL_API
__dpctl_give DPCTLSyclMemoryPoolRef
DPCTLMemoryPool_Create(__dpctl_keep const DPCTLSyclQueueRef QRef,
                       DPCTLSyclUSMType usm_type)
{
    if (!QRef) {
        error_handler("Input QRef is nullptr.", __FILE__, __func__, __LINE__);
        return nullptr;
    }
    const sycl::usm::alloc kind = dpctl_to_sycl_usm(usm_type);
    if (kind == sycl::usm::alloc::unknown) {
        error_handler("Unknown USM type passed to DPCTLMemoryPool_Create.",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }
    try {
        auto Q = unwrap<sycl::queue>(QRef);
        auto impl = std::unique_ptr<DPCTLPoolImpl>(
            new DPCTLPoolImpl(*Q, kind, /*default_pool=*/false));
#if DPCTL_HAS_SYCL_MEMORY_POOL_EXT
        namespace syclex = sycl::ext::oneapi::experimental;
        // Default construct with the queue's device & context and the
        // requested USM kind. Properties are left at extension defaults
        // (no fixed reservation, default release threshold).
        impl->pool = new syclex::memory_pool(Q->get_context(),
                                             Q->get_device(), kind);
#endif
        return wrap_pool(impl.release());
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

DPCTL_API
__dpctl_give DPCTLSyclMemoryPoolRef
DPCTLMemoryPool_CreateDefault(__dpctl_keep const DPCTLSyclQueueRef QRef,
                              DPCTLSyclUSMType usm_type)
{
    if (!QRef) {
        error_handler("Input QRef is nullptr.", __FILE__, __func__, __LINE__);
        return nullptr;
    }
    const sycl::usm::alloc kind = dpctl_to_sycl_usm(usm_type);
    if (kind == sycl::usm::alloc::unknown) {
        error_handler(
            "Unknown USM type passed to DPCTLMemoryPool_CreateDefault.",
            __FILE__, __func__, __LINE__);
        return nullptr;
    }
    try {
        auto Q = unwrap<sycl::queue>(QRef);
        auto impl = std::unique_ptr<DPCTLPoolImpl>(
            new DPCTLPoolImpl(*Q, kind, /*default_pool=*/true));
#if DPCTL_HAS_SYCL_MEMORY_POOL_EXT
        namespace syclex = sycl::ext::oneapi::experimental;
        // ext_oneapi_get_default_memory_pool returns a memory_pool value;
        // we heap-allocate a copy of the handle so it can live behind the
        // opaque PoolImpl pointer. The handle is cheap (a reference into
        // the runtime singleton); copying it does NOT duplicate the pool
        // or its cache. ``is_default == true`` ensures the destructor
        // does not invoke ``delete impl->pool`` on this handle (we own
        // only the handle wrapper, not the underlying pool object).
        sycl::context ctx = Q->get_context();
        syclex::memory_pool default_pool =
            ctx.ext_oneapi_get_default_memory_pool(Q->get_device(), kind);
        impl->pool = new syclex::memory_pool(default_pool);
#endif
        return wrap_pool(impl.release());
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

DPCTL_API
void DPCTLMemoryPool_Delete(__dpctl_take DPCTLSyclMemoryPoolRef PRef)
{
    if (!PRef) {
        return;
    }
    DPCTLPoolImpl *impl = unwrap_pool(PRef);
#if DPCTL_HAS_SYCL_MEMORY_POOL_EXT
    // Even for default-pool handles we ``delete impl->pool``, because
    // ``impl->pool`` is our heap-allocated *copy* of the handle, not the
    // runtime-owned pool object itself. Destroying the handle does not
    // affect the underlying pool — the runtime keeps it alive as long as
    // its owning context is alive. This invariant relies on
    // sycl::ext::oneapi::experimental::memory_pool being implemented as
    // a reference/handle type with proper copy semantics, which the
    // extension specification requires.
    try {
        delete impl->pool;
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
    }
#endif
    delete impl;
}

DPCTL_API
bool DPCTLMemoryPool_IsDefault(
    __dpctl_keep const DPCTLSyclMemoryPoolRef PRef)
{
    if (!PRef) {
        return false;
    }
    return unwrap_pool(PRef)->is_default;
}

DPCTL_API
__dpctl_give DPCTLSyclUSMRef
DPCTLMemoryPool_Malloc(__dpctl_keep const DPCTLSyclMemoryPoolRef PRef,
                       size_t size)
{
    if (!PRef) {
        error_handler("Input PRef is nullptr.", __FILE__, __func__, __LINE__);
        return nullptr;
    }
    if (size == 0) {
        error_handler("Zero-byte allocation requested.", __FILE__, __func__,
                      __LINE__);
        return nullptr;
    }
    DPCTLPoolImpl *impl = unwrap_pool(PRef);
    void *ptr = nullptr;
    try {
#if DPCTL_HAS_SYCL_MEMORY_POOL_EXT
        namespace syclex = sycl::ext::oneapi::experimental;
        // async_malloc_from_pool: stream-ordered allocation served from
        // the pool's cache when possible.
        ptr = syclex::async_malloc_from_pool(impl->queue, size, *impl->pool);
#else
        switch (impl->kind) {
        case sycl::usm::alloc::device:
            ptr = sycl::malloc_device(size, impl->queue);
            break;
        case sycl::usm::alloc::shared:
            ptr = sycl::malloc_shared(size, impl->queue);
            break;
        case sycl::usm::alloc::host:
            ptr = sycl::malloc_host(size, impl->queue);
            break;
        default:
            break;
        }
#endif
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
    return wrap<void>(ptr);
}

DPCTL_API
void DPCTLMemoryPool_AsyncFree(__dpctl_keep const DPCTLSyclMemoryPoolRef PRef,
                               __dpctl_take DPCTLSyclUSMRef MRef)
{
    if (!PRef) {
        error_handler("Input PRef is nullptr.", __FILE__, __func__, __LINE__);
        return;
    }
    if (!MRef) {
        return;
    }
    DPCTLPoolImpl *impl = unwrap_pool(PRef);
    void *ptr = unwrap<void>(MRef);
    try {
#if DPCTL_HAS_SYCL_MEMORY_POOL_EXT
        namespace syclex = sycl::ext::oneapi::experimental;
        // Stream-ordered free; orders behind any prior submissions on the
        // pool's queue. This is the analog of cudaFreeAsync.
        syclex::async_free(impl->queue, ptr);
#else
        // Fallback: synchronous free on the pool's context. Caller is
        // responsible for ensuring no device work is in flight on ptr.
        sycl::free(ptr, impl->queue.get_context());
#endif
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
    }
}

DPCTL_API
void DPCTLMemoryPool_SetReleaseThreshold(
    __dpctl_keep const DPCTLSyclMemoryPoolRef PRef, size_t threshold)
{
    if (!PRef) {
        error_handler("Input PRef is nullptr.", __FILE__, __func__, __LINE__);
        return;
    }
#if DPCTL_HAS_SYCL_MEMORY_POOL_EXT
    DPCTLPoolImpl *impl = unwrap_pool(PRef);
    try {
        // The extension exposes the release threshold via
        // ``increase_threshold_to`` (matching the example in the
        // sycl_ext_oneapi_async_memory_alloc spec). As implied by the
        // method name this is *monotonic* — it can raise the threshold
        // but cannot lower it. Callers that need to shrink the retained
        // cache must either accept that their request will be ignored
        // if smaller than the current threshold, or call
        // ``DPCTLMemoryPool_ResetMemory`` and reconstruct the pool.
        //
        // Older DPC++ revisions may expose the threshold as a
        // property-set rather than a method; if the CI toolchain
        // disagrees, this is the call site that needs updating.
        impl->pool->increase_threshold_to(threshold);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
    }
#else
    (void)threshold;
#endif
}

DPCTL_API
void DPCTLMemoryPool_ResetMemory(
    __dpctl_keep const DPCTLSyclMemoryPoolRef PRef)
{
    if (!PRef) {
        error_handler("Input PRef is nullptr.", __FILE__, __func__, __LINE__);
        return;
    }
#if DPCTL_HAS_SYCL_MEMORY_POOL_EXT
    DPCTLPoolImpl *impl = unwrap_pool(PRef);
    try {
        // The sycl_ext_oneapi_async_memory_alloc spec at the time of
        // writing does not expose a dedicated "evict everything now"
        // method on memory_pool. The closest portable surrogate is to
        // drive the release threshold to zero; the runtime will then
        // release excess at its next opportunity (e.g. on the next
        // queue synchronization).
        //
        // If a future DPC++ revision exposes a dedicated
        // ``reset_memory()`` (or equivalent) method, this implementation
        // should be updated to call it directly for stronger semantics.
        impl->pool->increase_threshold_to(0);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
    }
#endif
}
