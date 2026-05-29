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
/// Implements ``dpctl_sycl_memory_pool_interface.h``. Uses the SYCL
/// memory-pool / async-alloc extensions when available; otherwise
/// falls back to plain ``sycl::malloc_*`` / ``sycl::free``.
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

// ``is_default`` records whether ``pool`` points at a private
// (heap-allocated, owned) object or at a copy of the runtime's
// default-pool handle. Only used for diagnostics; both flavors of
// handle are deleted the same way (see DPCTLMemoryPool_Delete).
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
        // The memory_pool returned by ext_oneapi_get_default_memory_pool
        // is a handle/reference type; copying it does NOT duplicate the
        // underlying runtime-owned pool object.
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
    // ``impl->pool`` is always our heap-allocated handle, not the
    // runtime-owned pool itself; deletion of the handle does not affect
    // the underlying pool.
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
        syclex::async_free(impl->queue, ptr);
#else
        // Caller is responsible for ensuring no device work is in
        // flight on ``ptr`` when the extension is unavailable.
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
        // ``increase_threshold_to`` is monotonic per the
        // sycl_ext_oneapi_async_memory_alloc spec.
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
        // No dedicated "evict now" API in the current spec; driving
        // the threshold to 0 lets the runtime release cached blocks
        // at its next opportunity.
        impl->pool->increase_threshold_to(0);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
    }
#endif
}

DPCTL_API
size_t DPCTLMemoryPool_GetUsedBytes(
    __dpctl_keep const DPCTLSyclMemoryPoolRef PRef)
{
    if (!PRef) {
        error_handler("Input PRef is nullptr.", __FILE__, __func__, __LINE__);
        return 0;
    }
#if DPCTL_HAS_SYCL_MEMORY_POOL_EXT
    DPCTLPoolImpl *impl = unwrap_pool(PRef);
    try {
        return impl->pool->get_used_size_current();
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return 0;
    }
#else
    return 0;
#endif
}

DPCTL_API
size_t DPCTLMemoryPool_GetReservedBytes(
    __dpctl_keep const DPCTLSyclMemoryPoolRef PRef)
{
    if (!PRef) {
        error_handler("Input PRef is nullptr.", __FILE__, __func__, __LINE__);
        return 0;
    }
#if DPCTL_HAS_SYCL_MEMORY_POOL_EXT
    DPCTLPoolImpl *impl = unwrap_pool(PRef);
    try {
        return impl->pool->get_reserved_size_current();
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return 0;
    }
#else
    return 0;
#endif
}
