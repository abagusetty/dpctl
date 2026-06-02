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

// The ``sycl_ext_oneapi_async_memory_alloc`` extension headers are
// not pulled in by <sycl/sycl.hpp>; include them explicitly when the
// compiler advertises the extension via SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC.
#if defined(SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC)
#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/memory_pool.hpp>
#endif

#include <mutex>
#include <new>
#include <type_traits>
#include <unordered_map>
#include <utility>

using namespace dpctl::syclinterface;

#if defined(SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC)
namespace
{
// SFINAE detection for memory_pool member functions whose names differ
// across DPC++ revisions. When a name is missing the corresponding
// query returns 0 / no-op rather than failing the build.
template <typename, typename = void>
struct has_used_size_current : std::false_type
{};
template <typename P>
struct has_used_size_current<
    P,
    std::void_t<decltype(std::declval<P>().get_used_size_current())>>
    : std::true_type
{};

template <typename, typename = void>
struct has_reserved_size_current : std::false_type
{};
template <typename P>
struct has_reserved_size_current<
    P,
    std::void_t<decltype(std::declval<P>().get_reserved_size_current())>>
    : std::true_type
{};

template <typename, typename = void>
struct has_increase_threshold_to : std::false_type
{};
template <typename P>
struct has_increase_threshold_to<
    P,
    std::void_t<decltype(std::declval<P>().increase_threshold_to(
        std::declval<size_t>()))>> : std::true_type
{};

template <typename P>
inline size_t safe_used_size(P &p)
{
    if constexpr (has_used_size_current<P>::value) {
        return p.get_used_size_current();
    }
    else {
        (void)p;
        return 0;
    }
}

template <typename P>
inline size_t safe_reserved_size(P &p)
{
    if constexpr (has_reserved_size_current<P>::value) {
        return p.get_reserved_size_current();
    }
    else {
        (void)p;
        return 0;
    }
}

template <typename P>
inline void safe_increase_threshold(P &p, size_t threshold)
{
    if constexpr (has_increase_threshold_to<P>::value) {
        p.increase_threshold_to(threshold);
    }
    else {
        (void)p;
        (void)threshold;
    }
}
} // namespace
#endif

namespace
{

// Pool identity is the (context, device) pair. The SYCL extension
// currently only supports USM-device allocations, so ``kind`` is
// implicitly ``sycl::usm::alloc::device``. ``is_default`` records
// whether the stored handle came from CreateDefault (true) or
// Create (false); both flavors are deleted via the same
// delete-handle path.
struct DPCTLPoolImpl
{
    sycl::context context;
    sycl::device device;
    bool is_default;
#if defined(SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC)
    sycl::ext::oneapi::experimental::memory_pool *pool;
#else
    void *pool; // always nullptr; kept for ABI symmetry
#endif

    DPCTLPoolImpl(const sycl::context &c,
                  const sycl::device &d,
                  bool default_pool)
        : context(c), device(d), is_default(default_pool), pool(nullptr)
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

} // namespace

DPCTL_API
bool DPCTLMemoryPool_Available()
{
#if defined(SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC)
    return true;
#else
    return false;
#endif
}

DPCTL_API
__dpctl_give DPCTLSyclMemoryPoolRef
DPCTLMemoryPool_Create(__dpctl_keep const DPCTLSyclContextRef CRef,
                       __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    if (!CRef || !DRef) {
        error_handler("Input CRef or DRef is nullptr.", __FILE__, __func__,
                      __LINE__);
        return nullptr;
    }
    try {
        auto C = unwrap<sycl::context>(CRef);
        auto D = unwrap<sycl::device>(DRef);
        auto impl = std::unique_ptr<DPCTLPoolImpl>(
            new DPCTLPoolImpl(*C, *D, /*default_pool=*/false));
#if defined(SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC)
        namespace syclex = sycl::ext::oneapi::experimental;
        impl->pool =
            new syclex::memory_pool(*C, *D, sycl::usm::alloc::device);
#endif
        return wrap_pool(impl.release());
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

DPCTL_API
__dpctl_give DPCTLSyclMemoryPoolRef
DPCTLMemoryPool_CreateDefault(__dpctl_keep const DPCTLSyclContextRef CRef,
                              __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    if (!CRef || !DRef) {
        error_handler("Input CRef or DRef is nullptr.", __FILE__, __func__,
                      __LINE__);
        return nullptr;
    }
    try {
        auto C = unwrap<sycl::context>(CRef);
        auto D = unwrap<sycl::device>(DRef);
        auto impl = std::unique_ptr<DPCTLPoolImpl>(
            new DPCTLPoolImpl(*C, *D, /*default_pool=*/true));
#if defined(SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC)
        namespace syclex = sycl::ext::oneapi::experimental;
        // The memory_pool returned by ext_oneapi_get_default_memory_pool
        // is a handle/reference type; copying it does NOT duplicate the
        // underlying runtime-owned pool object.
        syclex::memory_pool default_pool =
            C->ext_oneapi_get_default_memory_pool(
                *D, sycl::usm::alloc::device);
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
#if defined(SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC)
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

namespace
{

void *pool_malloc_on(DPCTLPoolImpl *impl, const sycl::queue &q, size_t size)
{
#if defined(SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC)
    namespace syclex = sycl::ext::oneapi::experimental;
    return syclex::async_malloc_from_pool(q, size, *impl->pool);
#else
    (void)impl;
    return sycl::malloc_device(size, q);
#endif
}

void pool_free_on(DPCTLPoolImpl *impl, const sycl::queue &q, void *ptr)
{
#if defined(SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC)
    namespace syclex = sycl::ext::oneapi::experimental;
    (void)impl;
    syclex::async_free(q, ptr);
#else
    (void)impl;
    // Synchronous free on the context shared by ``q``; caller must
    // ensure no device work is in flight on ``ptr``.
    sycl::free(ptr, q.get_context());
#endif
}

} // namespace

DPCTL_API
__dpctl_give DPCTLSyclUSMRef
DPCTLMemoryPool_Malloc(__dpctl_keep const DPCTLSyclMemoryPoolRef PRef,
                       __dpctl_keep const DPCTLSyclQueueRef QRef,
                       size_t size)
{
    if (!PRef || !QRef) {
        error_handler("Input PRef or QRef is nullptr.", __FILE__, __func__,
                      __LINE__);
        return nullptr;
    }
    if (size == 0) {
        error_handler("Zero-byte allocation requested.", __FILE__, __func__,
                      __LINE__);
        return nullptr;
    }
    DPCTLPoolImpl *impl = unwrap_pool(PRef);
    try {
        sycl::queue *q = unwrap<sycl::queue>(QRef);
        return wrap<void>(pool_malloc_on(impl, *q, size));
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

DPCTL_API
void DPCTLMemoryPool_AsyncFree(__dpctl_keep const DPCTLSyclMemoryPoolRef PRef,
                               __dpctl_keep const DPCTLSyclQueueRef QRef,
                               __dpctl_take DPCTLSyclUSMRef MRef)
{
    if (!PRef || !QRef) {
        error_handler("Input PRef or QRef is nullptr.", __FILE__, __func__,
                      __LINE__);
        return;
    }
    if (!MRef) {
        return;
    }
    DPCTLPoolImpl *impl = unwrap_pool(PRef);
    try {
        sycl::queue *q = unwrap<sycl::queue>(QRef);
        pool_free_on(impl, *q, unwrap<void>(MRef));
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
#if defined(SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC)
    DPCTLPoolImpl *impl = unwrap_pool(PRef);
    try {
        safe_increase_threshold(*impl->pool, threshold);
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
#if defined(SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC)
    DPCTLPoolImpl *impl = unwrap_pool(PRef);
    try {
        // No dedicated "evict now" API in the spec; setting the
        // release threshold to 0 asks the runtime to release excess
        // cached blocks at its next opportunity.
        safe_increase_threshold(*impl->pool, 0);
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
#if defined(SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC)
    DPCTLPoolImpl *impl = unwrap_pool(PRef);
    try {
        return safe_used_size(*impl->pool);
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
#if defined(SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC)
    DPCTLPoolImpl *impl = unwrap_pool(PRef);
    try {
        return safe_reserved_size(*impl->pool);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return 0;
    }
#else
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// Installed-pool registry (process-wide, non-owning).
//
// Mirrors the Python-side ``dpctl.memory._allocator._registry`` for
// device-USM only. Lets C++ consumers (e.g. dpnp's smart_malloc_*)
// look up the pool installed via ``dpctl.memory.set_allocator``
// without round-tripping through Python.
//
// Entries are non-owning: the caller (Python's set_allocator) is
// responsible for clearing the entry before letting the underlying
// MemoryPool Python wrapper die. MemoryPool.__dealloc__ does this
// automatically as a safety net.
// ---------------------------------------------------------------------------
namespace
{
using InstalledKey = std::pair<std::size_t, std::size_t>;

struct InstalledKeyHash
{
    std::size_t operator()(const InstalledKey &k) const noexcept
    {
        return k.first ^ (k.second + 0x9e3779b97f4a7c15ULL + (k.first << 6) +
                          (k.first >> 2));
    }
};

std::unordered_map<InstalledKey, DPCTLSyclMemoryPoolRef, InstalledKeyHash> &
installed_registry()
{
    static std::unordered_map<InstalledKey, DPCTLSyclMemoryPoolRef,
                              InstalledKeyHash>
        reg;
    return reg;
}

std::mutex &installed_registry_mutex()
{
    static std::mutex m;
    return m;
}

bool make_installed_key(DPCTLSyclContextRef CRef,
                        DPCTLSyclDeviceRef DRef,
                        InstalledKey &out) noexcept
{
    if (!CRef || !DRef) {
        return false;
    }
    try {
        auto *C = unwrap<sycl::context>(CRef);
        auto *D = unwrap<sycl::device>(DRef);
        out = {std::hash<sycl::context>{}(*C), std::hash<sycl::device>{}(*D)};
        return true;
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return false;
    }
}
} // namespace

DPCTL_API
void DPCTLMemoryPool_SetInstalled(
    __dpctl_keep const DPCTLSyclContextRef CRef,
    __dpctl_keep const DPCTLSyclDeviceRef DRef,
    __dpctl_keep const DPCTLSyclMemoryPoolRef PRef)
{
    InstalledKey key;
    if (!make_installed_key(CRef, DRef, key)) {
        return;
    }
    std::lock_guard<std::mutex> lock(installed_registry_mutex());
    auto &reg = installed_registry();
    if (PRef == nullptr) {
        reg.erase(key);
    }
    else {
        reg[key] = PRef;
    }
}

DPCTL_API
__dpctl_keep DPCTLSyclMemoryPoolRef DPCTLMemoryPool_GetInstalled(
    __dpctl_keep const DPCTLSyclContextRef CRef,
    __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    InstalledKey key;
    if (!make_installed_key(CRef, DRef, key)) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(installed_registry_mutex());
    auto &reg = installed_registry();
    auto it = reg.find(key);
    return (it == reg.end()) ? nullptr : it->second;
}
