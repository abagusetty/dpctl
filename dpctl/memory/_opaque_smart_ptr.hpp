//===--- _opaque_smart_ptr.hpp                                     --------===//
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
//===---------------------------------------------------------------------===//
///
/// \file
/// This file implements working with shared_ptr<void> with USM deleted
/// disguided as an opaque pointer.
///
//===----------------------------------------------------------------------===//

#pragma once

#ifndef __cplusplus
#error "C++ is required to compile this file"
#endif

#include "syclinterface/dpctl_sycl_memory_pool_interface.h"
#include "syclinterface/dpctl_sycl_type_casters.hpp"
#include "syclinterface/dpctl_sycl_types.h"
#include <functional>
#include <memory>
#include <sycl/sycl.hpp>
#include <utility>

#include <exception>
#include <iostream>

namespace detail
{

class USMDeleter
{
public:
    USMDeleter() = delete;
    USMDeleter(const USMDeleter &) = default;
    USMDeleter(USMDeleter &&) = default;
    USMDeleter(const ::sycl::queue &queue) : _context(queue.get_context()) {}
    USMDeleter(const ::sycl::context &context) : _context(context) {}
    template <typename T> void operator()(T *ptr) const
    {
        try {
            ::sycl::free(ptr, _context);
        } catch (const std::exception &e) {
            std::cout << "Call to sycl::free caught an exception: " << e.what()
                      << std::endl;
            // std::terminate();
        }
    }

private:
    ::sycl::context _context;
};

} // namespace detail

void *OpaqueSmartPtr_Make(void *usm_ptr, const sycl::queue &q)
{
    detail::USMDeleter _deleter(q);
    auto sptr = new std::shared_ptr<void>(usm_ptr, std::move(_deleter));

    return reinterpret_cast<void *>(sptr);
}

void *OpaqueSmartPtr_Make(void *usm_ptr, DPCTLSyclQueueRef QRef)
{
    sycl::queue *q_ptr = dpctl::syclinterface::unwrap<sycl::queue>(QRef);

    // make a copy of queue
    sycl::queue q{*q_ptr};

    void *res = OpaqueSmartPtr_Make(usm_ptr, q);

    return res;
}

void OpaqueSmartPtr_Delete(void *opaque_ptr)
{
    auto sptr = reinterpret_cast<std::shared_ptr<void> *>(opaque_ptr);

    delete sptr;
}

void *OpaqueSmartPtr_Copy(void *opaque_ptr)
{
    auto sptr = reinterpret_cast<std::shared_ptr<void> *>(opaque_ptr);
    auto copied_sptr = new std::shared_ptr<void>(*sptr);

    return reinterpret_cast<void *>(copied_sptr);
}

long OpaqueSmartPtr_UseCount(void *opaque_ptr)
{
    auto sptr = reinterpret_cast<std::shared_ptr<void> *>(opaque_ptr);
    return sptr->use_count();
}

void *OpaqueSmartPtr_Get(void *opaque_ptr)
{
    auto sptr = reinterpret_cast<std::shared_ptr<void> *>(opaque_ptr);

    return sptr->get();
}

// ---------------------------------------------------------------------------
// Pool-return callback support
// ---------------------------------------------------------------------------
//
// When a ``_Memory`` instance was produced by a user-installed allocator that
// is backed by a ``MemoryPool``, the destruction path must hand the
// allocation back to the pool rather than running the default
// shared_ptr-based ``sycl::free``. To make that polymorphic from Cython we
// store, alongside the opaque smart pointer, a separate heap-allocated
// ``std::function<void()>`` whose body knows how to enqueue the pool-return
// (typically by calling ``DPCTLMemoryPool_AsyncFree``).
//
// Both fields are independent: when no allocator hook is set, the callback
// pointer is left null and the destruction path is unchanged from earlier
// dpctl releases. The cost on the legacy code path is a single null check.
//
typedef void (*PoolReturnFn)(void *user_data);

struct PoolReturnCallback
{
    PoolReturnFn fn;
    void *user_data;
};

// Build a callback that asks ``DPCTLMemoryPool_AsyncFree(pool, usm_ptr)``.
// The caller is responsible for ensuring ``pool`` outlives the callback;
// typically the Python ``MemoryPool`` instance holding ``pool`` is kept
// alive by being captured as an attribute on the ``_Memory`` wrapper.
struct PoolFreeUserData
{
    DPCTLSyclMemoryPoolRef pool;
    DPCTLSyclUSMRef usm_ptr;
};

inline void _pool_return_invoke(void *user_data)
{
    auto *ud = reinterpret_cast<PoolFreeUserData *>(user_data);
    DPCTLMemoryPool_AsyncFree(ud->pool, ud->usm_ptr);
    delete ud;
}

void *PoolReturnCallback_Make(DPCTLSyclMemoryPoolRef pool,
                              DPCTLSyclUSMRef usm_ptr)
{
    try {
        auto *ud = new PoolFreeUserData{pool, usm_ptr};
        auto *cb = new PoolReturnCallback{&_pool_return_invoke,
                                          reinterpret_cast<void *>(ud)};
        return reinterpret_cast<void *>(cb);
    } catch (const std::exception &e) {
        std::cout << "PoolReturnCallback_Make caught an exception: " << e.what()
                  << std::endl;
        return nullptr;
    }
}

void PoolReturnCallback_Invoke(void *cb_ptr)
{
    if (!cb_ptr) {
        return;
    }
    auto *cb = reinterpret_cast<PoolReturnCallback *>(cb_ptr);
    try {
        cb->fn(cb->user_data);
    } catch (const std::exception &e) {
        std::cout << "PoolReturnCallback_Invoke caught an exception: "
                  << e.what() << std::endl;
    }
    delete cb;
}

void PoolReturnCallback_Discard(void *cb_ptr)
{
    // Used when the _Memory wrapper is being torn down without actually
    // freeing the underlying allocation (e.g. when ownership has been
    // transferred elsewhere). Releases the callback bookkeeping but does
    // NOT invoke the free.
    if (!cb_ptr) {
        return;
    }
    auto *cb = reinterpret_cast<PoolReturnCallback *>(cb_ptr);
    // The user_data was allocated as a PoolFreeUserData in
    // PoolReturnCallback_Make; release it without calling AsyncFree.
    auto *ud = reinterpret_cast<PoolFreeUserData *>(cb->user_data);
    delete ud;
    delete cb;
}
