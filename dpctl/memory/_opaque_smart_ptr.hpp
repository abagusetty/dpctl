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
/// shared_ptr<void> with USM deleter, plus a pool-return callback used
/// when a _Memory was produced by a user-installed pool allocator.
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
    sycl::queue q{*q_ptr};
    return OpaqueSmartPtr_Make(usm_ptr, q);
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

typedef void (*PoolReturnFn)(void *user_data);

struct PoolReturnCallback
{
    PoolReturnFn fn;
    void *user_data;
};

// Both ``pool`` and ``queue`` are non-owning raw handles. Their
// lifetimes are pinned by Python-side references held on the owning
// ``_Memory``: ``_pool_owner`` keeps the ``MemoryPool`` wrapper (and
// thus ``pool``) alive, and ``queue`` (the ``_Memory.queue`` Python
// field) keeps the ``SyclQueue`` wrapper (and thus the underlying
// ``DPCTLSyclQueueRef``) alive. Both refs are valid at the moment
// ``_Memory.__dealloc__`` runs and invokes this callback.
struct PoolFreeUserData
{
    DPCTLSyclMemoryPoolRef pool;
    DPCTLSyclQueueRef queue;
    DPCTLSyclUSMRef usm_ptr;
};

inline void _pool_return_invoke(void *user_data)
{
    auto *ud = reinterpret_cast<PoolFreeUserData *>(user_data);
    DPCTLMemoryPool_AsyncFree(ud->pool, ud->queue, ud->usm_ptr);
    delete ud;
}

void *PoolReturnCallback_Make(DPCTLSyclMemoryPoolRef pool,
                              DPCTLSyclQueueRef queue,
                              DPCTLSyclUSMRef usm_ptr)
{
    try {
        auto *ud = new PoolFreeUserData{pool, queue, usm_ptr};
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
    // Releases the callback bookkeeping WITHOUT invoking the free.
    if (!cb_ptr) {
        return;
    }
    auto *cb = reinterpret_cast<PoolReturnCallback *>(cb_ptr);
    auto *ud = reinterpret_cast<PoolFreeUserData *>(cb->user_data);
    delete ud;
    delete cb;
}
