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

#include "syclinterface/dpctl_sycl_type_casters.hpp"
#include "syclinterface/dpctl_sycl_types.h"
#include <memory>
#include <sycl/sycl.hpp>
#include <utility>

#include <exception>
#include <iostream>

#if defined(SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS)
// eventless submit used by OpaqueSmartPtr_AsyncDelete on in-order queues
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#endif

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

// Release the USM allocation managed by ``opaque_ptr``. For an in-order queue
// the release is ordered against work already submitted to the queue: a host
// task holding a copy of the managing ``shared_ptr`` is submitted, so the
// allocation is freed only when that host task runs, i.e. after all previously
// submitted work (including kernels enqueued by external libraries that share
// the queue) has completed. This avoids releasing memory that is still in use.
//
// Deferring via a plain host task is correct ONLY for in-order queues, which
// serialize the host task after prior work. For out-of-order queues (or when
// the queue cannot be unwrapped) the host task would carry no dependency on
// that work, so the allocation is released eagerly instead -- lifetime on
// out-of-order queues is managed by the caller through explicit event
// dependencies.
//
// The original ``opaque_ptr`` is deleted before returning. Falls back to an
// eager delete if the host task cannot be submitted.
void OpaqueSmartPtr_AsyncDelete(void *opaque_ptr, DPCTLSyclQueueRef QRef)
{
    auto sptr = reinterpret_cast<std::shared_ptr<void> *>(opaque_ptr);
    sycl::queue *q_ptr = dpctl::syclinterface::unwrap<sycl::queue>(QRef);

    if (q_ptr && q_ptr->is_in_order()) {
        try {
            // copy the shared_ptr, extending the allocation's lifetime until
            // the host task below executes and the copy is destroyed
            std::shared_ptr<void> shp_copy = *sptr;
#if defined(SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS)
            // Eventless submission (sycl_ext_oneapi_enqueue_functions): on an
            // in-order queue ordering is implicit, so there is no need to
            // create a sycl::event for this fire-and-forget host task. This
            // removes the per-free event object on the common in-order path.
            namespace syclex = sycl::ext::oneapi::experimental;
            syclex::submit(*q_ptr, [&](sycl::handler &cgh) {
                cgh.host_task([shp = std::move(shp_copy)]() {
                    // no body; ``shp`` is released here, after prior work on
                    // the in-order queue has completed
                });
            });
#else
            q_ptr->submit([&](sycl::handler &cgh) {
                cgh.host_task([shp = std::move(shp_copy)]() {
                    // no body; ``shp`` is released here, after prior work on
                    // the in-order queue has completed
                });
            });
#endif
        } catch (const std::exception &e) {
            std::cout << "Deferred USM release submission caught an exception: "
                      << e.what() << std::endl;
            // fall through: the eager delete below still releases the memory
        }
    }

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
