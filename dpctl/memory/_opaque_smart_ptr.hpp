//===--- _opaque_smart_ptr.hpp                                     --------===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2024 Intel Corporation
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

#include <algorithm>
#include <exception>
#include <functional>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

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

namespace detail
{

// Registry of deferred-free host-task events, keyed by the underlying
// ``sycl::queue``. ``sycl::malloc_device`` (allocation) is a synchronous
// host call, but on an in-order queue ``sycl::free`` is deferred behind a
// host task (see ``OpaqueSmartPtr_AsyncDelete``). The Level-Zero USM
// allocator can hand back a virtual address whose pending free host task
// has not run yet, so a fresh allocation may alias memory that is about to
// be unmapped -- a use-after-free that manifests as an intermittent GPU
// page fault. To close that window, allocation drains the pending frees for
// its queue before calling ``malloc_*``.
//
// Keyed by ``sycl::queue`` value (via ``std::hash<sycl::queue>`` and
// ``operator==``) rather than the wrapper pointer, so it stays correct when
// higher layers (e.g. dpnp) reconstruct fresh ``sycl::queue`` copies around
// the same underlying backend queue.
class PendingFreeRegistry
{
public:
    static PendingFreeRegistry &instance()
    {
        static PendingFreeRegistry inst;
        return inst;
    }

    void add(const sycl::queue &q, sycl::event ev)
    {
        std::lock_guard<std::mutex> lock(_mtx);
        auto &events = _map[q];
        // Prune events that have already completed to bound growth.
        events.erase(std::remove_if(events.begin(), events.end(),
                                    [](const sycl::event &e) {
                                        return e.get_info<sycl::info::event::
                                                              command_execution_status>() ==
                                               sycl::info::event_command_status::
                                                   complete;
                                    }),
                     events.end());
        events.push_back(std::move(ev));
    }

    void drain(const sycl::queue &q)
    {
        std::vector<sycl::event> events;
        {
            std::lock_guard<std::mutex> lock(_mtx);
            auto it = _map.find(q);
            if (it == _map.end() || it->second.empty()) {
                return;
            }
            events.swap(it->second);
        }
        for (auto &ev : events) {
            try {
                ev.wait();
            } catch (const std::exception &e) {
                std::cout << "Draining pending USM free caught an exception: "
                          << e.what() << std::endl;
            }
        }
    }

private:
    std::mutex _mtx;
    std::unordered_map<sycl::queue, std::vector<sycl::event>> _map;
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

// Release the USM allocation managed by ``opaque_ptr`` in a way that is
// ordered against work already submitted to the given queue. A host task
// holding a copy of the managing ``shared_ptr`` is submitted to the queue;
// the allocation is freed only when that host task runs. On an in-order
// queue this happens after all previously submitted work (including kernels
// enqueued by external libraries that share the queue) has completed, which
// avoids releasing memory that is still in use. The original ``opaque_ptr``
// is deleted before returning. Falls back to an eager delete if the host
// task cannot be submitted.
void OpaqueSmartPtr_AsyncDelete(void *opaque_ptr, DPCTLSyclQueueRef QRef)
{
    auto sptr = reinterpret_cast<std::shared_ptr<void> *>(opaque_ptr);
    sycl::queue *q_ptr = dpctl::syclinterface::unwrap<sycl::queue>(QRef);

    if (q_ptr) {
        try {
            // copy the shared_ptr, extending the allocation's lifetime until
            // the host task below executes and the copy is destroyed
            std::shared_ptr<void> shp_copy = *sptr;
            sycl::event free_ev = q_ptr->submit([&](sycl::handler &cgh) {
                cgh.host_task([shp = std::move(shp_copy)]() {
                    // no body; ``shp`` is released here, after prior work on
                    // the (in-order) queue has completed
                });
            });
            // Record the free event so a subsequent allocation on this queue
            // can wait for it before reusing the released virtual address.
            detail::PendingFreeRegistry::instance().add(*q_ptr,
                                                        std::move(free_ev));
        } catch (const std::exception &e) {
            std::cout << "Deferred USM release submission caught an exception: "
                      << e.what() << std::endl;
            // fall through: the eager delete below still releases the memory
        }
    }

    delete sptr;
}

// Wait for any deferred USM free host tasks previously submitted on the queue
// identified by ``QRef`` to complete, then clear them from the registry. This
// must be called before allocating on an in-order queue to prevent the USM
// allocator from reusing a virtual address whose ``sycl::free`` host task is
// still pending (which would lead to a use-after-free).
void OpaqueSmartPtr_DrainPendingFrees(DPCTLSyclQueueRef QRef)
{
    sycl::queue *q_ptr = dpctl::syclinterface::unwrap<sycl::queue>(QRef);
    if (q_ptr) {
        detail::PendingFreeRegistry::instance().drain(*q_ptr);
    }
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
