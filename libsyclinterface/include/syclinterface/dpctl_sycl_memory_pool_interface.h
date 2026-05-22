//===-- dpctl_sycl_memory_pool_interface.h - C API for SYCL mem pool *-C++-*-=//
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
/// This header declares a C interface to a SYCL memory-pool object backed by
/// the ``sycl_ext_oneapi_memory_pool`` and ``sycl_ext_oneapi_async_alloc``
/// extensions. When neither extension is detected at compile-time, the
/// implementation degrades gracefully to a pass-through that uses the plain
/// ``sycl::malloc_*`` / ``sycl::free`` primitives.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_data_types.h"
#include "dpctl_sycl_enum_types.h"
#include "dpctl_sycl_types.h"

DPCTL_C_EXTERN_C_BEGIN

/*!
 * @brief Opaque handle to a SYCL memory pool object.
 *
 * The handle either wraps a
 * ``sycl::ext::oneapi::experimental::memory_pool`` (when the implementation
 * is built against a DPC++ that defines ``SYCL_EXT_ONEAPI_MEMORY_POOL``)
 * or a lightweight fallback bookkeeping struct used to provide the same
 * behavioral contract on older toolchains.
 *
 * @ingroup MemoryPoolInterface
 */
typedef struct DPCTLOpaqueSyclMemoryPool *DPCTLSyclMemoryPoolRef;

/*!
 * @brief Query whether the SYCL memory-pool / async-alloc extension is
 * available in this build of libsyclinterface.
 *
 * When this returns ``false`` the pool object still functions, but its
 * allocations go through ``sycl::malloc_*`` and its frees through
 * ``sycl::free``; ``DPCTLMemoryPool_TrimTo`` is a no-op. Callers can use
 * this query to decide whether to construct a pool at all.
 *
 * @return ``true`` if backed by the experimental SYCL extension, ``false``
 * otherwise.
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
bool DPCTLMemoryPool_Available();

/*!
 * @brief Create a memory pool bound to a given SYCL queue and USM type.
 *
 * @param  QRef       The SYCL queue whose device/context the pool is bound
 *                    to. Allocations from this pool are valid in the
 *                    queue's context and may be touched by the queue's
 *                    device.
 * @param  usm_type   One of ``DPCTL_USM_DEVICE``, ``DPCTL_USM_SHARED``, or
 *                    ``DPCTL_USM_HOST``. Selects the USM allocation kind
 *                    that this pool will serve.
 * @return An opaque pool handle that must be freed with
 *         ``DPCTLMemoryPool_Delete``, or ``nullptr`` on failure.
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclMemoryPoolRef
DPCTLMemoryPool_Create(__dpctl_keep const DPCTLSyclQueueRef QRef,
                       DPCTLSyclUSMType usm_type);

/*!
 * @brief Destroy a memory pool handle.
 *
 * Outstanding allocations served by this pool remain valid; their actual
 * release happens when their owning ``MemoryUSM*`` objects are destroyed.
 * This call only releases the pool bookkeeping object itself.
 *
 * @param  PRef   Pool handle previously returned by
 *                ``DPCTLMemoryPool_Create``.
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
void DPCTLMemoryPool_Delete(__dpctl_take DPCTLSyclMemoryPoolRef PRef);

/*!
 * @brief Allocate USM memory from a pool.
 *
 * The returned pointer is bound to the same SYCL context as the queue used
 * to construct the pool. The allocation is stream-ordered against work
 * previously submitted to that queue when the underlying extension is
 * available; otherwise it is a regular ``sycl::malloc_*`` call.
 *
 * @param  PRef       Pool handle.
 * @param  size       Number of bytes to allocate. Must be positive.
 * @return USM pointer, or ``nullptr`` on failure.
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclUSMRef
DPCTLMemoryPool_Malloc(__dpctl_keep const DPCTLSyclMemoryPoolRef PRef,
                       size_t size);

/*!
 * @brief Stream-ordered free of a pool-allocated pointer.
 *
 * When the SYCL extension is available, this issues an ``async_free`` on
 * the queue used to construct the pool, which orders the deallocation
 * behind any work previously submitted to that queue. When the extension
 * is unavailable, this falls back to a synchronous ``sycl::free`` on the
 * pool's context; in that case the caller is responsible for ensuring no
 * device work is still in flight on the pointer.
 *
 * @param  PRef       Pool handle.
 * @param  MRef       USM pointer previously returned by
 *                    ``DPCTLMemoryPool_Malloc``.
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
void DPCTLMemoryPool_AsyncFree(__dpctl_keep const DPCTLSyclMemoryPoolRef PRef,
                               __dpctl_take DPCTLSyclUSMRef MRef);

/*!
 * @brief Hint the pool to release cached blocks back to the driver, leaving
 * at most ``min_bytes_to_keep`` bytes in its internal cache.
 *
 * No-op when the SYCL extension is unavailable.
 *
 * @param  PRef                Pool handle.
 * @param  min_bytes_to_keep   Lower bound on the cached size after trimming.
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
void DPCTLMemoryPool_TrimTo(__dpctl_keep const DPCTLSyclMemoryPoolRef PRef,
                            size_t min_bytes_to_keep);

DPCTL_C_EXTERN_C_END
