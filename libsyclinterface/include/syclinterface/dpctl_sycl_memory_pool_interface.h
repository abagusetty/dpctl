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
/// C interface to a SYCL memory-pool object backed by the
/// ``sycl_ext_oneapi_memory_pool`` and ``sycl_ext_oneapi_async_alloc``
/// extensions, with a pass-through fallback when the extensions are
/// not available.
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
 * @ingroup MemoryPoolInterface
 */
typedef struct DPCTLOpaqueSyclMemoryPool *DPCTLSyclMemoryPoolRef;

/*!
 * @brief Query whether the SYCL memory-pool / async-alloc extension is
 * available in this build of libsyclinterface.
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
bool DPCTLMemoryPool_Available();

/*!
 * @brief Create a private memory pool bound to a SYCL queue and USM type.
 *
 * @param  QRef       SYCL queue whose device/context the pool is bound to.
 * @param  usm_type   One of ``DPCTL_USM_DEVICE``, ``DPCTL_USM_SHARED``,
 *                    or ``DPCTL_USM_HOST``.
 * @return Pool handle, freed with ``DPCTLMemoryPool_Delete``, or
 *         ``nullptr`` on failure.
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclMemoryPoolRef
DPCTLMemoryPool_Create(__dpctl_keep const DPCTLSyclQueueRef QRef,
                       DPCTLSyclUSMType usm_type);

/*!
 * @brief Obtain a handle to the SYCL runtime's default memory pool for
 * the queue's ``(context, device, usm_type)`` tuple.
 *
 * The underlying pool is a runtime-managed singleton; the returned
 * handle does not own it.
 *
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclMemoryPoolRef
DPCTLMemoryPool_CreateDefault(__dpctl_keep const DPCTLSyclQueueRef QRef,
                              DPCTLSyclUSMType usm_type);

/*!
 * @brief Destroy a memory pool handle. Outstanding allocations remain
 * valid until their owning ``MemoryUSM*`` wrappers are destroyed. When
 * the handle was produced by ``DPCTLMemoryPool_CreateDefault``, the
 * underlying runtime-owned pool object is left untouched.
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
void DPCTLMemoryPool_Delete(__dpctl_take DPCTLSyclMemoryPoolRef PRef);

/*!
 * @brief Query whether a pool handle wraps a runtime-default pool
 * (``true``) or a privately-constructed pool (``false``).
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
bool DPCTLMemoryPool_IsDefault(
    __dpctl_keep const DPCTLSyclMemoryPoolRef PRef);

/*!
 * @brief Allocate USM memory from a pool, ordered against the pool's
 * bound queue.
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclUSMRef
DPCTLMemoryPool_Malloc(__dpctl_keep const DPCTLSyclMemoryPoolRef PRef,
                       size_t size);

/*!
 * @brief Allocate USM memory from a pool, ordered against ``QRef``.
 * ``QRef`` must share the pool's SYCL context.
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclUSMRef DPCTLMemoryPool_MallocOnQueue(
    __dpctl_keep const DPCTLSyclMemoryPoolRef PRef,
    __dpctl_keep const DPCTLSyclQueueRef QRef,
    size_t size);

/*!
 * @brief Stream-ordered free of a pool-allocated pointer against the
 * pool's bound queue.
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
void DPCTLMemoryPool_AsyncFree(__dpctl_keep const DPCTLSyclMemoryPoolRef PRef,
                               __dpctl_take DPCTLSyclUSMRef MRef);

/*!
 * @brief Stream-ordered free of a pool-allocated pointer against
 * ``QRef``. ``QRef`` must share the pool's SYCL context.
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
void DPCTLMemoryPool_AsyncFreeOnQueue(
    __dpctl_keep const DPCTLSyclMemoryPoolRef PRef,
    __dpctl_keep const DPCTLSyclQueueRef QRef,
    __dpctl_take DPCTLSyclUSMRef MRef);

/*!
 * @brief Set the pool's release threshold (analog of
 * ``cudaMemPoolAttrReleaseThreshold``). Monotonic — can only be
 * raised. No-op when the SYCL extension is unavailable.
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
void DPCTLMemoryPool_SetReleaseThreshold(
    __dpctl_keep const DPCTLSyclMemoryPoolRef PRef, size_t threshold);

/*!
 * @brief Best-effort eviction: ask the runtime to release cached,
 * unused blocks. No-op when the SYCL extension is unavailable.
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
void DPCTLMemoryPool_ResetMemory(
    __dpctl_keep const DPCTLSyclMemoryPoolRef PRef);

/*!
 * @brief Return the number of bytes currently handed out by the pool
 * to live allocations. Returns 0 on the fallback path.
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
size_t DPCTLMemoryPool_GetUsedBytes(
    __dpctl_keep const DPCTLSyclMemoryPoolRef PRef);

/*!
 * @brief Return the total number of bytes the pool has reserved
 * (used + free). Returns 0 on the fallback path.
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
size_t DPCTLMemoryPool_GetReservedBytes(
    __dpctl_keep const DPCTLSyclMemoryPoolRef PRef);

DPCTL_C_EXTERN_C_END
