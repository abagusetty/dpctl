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
 * @brief Create a *private* memory pool bound to a given SYCL queue and
 * USM type.
 *
 * A private pool has an isolated cache; allocations served by it do not
 * benefit from (and do not contribute to) the cache state of any other
 * pool. Use this when you need pool-level isolation (memory budget
 * enforcement, debugging fragmentation, multi-tenancy in one process).
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
 * @brief Obtain a handle to the SYCL runtime's default memory pool for
 * the queue's ``(context, device, usm_type)`` tuple.
 *
 * The underlying pool is a runtime-managed singleton: every call with
 * the same ``(context, device, usm_type)`` returns a handle that refers
 * to the same underlying ``sycl::ext::oneapi::experimental::memory_pool``
 * object. Cache state (free blocks, reservation) is therefore shared
 * across all callers within a single process, which is the recommended
 * mode of use for libraries that want to amortize allocation cost
 * across the entire application (dpnp, gpu4pyscf, third-party SYCL
 * code, etc.).
 *
 * The returned handle does *not* own the pool; ``DPCTLMemoryPool_Delete``
 * on a default-pool handle releases only the wrapper, never the
 * underlying runtime-owned pool object.
 *
 * When the SYCL extension is unavailable at build time this falls back
 * to the same shape as ``DPCTLMemoryPool_Create``, behaving as a
 * private pool (because there is no runtime singleton to alias).
 *
 * @param  QRef       SYCL queue whose ``(context, device)`` selects the
 *                    pool. Must not be null.
 * @param  usm_type   One of ``DPCTL_USM_DEVICE``, ``DPCTL_USM_SHARED``,
 *                    ``DPCTL_USM_HOST``.
 * @return Pool handle on success, or ``nullptr`` on failure.
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclMemoryPoolRef
DPCTLMemoryPool_CreateDefault(__dpctl_keep const DPCTLSyclQueueRef QRef,
                              DPCTLSyclUSMType usm_type);

/*!
 * @brief Destroy a memory pool handle.
 *
 * Outstanding allocations served by this pool remain valid; their actual
 * release happens when their owning ``MemoryUSM*`` objects are destroyed.
 * This call only releases the wrapper bookkeeping. When the handle was
 * produced by ``DPCTLMemoryPool_CreateDefault``, the underlying
 * runtime-owned pool object is left untouched.
 *
 * @param  PRef   Pool handle previously returned by
 *                ``DPCTLMemoryPool_Create`` or
 *                ``DPCTLMemoryPool_CreateDefault``.
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
void DPCTLMemoryPool_Delete(__dpctl_take DPCTLSyclMemoryPoolRef PRef);

/*!
 * @brief Query whether a pool handle wraps a runtime-default pool
 * (``true``) or a privately-constructed pool (``false``).
 *
 * Useful for diagnostics and for sanity-checks in higher-level
 * bindings that need to enforce "default-only" or "private-only"
 * invariants.
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
bool DPCTLMemoryPool_IsDefault(
    __dpctl_keep const DPCTLSyclMemoryPoolRef PRef);

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
 * @brief Set the pool's *release threshold* — the lower bound (in
 * bytes) on the size of the pool's internal cache below which the
 * implementation should not release blocks back to the underlying
 * memory provider.
 *
 * This is **not** a "trim now" operation. It controls the pool's policy
 * for future release decisions:
 *
 *   * ``cached_bytes <= threshold``: implementation should retain the
 *     cache (cheap reuse for subsequent allocations).
 *   * ``cached_bytes >  threshold``: implementation may release the
 *     excess back to the driver.
 *
 * The analog in CUDA is ``cudaMemPoolAttrReleaseThreshold``.
 *
 * No-op when the SYCL extension is unavailable.
 *
 * @param  PRef        Pool handle.
 * @param  threshold   New release-threshold value in bytes.
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
void DPCTLMemoryPool_SetReleaseThreshold(
    __dpctl_keep const DPCTLSyclMemoryPoolRef PRef, size_t threshold);

/*!
 * @brief Attempt to immediately release all currently-cached, unused
 * blocks back to the underlying memory provider.
 *
 * Blocks that are still in use (handed out to live allocations) are
 * untouched. The amount actually released is implementation-defined
 * and may be bounded by the pool's release threshold.
 *
 * No-op when the SYCL extension is unavailable.
 *
 * @param  PRef   Pool handle.
 * @ingroup MemoryPoolInterface
 */
DPCTL_API
void DPCTLMemoryPool_ResetMemory(
    __dpctl_keep const DPCTLSyclMemoryPoolRef PRef);

DPCTL_C_EXTERN_C_END
