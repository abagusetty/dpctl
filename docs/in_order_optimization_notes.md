# In-order queue optimization & SYCL-extension strengthening (design notes)

Target: Aurora (Intel Data Center GPU Max / PVC, Level Zero backend), recent
Intel DPC++. Goal: make dpctl's in-order-queue path the most optimal and stable
design for performance and stability — a CUDA-stream-class model.

These are **sketches to implement and benchmark on hardware**. They were written
without a SYCL build available, so every item is gated behind a feature-test
macro with a fallback to today's behavior. Validate each with `SYCL_EXT_*`
guards on the Aurora toolchain before relying on it.

## Implementation status

| Item | Status | Where |
| --- | --- | --- |
| A. Same-queue in-order keep-alive elision | Implemented (enabler) | `dpctl::utils::keep_args_alive_in_order` in `dpctl4pybind11.hpp`; caller opt-in |
| B1. Eventless deferred USM free | Implemented | `OpaqueSmartPtr_AsyncDelete` in `dpctl/memory/_opaque_smart_ptr.hpp` |
| B1. Eventless synchronous memcpy/memset | Resolved — not pursued | see Part E.6 (precise event wait kept; pool is the real win) |
| B2. `get_last_event` / `set_external_event` | Implemented | C-API + `SyclQueue` methods |
| B3. Native-command interop | Implemented | `dpctl::utils::enqueue_native_command` in `dpctl4pybind11.hpp` |
| E.2. In-order USM caching pool (was B4) | Designed (implementation-ready) | Part E; opt-in core allocator, needs SYCL build + tests |
| E.4. Runtime async USM pool | Future work | `sycl_ext_oneapi_async_memory_alloc` is *proposed* (not usable yet) |

All implemented items are macro-gated and fall back to current behavior; they
have **not** been compiled or run — validate on Aurora.

---

## Part A — Immediate, testable: drop per-op keep-alive for same-queue in-order USM args

### Why it is safe (and where it is NOT)

`keep_args_alive` / `_submit_keep_args_alive` submit a host task that holds a
ref to each argument until the op completes. On an **in-order** queue, a USM
allocation made **on that same queue** already has its free deferred behind a
host task ordered after all prior work (`OpaqueSmartPtr_AsyncDelete`,
`dpctl/memory/_opaque_smart_ptr.hpp`). So for such args the keep-alive host task
is redundant: whether the Python wrapper outlives the op, or is dropped right
after submit (its `__dealloc__` enqueues the deferred free, which the in-order
queue serializes *after* the kernel), the memory cannot be freed early.

This is **only** valid when both hold:
1. the queue is in-order, and
2. the USM arg was allocated on *that same queue*.

It is **unsafe** for (a) non-USM Python objects (e.g. a NumPy host buffer feeding
an async `memcpy` — host memory must persist until the read completes), and
(b) USM allocated on a *different* queue/context (its deferred free is ordered on
the other queue, not the master). dpctl cannot cheaply prove (2) inside the
pybind11 header (SYCL does not expose an event's originating queue), so the
decision belongs to the **caller** (dpnp / your app), which knows the topology.

### Caller-side pattern (dpnp / app)

```python
# q is THE in-order master queue; all USM operands were allocated on q.
def submit_op(q, kernel, usm_operands, host_operands=()):
    ev = kernel(q, ...)            # in-order: no explicit deps needed

    # USM operands: NO keep-alive. Their MemoryUSM*.__dealloc__ defers the
    # free on q, which the in-order queue serializes after `ev`.
    #
    # Only non-USM host objects still need a lifetime anchor, and even then
    # the deps list can be empty: the in-order queue orders the decref host
    # task after `ev` automatically.
    if host_operands:
        q._submit_keep_args_alive(host_operands, [])   # note: empty deps

    return ev
```

Two wins per op for the common USM-only case: one fewer host-task submission and
no `SyclEvent` dependency-array marshalling. For workloads with many small ops
this is the bulk of the per-op Python/runtime overhead.

### Optional dpctl helper to make the contract explicit

Add to `SyclQueue` so callers state the precondition once instead of hand-rolling
the split:

```cython
cpdef SyclEvent _submit_keep_nonusm_alive(self, object args):
    """Keep ONLY the non-USM-managed objects in ``args`` alive, with no
    explicit event dependencies. Correct ONLY on an in-order queue when every
    USM-managed arg was allocated on this queue (its deferred free then orders
    after prior work). USM-managed args are skipped intentionally."""
    # filter out OpaqueSmartPtr-managed USM objects (see ManagedMemory in
    # dpctl4pybind11.hpp: is_usm_managed_by_shared_ptr), then call
    # async_dec_ref(self._queue_ref, &nonusm, n, NULL, 0, &status)
    ...
```

Mirror it in the C++ header (`keep_args_alive` already classifies args via
`detail::ManagedMemory::is_usm_managed_by_shared_ptr` — line ~781): add an
`in_order`/`assume_same_queue` overload that skips the `shp_usm` host task and
emits only the `shp_arr` (Python handle) host task with no `depends_on`.

---

## Part B — Strengthen in-order with SYCL extensions

### B1. Eventless submission — `sycl_ext_oneapi_enqueue_functions` (experimental)

Macro: `SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS`. Namespace:
`sycl::ext::oneapi::experimental`. Eventless free functions:
`submit`, `single_task`, `parallel_for`, `nd_launch`, `memcpy`, `copy`,
`memset`, `fill`, `prefetch`; event-returning variant `submit_with_event`.

> NOTE: this replaces the **deprecated** `discard_events` queue property. Do not
> use `discard_events`; opt out of events per-submission with `submit`.

On an in-order queue, ordering needs no event, so every place dpctl creates a
`sycl::event` only to enforce ordering or to wait is pure overhead. Apply:

**Deferred USM free** (`_opaque_smart_ptr.hpp`, `OpaqueSmartPtr_AsyncDelete`):
```cpp
#ifdef SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS
namespace syclex = sycl::ext::oneapi::experimental;
syclex::submit(*q_ptr, [&](sycl::handler &cgh) {
    cgh.host_task([shp = std::move(shp_copy)]() {});   // no event created
});
#else
q_ptr->submit([&](sycl::handler &cgh) {                 // current path
    cgh.host_task([shp = std::move(shp_copy)]() {});
});
#endif
```

**Synchronous memory ops** (`_sycl_queue.pyx` `memcpy`/`prefetch`/`mem_advise`;
`_memory.pyx` `copy_to_host`/`copy_from_host`/`copy_from_device`/`memset`):
today they create an event then `DPCTLEvent_Wait(ERef)`. On in-order, replace
with an eventless enqueue + `DPCTLQueue_Wait`:
```cpp
// eventless: syclex::memcpy(q, dst, src, n); then q.wait();
```
Tradeoff: `queue.wait()` drains *all* pending work on the (possibly shared)
queue, not just this op. On in-order that op is last in line so the effect is the
same completion point; acceptable for a synchronous API, but document it. Needs a
new C-API entry (e.g. `DPCTLQueue_MemcpyEventless`) wrapping the eventless call.

**`memcpy_async` fast path**: when the caller passes no `dEvents` and ignores the
returned event, offer an eventless variant to avoid the `SyclEvent` allocation.

### B2. In-order event handoff — `sycl_ext_oneapi_in_order_queue_events` (experimental)

Macro: `SYCL_EXT_ONEAPI_IN_ORDER_QUEUE_EVENTS`. Two queue methods:
```cpp
std::optional<sycl::event> queue::ext_oneapi_get_last_event() const;
void queue::ext_oneapi_set_external_event(const sycl::event &externalEvent);
```

This is the clean primitive that lets dpctl stop maintaining event lists for
in-order queues entirely, and resolves the cross-queue-dependency concern from
Part A without threading `depends_on` through every call:

- **Cross-queue producer → in-order consumer**: instead of passing a deps list,
  inject the dependency once:
  ```cpp
  in_order_q.ext_oneapi_set_external_event(producer_ev);
  // the NEXT submission on in_order_q depends on producer_ev (== handler::depends_on)
  ```
- **In-order producer → other consumer**: hand off the implicit last event:
  ```cpp
  auto last = in_order_q.ext_oneapi_get_last_event();  // std::nullopt if empty
  ```

Design impact: for in-order queues, `SequentialOrderManager` already returns the
no-op manager (no bookkeeping). With B2, even the *cross-queue* cases that today
justify keeping explicit deps can be expressed via `set_external_event` /
`get_last_event`, so dpctl/dpnp can drop dependency-array marshalling across the
board for in-order queues and rely on the queue's own last-event tracking.
(Throws if the queue is not in-order — gate on `q.is_in_order`.)

### B3. Order native/external kernels into the in-order queue — `sycl_ext_codeplay_enqueue_native_command` (experimental)

Macro: `SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND`. This is the correct mechanism
for the original motivation — an external C++/Level-Zero library sharing dpctl's
in-order queue — replacing "pass the raw queue and hope ordering holds." It hooks
the native command into the SYCL dependency graph so it is ordered w.r.t. dpnp
ops on the same in-order queue. Add a helper to `dpctl4pybind11.hpp`:

```cpp
// Order a native Level-Zero (or other backend) command inside `q`.
template <typename NativeCallable>
sycl::event enqueue_native_command(
    sycl::queue &q, NativeCallable &&fn,
    const std::vector<sycl::event> &deps = {})
{
    return q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);
        cgh.ext_codeplay_enqueue_native_command(
            [=](sycl::interop_handle ih) {
                auto native_q =
                    ih.get_native_queue<sycl::backend::ext_oneapi_level_zero>();
                fn(native_q);   // library enqueues onto the native L0 queue/list
            });
    });
}
```
On an in-order queue, `deps` is usually empty (implicit ordering); the returned
event completes when the native async work finishes, so dpnp ops submitted after
it are correctly ordered. This is the stability cornerstone for shared-queue
interop.

### B4. Stream-ordered USM allocation — `sycl_ext_oneapi_async_memory_alloc` (proposed)

Macro: `SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC`. This is the long-term "most optimal
and stable" memory model — the direct analog of CUDA's `cudaMallocAsync` /
`cudaFreeAsync` with a `memory_pool`:
```cpp
namespace syclex = sycl::ext::oneapi::experimental;
syclex::memory_pool pool(q.get_context(), q.get_device(), sycl::usm::alloc::device);
void *p = syclex::async_malloc_from_pool(q, nbytes, pool);   // valid after the
                                                             // malloc command runs
// ... use p on q ...
syclex::async_free(q, p);     // freed once its in-order deps are satisfied
```
Replaces the host-task deferred-free entirely for in-order queues: the runtime
orders both allocation and free natively on the queue — no host task, no
`shared_ptr` copy round trip, and pooled reuse removes per-op driver malloc cost.

Integration sketch (`dpctl/memory/_memory.pyx` + `_opaque_smart_ptr.hpp`):
- Keep a per-(context,device) `memory_pool` (alongside the cached queue).
- `_cinit_alloc`: if the extension is present and a queue is known, allocate via
  `async_malloc_from_pool`; record that the block is pool-managed.
- `__dealloc__`: for pool-managed blocks on an in-order queue, call `async_free`
  instead of `OpaqueSmartPtr_AsyncDelete`.
- **Fallback**: when `SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC` is undefined, keep
  today's `OpaqueSmartPtr` host-task deferral. Since this extension is still
  *proposed*, verify availability on your Aurora DPC++ first.

---

## Part C — Feature detection & fallback policy

Probe each extension and degrade gracefully; never hard-require an experimental
API:
```cpp
#if defined(SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS)
  // eventless path
#else
  // current event-returning path
#endif
```
Same pattern for `SYCL_EXT_ONEAPI_IN_ORDER_QUEUE_EVENTS`,
`SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND`, `SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC`.
Expose the detected capabilities to Python (e.g. a `dpctl.sycl_extensions`
dict) so dpnp/app code can choose the fast path at runtime.

## Part D — Do NOT

- Do **not** use the `discard_events` queue property — deprecated; use eventless
  `enqueue_functions::submit` instead.
- Do **not** blindly drop `depends_on` for in-order queues — only same-queue
  ordering is implicit. Use `ext_oneapi_set_external_event` for cross-queue deps.
- Do **not** skip keep-alive for non-USM host objects, or for USM allocated on a
  different queue than the in-order master.

---

## Suggested implementation / benchmarking order

1. **Part A** (pure caller-side; no extension; biggest immediate win for dpnp
   small-op throughput). Benchmark host-side op submission rate.
2. **B1** eventless deferred-free + eventless internal sync memcpy. Benchmark
   alloc/free churn and H2D/D2H latency.
3. **B2** to retire explicit dependency marshalling for in-order (incl. the
   cross-queue cases). Benchmark dependency-heavy graphs.
4. **B3** if/when external native kernels share the queue (stability).
5. **B4** when the async-alloc extension is available on Aurora (replaces the
   host-task free path; largest allocator-side win).

---

# Part E — CuPy-informed memory & synchronization design (in-order)

The single biggest performance gap between dpctl and CuPy is **allocation**:
dpctl calls the SYCL runtime (`DPCTLmalloc_*` / `sycl::free`) on every alloc and
free, while CuPy's defining optimization is a caching, stream-ordered memory
pool. This section captures what CuPy does and a concrete, in-order-aware design
to close the gap. It is **design-only**: the allocator is memory-safety-critical
core code and must be implemented with a SYCL build and tests, not landed blind.

## E.0 What CuPy does (and why it is fast)

- **Caching memory pool as the default allocator** (`cupy.cuda.MemoryPool`,
  `SingleDeviceMemoryPool`). `malloc` rounds the size to a bin and pops a cached
  block from that bin's free-list; `free` returns the block to the pool (it does
  **not** call the driver). The block is recorded against the **stream** it was
  last used on, so reuse on that same stream is safe with no synchronization.
  `free_all_blocks()` returns cached memory to the driver under pressure.
- **Stream-ordered async pool** (`MemoryAsyncPool`) backed by
  `cudaMallocAsync` / `cudaFreeAsync` — the driver's own stream-ordered
  allocator (the CUDA analog of `sycl_ext_oneapi_async_memory_alloc`).
- **Lifetime by refcount + stream tracking**: an `ndarray` holds a
  `MemoryPointer`; the underlying block records its last-use stream. Freeing a
  block while work is still pending is safe because the block is only *reused*
  once that stream has passed the free point — no host task needed.
- **Minimal device synchronization**: everything is stream-ordered; explicit
  sync happens only at host transfer (`.get()`) or `stream.synchronize()`.
- **Retry-on-OOM**: on allocation failure, `free_all_blocks()` then retry once.

## E.1 dpctl gaps vs CuPy

- Every allocation is a driver round-trip; every free is a driver `sycl::free`
  (plus, on in-order, a host task via `OpaqueSmartPtr_AsyncDelete`). For array
  workloads with high alloc/free churn this dominates host overhead.
- No size-binned reuse, so transient temporaries repeatedly hit the runtime.

## E.2 Design: in-order USM caching pool

- One `UsmPool` per `(sycl::context, device, usm_kind)`. Free-list keyed by a
  rounded **size bin** (e.g. powers of two, or CuPy's 512 B-rounded bins).
- `alloc(nbytes, q)`: pop a cached block from the bin `>= nbytes`; else
  `DPCTLmalloc_*`. Record the block's `last_use_queue = q`.
- `free(block, q)`: push the block back onto its bin's free-list with
  `last_use_queue = q` — **no** `sycl::free`, **no** host task.
- **In-order reuse safety** (the crux):
  - *Same queue* (`new_q == block.last_use_queue`, in-order): reuse
    immediately, no sync — the in-order queue serializes the new use after the
    previous use, exactly like CuPy's same-stream reuse.
  - *Different queue / context*: inject a dependency instead of synchronizing —
    take `producer.ext_oneapi_get_last_event()` and
    `consumer.ext_oneapi_set_external_event(ev)` (this is **B2**), so the first
    use of the reused block waits on the previous user without a host-side stall.
    If the extension is unavailable, fall back to waiting on the block's last
    event before reuse.
- **Eviction**: `free_all_blocks()` calls `sycl::free` on cached blocks (memory
  pressure, device teardown). On `USMAllocationError`, flush the matching pool
  and retry the allocation once (mirrors CuPy).
- **Opt-in, safe by default**: gate behind `dpctl.memory.set_usm_allocator()` /
  an env var, default OFF. Default behavior — and stability — is unchanged until
  explicitly enabled, so the pool can be merged and matured behind a flag.

## E.3 Lifetime integration (`_memory.pyx`, `_opaque_smart_ptr.hpp`)

- `_Memory.__dealloc__`: when the block is pool-managed, return it to the pool
  (`free(block, self.queue)`) instead of `OpaqueSmartPtr_AsyncDelete`. The
  in-order reuse policy in E.2 replaces the per-free host task entirely.
- Track `last_use_queue` on the `_Memory` object. Minimal correct version: set
  it to `self.queue` (the allocation/owning queue); refine if an array is used
  on a different queue than it was allocated on (then update on use).
- Non-pooled allocations keep today's `OpaqueSmartPtr_AsyncDelete` path.

## E.4 When `sycl_ext_oneapi_async_memory_alloc` becomes usable on Aurora

Replace the software pool's `DPCTLmalloc_*`/`sycl::free` core with
`async_malloc_from_pool` / `async_free` against a `sycl::memory_pool` — the
runtime then manages binning and stream-ordering natively (the direct
`cudaMallocAsync` analog). Keep the software pool (E.2) as the portable fallback.

## E.5 Synchronization audit (current code, in-order lens)

Reviewed and found correct — no over-synchronization to remove:
- `copy_via_host` (`_memory.pyx`): cross-context; threads `E1`→`E2` dependency
  and waits on the final event. The host staging buffer outlives the copy
  because the function blocks on `E2` before returning. Correct.
- `copy_to_host` / `copy_from_host` / `copy_from_device` (same context) /
  `memset`: direct submit + wait on **that op's** event. On an in-order queue
  the op is serialized after prior work, so waiting on it also implies prior
  work has completed — precise, not a full-queue drain.
- `_NoOpOrderManager.wait()` and `SyclQueue.wait()`: full-queue drain, which is
  the intended "wait for everything" semantics.

## E.6 Resolved TODOs

- **B1 eventless synchronous memcpy/memset — not pursued.** `DPCTLEvent_Wait`
  on the op's event is already precise and backend-agnostic. Eventless +
  `queue.wait()` is only equivalent on in-order queues and would require a new
  C-API entry per op to save a single event object; on a shared queue it can
  also wait for unrelated work. The real allocation/synchronization win is the
  pool (E.2), so the sync-memcpy event path is kept as-is.
- **B4 stream-ordered async allocation — superseded by E.2/E.4.** Implement the
  software caching pool (E.2) now; switch its core to the runtime async
  allocator (E.4) once `sycl_ext_oneapi_async_memory_alloc` is available
  (currently a *proposed*, non-usable extension).
