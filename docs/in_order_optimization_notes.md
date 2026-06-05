# In-order queue optimization & SYCL-extension strengthening (design notes)

Target: Aurora (Intel Data Center GPU Max / PVC, Level Zero backend), recent
Intel DPC++. Goal: make dpctl's in-order-queue path the most optimal and stable
design for performance and stability — a CUDA-stream-class model.

These are **sketches to implement and benchmark on hardware**. They were written
without a SYCL build available, so every item is gated behind a feature-test
macro with a fallback to today's behavior. Validate each with `SYCL_EXT_*`
guards on the Aurora toolchain before relying on it.

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
