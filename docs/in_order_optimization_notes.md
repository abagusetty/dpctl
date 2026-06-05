# In-order queue optimization & SYCL-extension strengthening (design notes)

Targets: the Level Zero, CUDA and HIP backends (in that order of preference;
OpenCL is not a target), recent Intel/oneAPI DPC++. Goal: make dpctl's
in-order-queue path the most optimal and stable design for performance and
stability — a CUDA-stream-class model.

These are **sketches to implement and benchmark on hardware**. They were written
without a SYCL build available, so every item is gated behind a feature-test
macro with a fallback to today's behavior. Validate each with `SYCL_EXT_*`
guards on the Aurora toolchain before relying on it.

## Implementation status

| Item | Status | Where |
| --- | --- | --- |
| A. Same-queue in-order keep-alive elision | Implemented (enabler) | `dpctl::utils::keep_args_alive_in_order` in `dpctl4pybind11.hpp`; caller opt-in |
| B1. Eventless deferred USM free | Implemented | `OpaqueSmartPtr_AsyncDelete` in `dpctl/memory/_opaque_smart_ptr.hpp` |
| B1. Eventless synchronous memcpy/memset | Implemented | `DPCTLQueue_{Memcpy,Memset}Eventless` C-API; used by `SyclQueue.memcpy` and `_Memory.copy_to_host`/`copy_from_host`/`copy_from_device`/`memset` on in-order queues |
| B2. `get_last_event` / `set_external_event` | Implemented | C-API + `SyclQueue` methods (caller must serialize on shared queues — Part F) |
| B3. Native-command interop | Removed (unused) | targets L0/CUDA/HIP via ordinary SYCL submission |
| Memory pooling (CuPy-style) | Not pursued — known limitation | Part E |
| Thread-safety review | Done | Part F |

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

### B3. Order native/external kernels into the in-order queue — removed

A helper around `sycl_ext_codeplay_enqueue_native_command` was added and then
**removed as unused**: dpctl had no internal caller, and the project targets
Level Zero / CUDA / HIP through ordinary SYCL submission. If native-library
interop on a shared in-order queue is needed later, the consumer can wrap
`handler::ext_codeplay_enqueue_native_command` directly.

### B4. Stream-ordered USM allocation — NOT pursued

A pooled / stream-ordered allocator (`sycl_ext_oneapi_async_memory_alloc`, the
`cudaMallocAsync` analog) was considered and **deliberately not pursued**. See
Part E: dpctl's lack of a feature-complete memory pool relative to CuPy is a
known, accepted limitation, and the extension is *proposed* (not usable today).
dpctl keeps allocating/freeing through the SYCL runtime with the in-order
deferred free for safety.

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
Same pattern for `SYCL_EXT_ONEAPI_IN_ORDER_QUEUE_EVENTS` and
`SYCL_EXT_ONEAPI_QUEUE_EMPTY`.
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
2. **B1** eventless deferred-free. Benchmark alloc/free churn.
3. **B2** to retire explicit dependency marshalling for in-order (incl. the
   cross-queue cases). Benchmark dependency-heavy graphs. Honor the shared-queue
   thread-safety contract in Part F.
4. (B3 native-command interop removed — wrap
   `ext_codeplay_enqueue_native_command` in the consumer if ever needed.)

(Memory pooling / B4 is not pursued — see Part E.)

---

# Part E — Memory pooling: known limitation (not pursued)

A CuPy-style caching memory pool is the usual next allocation optimization, but
it is **deliberately not pursued** in dpctl. dpctl's lack of a feature-complete
pool relative to CuPy's (`cupy.cuda.MemoryPool` / `MemoryAsyncPool`) is a known,
accepted limitation: the implementation is not feature-complete, and the benefit
for our workloads is limited and does not justify the added correctness and
multi-threaded-safety surface in core allocation. dpctl continues to
allocate/free USM through the SYCL runtime (`DPCTLmalloc_*` / `sycl::free`), with
the in-order deferred free (`OpaqueSmartPtr_AsyncDelete`) for safety.

If pooling is ever revisited, the preferred route is the runtime stream-ordered
allocator (`sycl_ext_oneapi_async_memory_alloc`, the `cudaMallocAsync` analog),
so the SYCL runtime — not dpctl — owns pool correctness and thread-safety. That
extension is currently *proposed* and not usable today, so this remains future
work and no software pool is implemented in the meantime.

## E.1 Synchronization audit (current code, in-order lens)

Reviewed and found correct:
- `copy_to_host` / `copy_from_host` / `copy_from_device` (same context) /
  `memset` / `SyclQueue.memcpy`: eventless submit + `queue.wait()`. On an
  in-order queue the op is serialized after prior work, so the queue wait
  completes exactly this (last) submission — no per-op event, no extra waiting.
- `copy_via_host` (`_memory.pyx`): cross-context; threads `E1`→`E2` dependency
  and waits on the final event. The host staging buffer outlives the copy
  because the function blocks on `E2` before returning. Correct (cross-context,
  not queue-order, so it legitimately keeps events).
- `_NoOpOrderManager.wait()` and `SyclQueue.wait()`: full-queue drain, which is
  the intended "wait for everything" semantics.

### host_task / depends_on audit (in-order)

`host_task` usages — all appropriate (each is for lifetime management; the
host task is the correct mechanism and none are removable):
- `OpaqueSmartPtr_AsyncDelete` deferred USM free: host task with **no**
  `depends_on`; relies purely on in-order serialization. Minimal and correct.
- `async_dec_ref` (`_host_task_util.hpp`): single host task that `Py_DECREF`s
  the args; applies `depends_on` only when the caller passes events (empty
  under the no-op order manager), so no redundant edge. The host task reacquires
  the GIL; the deferred-free host task does not (it touches no Python). Correct.
- `keep_args_alive` / `keep_args_alive_in_order` (`dpctl4pybind11.hpp`):
  host tasks holding USM `shared_ptr`s / Python handles.

`depends_on` usages — caller-provided dependencies that may be cross-queue are
retained (correct): `memcpy_async` / `DPCTLQueue_MemcpyWithEvents`,
`submit_barrier`, kernel `submit`, `async_dec_ref`, `_submit_empty_task`. The one **redundant** edge — `keep_args_alive` chaining
the second host task onto the first via `depends_on(host_task_ev)` — was removed:
on an in-order queue the second host task is already serialized after the first
(which carries `depends`), so only the first submission needs the explicit
dependency.

## E.2 Resolved TODOs

- **B1 eventless synchronous memcpy/memset — implemented.** Added
  `DPCTLQueue_MemcpyEventless` / `DPCTLQueue_MemsetEventless` (eventless
  `enqueue_functions` submit, fallback discards the event). The synchronous
  paths (`SyclQueue.memcpy`, `_Memory.copy_to_host` / `copy_from_host` /
  `copy_from_device` (same context) / `memset`) now submit without creating a
  per-op `sycl::event` and synchronize with a single `queue.wait()` — the SYCL
  analog of CuPy's `cudaMemcpyAsync` + `stream.synchronize()`. Since dpctl
  queues are always in-order, there is no out-of-order fallback. `prefetch` is
  likewise eventless (`DPCTLQueue_PrefetchEventless`). `mem_advise` keeps an
  event (no `enqueue_functions` equivalent), and the cross-context
  `copy_via_host` keeps events (it needs the cross-queue dependency edge).
- **Out-of-order queue support — removed.** dpctl queues are always in-order;
  the raw-int escape hatch, the `_SequentialOrderManager` and the C++
  sequential order keeper are removed, and all out-of-order branches are gone.
- **Eventless kernel submission — implemented (opt-in).**
  `SyclQueue.submit_async(..., eventless=True)` submits a kernel without
  creating a `sycl::event` (returns `None`), via
  `DPCTLQueue_SubmitRangeEventless` / `DPCTLQueue_SubmitNDRangeEventless`.
  `dEvents` (possibly cross-queue) are still honored. The default
  (`eventless=False`) and the synchronous `submit` keep returning a usable
  event, since callers wait on / chain those returns.
- **Memory pool (was B4) — not pursued.** See Part E above (known limitation).

---

# Part F — Thread-safety (multi-threaded use)

dpctl relies on the Python GIL plus SYCL's own thread-safety guarantees. The
in-order optimizations added here were reviewed to hold when several Python
threads share queues and memory:

- **Cached immutable queue properties** (`_cached_is_in_order`,
  `_cached_has_enable_profiling`): lazily computed under the GIL; if two threads
  race they compute the *same* immutable value and write a plain `int` (no torn
  read), so the race is benign. Safe.
- **No-op order manager cached on the queue** (`q._no_op_order_manager`): the
  read/create/store runs under the GIL; a race can at worst construct two
  equivalent *stateless* `_NoOpOrderManager` objects, one of which wins. The
  order-manager `_map` is a `ContextVar` (per thread/task), so each thread has
  its own map; the shared `_in_order_managers` is a `WeakSet` mutated under the
  GIL. Safe.
- **`get_device_cached_queue` / `_global_device_queue_cache`**: backed by a
  `ContextVar` (per thread/task) — no cross-thread mutation. Safe.
- **Eventless deferred USM free** (`OpaqueSmartPtr_AsyncDelete`): invoked with
  the GIL released. It performs only atomic `shared_ptr` refcount operations and
  a thread-safe `sycl::queue::submit`; the host-task body is empty (acquires no
  GIL). Concurrent frees on the same in-order queue from different threads are
  each atomic. Safe.
- **`keep_args_alive` / `async_dec_ref`**: the host task reacquires the GIL and
  checks interpreter finalization before `Py_DECREF`. Safe.

## F.1 The contract callers MUST honor: shared in-order queues

Treat an in-order queue like a CUDA stream: **one logical owner / submitter at a
time**. Two facts make a shared in-order queue order-sensitive across threads:

1. The order of submissions to one in-order queue from multiple threads is
   non-deterministic, so "ordered after prior work" is only well-defined per
   logical stream of submission.
2. `SyclQueue.set_external_event(ev)` records a dependency for the **next**
   submission. The `set_external_event` → submit pair is **not atomic**: if two
   threads interleave it on the same queue, the external event can attach to the
   wrong submission. `get_last_event()` is likewise a snapshot of mutable queue
   state.

When sharing an in-order queue across threads, the caller must serialize the
`set_external_event`→submit sequence (and submissions in general) under its own
lock. dpctl intentionally does **not** add an internal lock around these: it
could not make cross-thread submission *ordering* meaningful, and it would
penalize the common single-owner (one queue per worker/stream) pattern. This
contract is documented on `set_external_event` / `get_last_event`.

---

# Part G — The same-queue assumption (cross-queue dependencies)

**Question:** is it safe to assume dpnp/dpctl never produce cross-queue
dependencies, and optimize for that?

**Answer:** split it by operation kind — the assumption holds for *compute*,
not for explicit *data movement*.

## G.1 Where the assumption holds (compute path)

dpnp / dpctl.tensor resolve a single **execution queue** for every operation
(`dpctl.utils.get_execution_queue`, now in dpnp) and raise
`ExecutionPlacementError` if operands disagree; temporaries are allocated on
that same queue. CuPy is analogous (work runs on the current stream). So a chain
of elementwise / reduction / linalg ops on one queue has **no cross-queue
dependencies** — submission order on the in-order queue is the only ordering
needed. This is the hot path, and it is already fully optimized:

- the no-op order manager makes dpnp pass **empty** dependency lists, so dpctl's
  dependency-handling code is a no-op on this path;
- `memcpy`/`memset`/`prefetch` and the deferred USM free are eventless;
- kernels can be launched eventlessly (`submit_async(eventless=True)`),
  USM lifetime via `keep_args_alive_in_order` (eventless, USM args skipped).

The win is obtained **without** assuming "no cross-queue ever": dpctl honors
dependencies only when explicitly given, and on this path nothing gives any.

## G.2 Where it does NOT hold (data-movement / boundary path)

These dpnp/dpctl operations genuinely cross queues or contexts and **must** keep
their event dependencies — they are isolated, synchronizing, and never on the
eventless hot path:

- **cross-context / cross-device copies** — `copy_via_host` (`E1`→`E2` event),
  peer-to-peer, multi-GPU;
- **interchange** — DLPack import, `__sycl_usm_array_interface__`,
  `dpnp.asarray(x, sycl_queue=other)` migration;
- **native / library interop** — a native (L0/CUDA/HIP) library or oneMKL
  enqueueing on another queue;
- **user multi-stream programs** — explicit overlap / pipelining across several
  in-order queues, coordinated with `dEvents` / `set_external_event` /
  `get_last_event`.

## G.3 Practical rule

Do **not** bake "no cross-queue" into dpctl globally — dpctl is the layer that
*implements* the boundary crossings above, and its public APIs
(`memcpy_async`/`submit_async` `dEvents`, `keep_args_alive`,
`set_external_event`) are the substrate user multi-stream code relies on.

Instead, the assumption lives at the **caller (dpnp/app) layer**, expressed by
*opting into* the eventless, same-queue-assuming entry points:
`submit_async(eventless=True)`, the eventless `memcpy`/`memset`/`prefetch`,
`keep_args_alive_in_order`, and relying on the no-op order manager (empty deps).
dpctl stays correct for the boundary cases; dpnp gets zero per-op event overhead
on the compute path. The two coexist with no unsafe global assumption.

---

# Part H — Completion polling

## H.1 `SyclQueue.empty()` — non-blocking completion (`sycl_ext_oneapi_queue_empty`)

The eventless style removes per-operation events, so an event's status can no
longer be polled. `SyclQueue.empty()` (`queue::ext_oneapi_empty`) fills that gap:
a **non-blocking** check of whether all submitted work has drained — the SYCL
analog of `cudaStreamQuery()` / CuPy's `Stream.done`. Use it for lazy
synchronization (only `wait()` if not empty), host-side progress/overlap, and
checking that deferred-free host tasks have run before teardown. It is reliable
on the Level Zero / CUDA / HIP backends (the project's targets) regardless of
submission style. `DPCTLQueue_Empty` catches any backend exception and returns
`false`; for a blocking guarantee always use `wait()`.
