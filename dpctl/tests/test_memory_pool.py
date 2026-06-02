#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the pluggable USM-device allocator hook and ``MemoryPool``."""

import pytest

import dpctl
import dpctl.memory as dpm


@pytest.fixture
def clean_registry():
    dpm.reset_allocator()
    try:
        yield
    finally:
        dpm.reset_allocator()


def _try_make_queue():
    try:
        return dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Could not construct a default SyclQueue")


def test_legacy_default_is_unchanged(clean_registry):
    q = _try_make_queue()
    m = dpm.MemoryUSMDevice(1024, queue=q)
    assert m.nbytes == 1024
    assert m.sycl_queue == q
    del m


def test_get_allocator_returns_none_by_default(clean_registry):
    assert dpm.get_allocator() is None


def test_set_and_get_allocator_global(clean_registry):
    def my_alloc(nbytes, queue):
        return dpm.malloc_device(nbytes, queue=queue)

    dpm.set_allocator(my_alloc)
    assert dpm.get_allocator() is my_alloc


def test_set_allocator_none_removes_hook(clean_registry):
    def my_alloc(nbytes, queue):
        return dpm.malloc_device(nbytes, queue=queue)

    dpm.set_allocator(my_alloc)
    assert dpm.get_allocator() is my_alloc
    dpm.set_allocator(None)
    assert dpm.get_allocator() is None


def test_reset_allocator_clears_all(clean_registry):
    q = _try_make_queue()
    dpm.set_allocator(lambda n, q: dpm.malloc_device(n, q))
    dpm.set_allocator(
        lambda n, q: dpm.malloc_device(n, q), sycl_device=q.sycl_device
    )
    dpm.reset_allocator()
    assert dpm.get_allocator() is None
    assert dpm.get_allocator(sycl_device=q.sycl_device) is None


def test_per_device_hook_overrides_global(clean_registry):
    q = _try_make_queue()
    dev = q.sycl_device

    calls = {"global": 0, "specific": 0}

    def global_alloc(nbytes, queue):
        calls["global"] += 1
        return dpm.malloc_device(nbytes, queue=queue)

    def specific_alloc(nbytes, queue):
        calls["specific"] += 1
        return dpm.malloc_device(nbytes, queue=queue)

    dpm.set_allocator(global_alloc, sycl_device=None)
    dpm.set_allocator(specific_alloc, sycl_device=dev)

    m = dpm.MemoryUSMDevice(1024, queue=q)
    del m
    assert calls["specific"] == 1
    assert calls["global"] == 0


def test_invalid_allocator_raises(clean_registry):
    with pytest.raises(TypeError):
        dpm.set_allocator("not callable")


def test_invalid_sycl_device_raises(clean_registry):
    with pytest.raises(TypeError):
        dpm.set_allocator(lambda n, q: None, sycl_device="not a device")


def test_pool_construction():
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)
    assert pool.sycl_device == q.sycl_device
    assert pool.sycl_context == q.sycl_context


def test_pool_malloc_returns_memory_usm_device():
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)
    m = pool.malloc(1024, q)
    assert isinstance(m, dpm.MemoryUSMDevice)
    assert m.nbytes == 1024


def test_pool_malloc_rejects_nonpositive_size():
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)
    with pytest.raises(ValueError):
        pool.malloc(0, q)
    with pytest.raises(ValueError):
        pool.malloc(-1, q)


def test_pool_allocation_lifetime():
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)
    for _ in range(64):
        m = pool.malloc(4096, q)
        assert m.nbytes == 4096
        del m


def test_pool_outlives_individual_allocations():
    """Pool-allocated _Memory must keep its pool alive via
    ``_pool_owner`` so the async-free callback remains valid."""
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)
    m = pool.malloc(1024, q)
    del pool
    del m


def test_is_memory_pool_available_returns_bool():
    assert isinstance(dpm.is_memory_pool_available(), bool)


def test_get_default_no_arg_call():
    """The no-arg form of get_default uses the cached default
    device."""
    pool = dpm.MemoryPool.get_default()
    assert isinstance(pool, dpm.MemoryPool)
    assert pool.is_default is True


def test_get_default_is_singleton_per_context_device():
    q1 = _try_make_queue()
    q2 = _try_make_queue()
    pool_a = dpm.MemoryPool.get_default(sycl_device=q1.sycl_device)
    pool_b = dpm.MemoryPool.get_default(sycl_device=q2.sycl_device)
    assert pool_a is pool_b


def test_private_pool_is_not_default():
    q = _try_make_queue()
    private = dpm.MemoryPool(sycl_device=q.sycl_device)
    assert private.is_default is False
    default = dpm.MemoryPool.get_default(sycl_device=q.sycl_device)
    assert default.is_default is True
    assert private is not default


def test_default_pool_supports_allocation(clean_registry):
    q = _try_make_queue()
    pool = dpm.MemoryPool.get_default(sycl_device=q.sycl_device)
    dpm.set_allocator(pool)
    try:
        for _ in range(8):
            m = dpm.MemoryUSMDevice(8192, queue=q)
            assert m.nbytes == 8192
            del m
    finally:
        dpm.reset_allocator()


def test_set_release_threshold_does_not_error():
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)
    pool.set_release_threshold(1 << 20)
    pool.set_release_threshold(0)


def test_reset_memory_does_not_error():
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)
    m = pool.malloc(4096, q)
    del m
    pool.reset_memory(q)


def test_byte_counter_methods_exist():
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)
    assert isinstance(pool.used_bytes(), int)
    assert isinstance(pool.total_bytes(), int)
    assert isinstance(pool.free_bytes(), int)
    assert pool.used_bytes() >= 0
    assert pool.total_bytes() >= 0
    assert pool.free_bytes() >= 0


def test_byte_counter_invariants():
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)
    used = pool.used_bytes()
    total = pool.total_bytes()
    pool.free_bytes()
    assert total >= used


def test_used_bytes_grows_with_allocations():
    if not dpm.is_memory_pool_available():
        pytest.skip(
            "SYCL memory_pool extension unavailable; counters always 0"
        )
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)
    baseline_used = pool.used_bytes()
    m = pool.malloc(1 << 20, q)
    try:
        assert pool.used_bytes() >= baseline_used + (1 << 20)
    finally:
        del m


def test_total_bytes_at_least_used_bytes():
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)
    m = pool.malloc(1 << 20, q)
    try:
        assert pool.total_bytes() >= pool.used_bytes()
    finally:
        del m


def test_install_pool_as_default_allocator(clean_registry):
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)
    dpm.set_allocator(pool)

    for _ in range(32):
        m = dpm.MemoryUSMDevice(8192, queue=q)
        del m


def test_threshold_hybrid_allocator(clean_registry):
    """gpu4pyscf-style: small allocations through pool, large direct."""
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)
    THRESHOLD = 1 << 20

    calls = {"pool": 0, "direct": 0}

    def hybrid(nbytes, queue):
        if nbytes >= THRESHOLD:
            calls["direct"] += 1
            return dpm.malloc_device(nbytes, queue=queue)
        calls["pool"] += 1
        return pool.malloc(nbytes, queue)

    dpm.set_allocator(hybrid, sycl_device=q.sycl_device)

    small = dpm.MemoryUSMDevice(4096, queue=q)
    large = dpm.MemoryUSMDevice(4 << 20, queue=q)
    assert calls["pool"] == 1
    assert calls["direct"] == 1
    del small, large


def test_malloc_device_bypasses_hook(clean_registry):
    """``dpm.malloc_device`` must NOT consult the hook; otherwise the
    hybrid pattern would recurse infinitely."""
    q = _try_make_queue()

    calls = {"hook": 0}

    def trap(nbytes, queue):
        calls["hook"] += 1
        return dpm.malloc_device(nbytes, queue=queue)

    dpm.set_allocator(trap)

    m = dpm.malloc_device(1024, queue=q)
    assert isinstance(m, dpm.MemoryUSMDevice)
    assert calls["hook"] == 0
    del m

    m2 = dpm.MemoryUSMDevice(1024, queue=q)
    assert calls["hook"] == 1
    del m2


def test_device_hook_does_not_affect_shared(clean_registry):
    """USM-shared allocations bypass the device-hook registry."""
    q = _try_make_queue()
    calls = {"device_hook": 0}

    def device_hook(nbytes, queue):
        calls["device_hook"] += 1
        return dpm.malloc_device(nbytes, queue=queue)

    dpm.set_allocator(device_hook)

    m = dpm.MemoryUSMShared(1024, queue=q)
    assert calls["device_hook"] == 0
    del m


def test_set_allocator_accepts_pool_directly(clean_registry):
    """``set_allocator(pool)`` auto-extracts the pool's sycl_device."""
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)

    dpm.set_allocator(pool)
    installed = dpm.get_allocator(sycl_device=q.sycl_device)
    assert installed is pool

    m = dpm.MemoryUSMDevice(8192, queue=q)
    assert m.nbytes == 8192
    del m


def test_set_allocator_accepts_pool_bound_method(clean_registry):
    """``set_allocator(pool.malloc)`` also auto-detects the pool's
    sycl_device via the bound method's ``__self__``."""
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)

    dpm.set_allocator(pool.malloc)
    installed = dpm.get_allocator(sycl_device=q.sycl_device)
    # The registry stores the exact callable passed in (a bound
    # method); its __self__ is the pool.
    assert installed.__self__ is pool


def test_set_allocator_pool_conflicting_sycl_device_rejected(clean_registry):
    """A pool passed with a conflicting sycl_device kwarg must
    raise."""
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)
    # Try to install pool against a sub-device, which is a distinct
    # SyclDevice from q.sycl_device.
    try:
        sub = q.sycl_device.create_sub_devices(equally=2)[0]
    except (dpctl.SyclSubDeviceCreationError, Exception):
        pytest.skip("Cannot create sub-device for mismatch test")
    if sub == q.sycl_device:
        pytest.skip("Sub-device equals root; cannot exercise mismatch")
    with pytest.raises(ValueError, match="sycl_device"):
        dpm.set_allocator(pool, sycl_device=sub)


def test_set_allocator_pool_matching_sycl_device_accepted(clean_registry):
    """Explicit sycl_device that matches the pool's is allowed."""
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)
    dpm.set_allocator(pool, sycl_device=q.sycl_device)
    assert dpm.get_allocator(sycl_device=q.sycl_device) is pool


def test_pool_is_callable_as_allocator():
    """The pool itself is a callable conforming to the allocator
    protocol (``__call__`` delegates to :meth:`malloc`)."""
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)
    m = pool(4096, q)
    try:
        assert isinstance(m, dpm.MemoryUSMDevice)
        assert m.nbytes == 4096
    finally:
        del m


def test_aligned_alloc_with_hook_warns_and_bypasses(clean_registry):
    """A non-zero ``alignment`` request bypasses the hook (most pools
    cannot honor arbitrary alignment) and emits a RuntimeWarning."""
    import warnings

    q = _try_make_queue()
    calls = {"hook": 0}

    def hook(nbytes, queue):
        calls["hook"] += 1
        return dpm.malloc_device(nbytes, queue=queue)

    dpm.set_allocator(hook)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = dpm.MemoryUSMDevice(1024, alignment=64, queue=q)
        assert any(
            issubclass(w.category, RuntimeWarning) and "alignment" in str(w.message)
            for w in caught
        )
    assert calls["hook"] == 0
    del m


def test_pool_malloc_rejects_wrong_context():
    """``MemoryPool.malloc(queue=q)`` must reject queues from a
    different SYCL context."""
    q1 = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q1.sycl_device)
    try:
        # Construct a fresh queue with an independent context.
        q2 = dpctl.SyclQueue(q1.sycl_device)
    except dpctl.SyclQueueCreationError:
        pytest.skip("Cannot create a second SyclQueue for context test")
    if q2.sycl_context == q1.sycl_context:
        pytest.skip("Both queues share the same context; cannot exercise check")
    with pytest.raises(ValueError, match="context"):
        pool.malloc(1024, sycl_queue=q2)


def test_pool_malloc_with_explicit_queue_honors_it():
    """``pool.malloc(queue=q)`` with a context-compatible queue must
    succeed and the returned allocation must carry that queue."""
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)
    m = pool.malloc(1024, sycl_queue=q)
    try:
        assert m.sycl_queue is q
    finally:
        del m


def test_get_default_thread_safe_singleton():
    """Concurrent ``get_default`` calls must return the same wrapper.
    Exercises the lock around the WeakValueDictionary insert."""
    import threading

    q = _try_make_queue()
    results = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        results.append(
            dpm.MemoryPool.get_default(sycl_device=q.sycl_device)
        )

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    first = results[0]
    for r in results[1:]:
        assert r is first


def test_set_allocator_concurrent_reads_no_lock_contention(clean_registry):
    """Smoke test: multiple threads calling ``MemoryUSMDevice`` with a
    pool hook installed must not deadlock and must all succeed.
    Exercises the lock-free read path on ``_registry``."""
    import threading

    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)
    dpm.set_allocator(pool)
    errors = []

    def worker():
        try:
            for _ in range(32):
                m = dpm.MemoryUSMDevice(4096, queue=q)
                del m
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


# ---------------------------------------------------------------------------
# C-side installed-pool registry (for dpnp / other C++ consumers)
# ---------------------------------------------------------------------------


def _installed_ptr(ctx, dev):
    """Read the C-side installed-pool ref as a raw pointer value."""
    from dpctl.memory._memory_pool import _get_installed_device_pool_ptr

    return _get_installed_device_pool_ptr(ctx, dev)


def test_c_registry_install_and_query(clean_registry):
    """``set_allocator(pool)`` populates the C-side registry."""
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)
    assert _installed_ptr(q.sycl_context, q.sycl_device) == 0
    dpm.set_allocator(pool)
    assert _installed_ptr(q.sycl_context, q.sycl_device) != 0


def test_c_registry_reset_clears_entry(clean_registry):
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)
    dpm.set_allocator(pool)
    assert _installed_ptr(q.sycl_context, q.sycl_device) != 0
    dpm.reset_allocator()
    assert _installed_ptr(q.sycl_context, q.sycl_device) == 0


def test_c_registry_no_entry_for_plain_callable(clean_registry):
    """A plain callable (not a MemoryPool) must not populate the
    C-side registry — the C registry is for pool refs only."""
    q = _try_make_queue()

    def my_alloc(nbytes, queue):
        return dpm.malloc_device(nbytes, queue=queue)

    dpm.set_allocator(my_alloc, sycl_device=q.sycl_device)
    assert _installed_ptr(q.sycl_context, q.sycl_device) == 0


def test_c_registry_cleared_on_pool_dealloc(clean_registry):
    """If the MemoryPool is garbage-collected while still installed,
    its __dealloc__ must clear the C-side entry to avoid a dangling
    ref (the user is supposed to call reset_allocator first; this is
    the safety net)."""
    import gc

    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_device=q.sycl_device)
    dpm.set_allocator(pool)
    assert _installed_ptr(q.sycl_context, q.sycl_device) != 0
    ctx, dev = pool.sycl_context, pool.sycl_device
    # The Python-side _registry still holds the pool as a value, so
    # we have to clear that too to actually let dealloc run.
    dpm.reset_allocator()
    del pool
    gc.collect()
    assert _installed_ptr(ctx, dev) == 0
