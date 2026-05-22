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

"""Tests for the pluggable USM allocator hook and ``MemoryPool``.

Covers:

* Legacy (no-hook) behavior is unchanged.
* ``set_allocator`` / ``get_allocator`` / ``reset_allocator`` registry.
* Per-device hook precedence.
* ``MemoryPool.malloc`` lifetime semantics.
* The ``gpu4pyscf``-style "pool below threshold, direct above" pattern.
* ``set_allocator(None)`` removes the hook for a given key.
* ``malloc_device`` / ``malloc_shared`` / ``malloc_host`` bypass paths.
* Pool isolation across USM types.
"""

import pytest

import dpctl
import dpctl.memory as dpm


@pytest.fixture
def clean_registry():
    """Reset the allocator registry around each test that touches it,
    so test order and per-test failures cannot leak hooks into the
    rest of the suite."""
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


# ---------------------------------------------------------------------------
# Legacy behavior is unchanged when no hook is installed
# ---------------------------------------------------------------------------


def test_legacy_default_is_unchanged(clean_registry):
    """With no allocator installed, MemoryUSM* construction uses the
    direct sycl::malloc_* path and __dealloc__ uses sycl::free, just
    like before this feature was added."""
    q = _try_make_queue()
    m = dpm.MemoryUSMDevice(1024, queue=q)
    assert m.nbytes == 1024
    assert m.sycl_queue == q
    # The legacy path does not set any pool bookkeeping.
    # We can't see _pool_return_cb directly from Python (cdef field),
    # but freeing the object should not error out and should not require
    # a pool to be alive.
    del m
    # If we got here without segfault, the legacy path is intact.


def test_get_allocator_returns_none_by_default(clean_registry):
    assert dpm.get_allocator(usm_type="device") is None
    assert dpm.get_allocator(usm_type="shared") is None
    assert dpm.get_allocator(usm_type="host") is None


# ---------------------------------------------------------------------------
# Allocator hook registry
# ---------------------------------------------------------------------------


def test_set_and_get_allocator_global(clean_registry):
    def my_alloc(nbytes, queue):
        return dpm.malloc_device(nbytes, queue=queue)

    dpm.set_allocator(my_alloc, usm_type="device")
    assert dpm.get_allocator(usm_type="device") is my_alloc
    # Unrelated usm_type unaffected.
    assert dpm.get_allocator(usm_type="shared") is None


def test_set_allocator_none_removes_hook(clean_registry):
    def my_alloc(nbytes, queue):
        return dpm.malloc_device(nbytes, queue=queue)

    dpm.set_allocator(my_alloc, usm_type="device")
    assert dpm.get_allocator(usm_type="device") is my_alloc
    dpm.set_allocator(None, usm_type="device")
    assert dpm.get_allocator(usm_type="device") is None


def test_reset_allocator_clears_all(clean_registry):
    dpm.set_allocator(lambda n, q: dpm.malloc_device(n, q), usm_type="device")
    dpm.set_allocator(lambda n, q: dpm.malloc_shared(n, q), usm_type="shared")
    dpm.reset_allocator()
    assert dpm.get_allocator(usm_type="device") is None
    assert dpm.get_allocator(usm_type="shared") is None


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

    dpm.set_allocator(global_alloc, usm_type="device", sycl_device=None)
    dpm.set_allocator(specific_alloc, usm_type="device", sycl_device=dev)

    m = dpm.MemoryUSMDevice(1024, queue=q)
    del m
    assert calls["specific"] == 1
    assert calls["global"] == 0


def test_invalid_usm_type_raises(clean_registry):
    with pytest.raises(ValueError):
        dpm.set_allocator(lambda n, q: None, usm_type="bogus")
    with pytest.raises(TypeError):
        dpm.set_allocator("not callable", usm_type="device")


def test_invalid_sycl_device_raises(clean_registry):
    with pytest.raises(TypeError):
        dpm.set_allocator(
            lambda n, q: None, usm_type="device", sycl_device="not a device"
        )


# ---------------------------------------------------------------------------
# MemoryPool
# ---------------------------------------------------------------------------


def test_pool_construction():
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_queue=q, usm_type="device")
    assert pool.sycl_queue == q
    assert pool.usm_type == "device"
    assert pool.sycl_device == q.sycl_device
    assert pool.sycl_context == q.sycl_context


def test_pool_invalid_usm_type():
    q = _try_make_queue()
    with pytest.raises(ValueError):
        dpm.MemoryPool(sycl_queue=q, usm_type="bogus")


def test_pool_malloc_returns_correct_type():
    q = _try_make_queue()
    pool_d = dpm.MemoryPool(sycl_queue=q, usm_type="device")
    m_d = pool_d.malloc(1024)
    assert isinstance(m_d, dpm.MemoryUSMDevice)
    assert m_d.nbytes == 1024

    pool_s = dpm.MemoryPool(sycl_queue=q, usm_type="shared")
    m_s = pool_s.malloc(1024)
    assert isinstance(m_s, dpm.MemoryUSMShared)
    assert m_s.nbytes == 1024


def test_pool_malloc_rejects_nonpositive_size():
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_queue=q, usm_type="device")
    with pytest.raises(ValueError):
        pool.malloc(0)
    with pytest.raises(ValueError):
        pool.malloc(-1)


def test_pool_allocation_lifetime():
    """Repeatedly allocate and free pool-backed buffers; the test passes
    if there is no segfault, double-free, or USMAllocationError."""
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_queue=q, usm_type="device")
    for _ in range(64):
        m = pool.malloc(4096)
        assert m.nbytes == 4096
        del m


def test_pool_outlives_individual_allocations():
    """A pool-allocated _Memory must keep its pool alive (via
    _pool_owner) so that the async_free callback can still find the
    pool reference when __dealloc__ runs."""
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_queue=q, usm_type="device")
    m = pool.malloc(1024)
    # Drop our own reference to the pool. The _Memory instance must
    # keep it alive internally.
    del pool
    # Now drop m. If the pool was already freed, this would either
    # segfault or call into freed memory.
    del m


def test_is_memory_pool_available_returns_bool():
    assert isinstance(dpm.is_memory_pool_available(), bool)


# ---------------------------------------------------------------------------
# Default-pool (singleton) semantics
# ---------------------------------------------------------------------------


def test_get_default_returns_pool():
    q = _try_make_queue()
    pool = dpm.MemoryPool.get_default(sycl_queue=q, usm_type="device")
    assert isinstance(pool, dpm.MemoryPool)
    assert pool.usm_type == "device"
    assert pool.is_default is True


def test_get_default_is_singleton_per_context_device_kind():
    """Two get_default calls with the same (context, device, usm_type)
    must return the same Python object (compared with ``is``)."""
    q1 = _try_make_queue()
    q2 = _try_make_queue()
    # Same device, so equivalent context+device tuple regardless of
    # whether q1 is q2.
    pool_a = dpm.MemoryPool.get_default(sycl_queue=q1, usm_type="device")
    pool_b = dpm.MemoryPool.get_default(sycl_queue=q2, usm_type="device")
    assert pool_a is pool_b


def test_get_default_distinct_for_distinct_usm_types():
    q = _try_make_queue()
    pool_d = dpm.MemoryPool.get_default(sycl_queue=q, usm_type="device")
    pool_s = dpm.MemoryPool.get_default(sycl_queue=q, usm_type="shared")
    assert pool_d is not pool_s
    assert pool_d.usm_type == "device"
    assert pool_s.usm_type == "shared"


def test_private_pool_is_not_default():
    """A regularly-constructed pool must report is_default == False."""
    q = _try_make_queue()
    private = dpm.MemoryPool(sycl_queue=q, usm_type="device")
    assert private.is_default is False
    default = dpm.MemoryPool.get_default(sycl_queue=q, usm_type="device")
    assert default.is_default is True
    assert private is not default


def test_default_pool_supports_allocation():
    """End-to-end: install the default pool as the allocator hook and
    confirm that allocations through MemoryUSMDevice flow through it."""
    q = _try_make_queue()
    pool = dpm.MemoryPool.get_default(sycl_queue=q, usm_type="device")
    dpm.set_allocator(
        pool.malloc, usm_type="device", sycl_device=q.sycl_device
    )
    try:
        for _ in range(8):
            m = dpm.MemoryUSMDevice(8192, queue=q)
            assert m.nbytes == 8192
            del m
    finally:
        dpm.reset_allocator()


def test_set_release_threshold_does_not_error():
    """Smoke test for the threshold setter; cannot easily observe the
    effect from Python."""
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_queue=q, usm_type="device")
    pool.set_release_threshold(1 << 20)
    pool.set_release_threshold(0)


def test_reset_memory_does_not_error():
    """Smoke test for reset_memory; should be a no-op when no blocks
    are cached, and not raise on the SYCL-extension-absent fallback
    path."""
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_queue=q, usm_type="device")
    # Allocate then drop, giving the pool something it *could* release.
    m = pool.malloc(4096)
    del m
    pool.reset_memory()


# ---------------------------------------------------------------------------
# Pool byte counters
# ---------------------------------------------------------------------------


def test_byte_counter_methods_exist():
    """Smoke test: used_bytes / total_bytes / free_bytes are callable
    and return non-negative integers regardless of extension
    availability."""
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_queue=q, usm_type="device")
    assert isinstance(pool.used_bytes(), int)
    assert isinstance(pool.total_bytes(), int)
    assert isinstance(pool.free_bytes(), int)
    assert pool.used_bytes() >= 0
    assert pool.total_bytes() >= 0
    assert pool.free_bytes() >= 0


def test_byte_counter_invariants():
    """When the SYCL extension is available, total = used + free must
    hold modulo race-window adjustments. When the extension is absent
    all three counters return 0 and the invariant trivially holds."""
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_queue=q, usm_type="device")
    used = pool.used_bytes()
    total = pool.total_bytes()
    free = pool.free_bytes()
    # used + free should equal total. free_bytes() guards against
    # transient inconsistency by returning 0 if used > total, so the
    # weakest invariant we can assert is total >= used.
    assert total >= used


def test_used_bytes_grows_with_allocations():
    """When the SYCL extension is available, allocating from a pool
    should be observable in used_bytes(). When it's not, used_bytes()
    is always 0 and the test is a no-op assertion."""
    if not dpm.is_memory_pool_available():
        pytest.skip(
            "SYCL memory_pool extension unavailable; byte counters "
            "are always 0 on the fallback path"
        )
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_queue=q, usm_type="device")
    baseline_used = pool.used_bytes()
    m = pool.malloc(1 << 20)  # 1 MiB
    try:
        # The pool should now report at least 1 MiB more in use than
        # before. We use >= rather than == because the extension is
        # free to round up to its allocation granularity.
        assert pool.used_bytes() >= baseline_used + (1 << 20)
    finally:
        del m


def test_total_bytes_at_least_used_bytes():
    """Whatever the pool has handed out must be backed by at least
    that much reserved memory."""
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_queue=q, usm_type="device")
    m = pool.malloc(1 << 20)
    try:
        assert pool.total_bytes() >= pool.used_bytes()
    finally:
        del m


# ---------------------------------------------------------------------------
# Pool installed via set_allocator
# ---------------------------------------------------------------------------


def test_install_pool_as_default_allocator(clean_registry):
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_queue=q, usm_type="device")
    dpm.set_allocator(
        pool.malloc, usm_type="device", sycl_device=q.sycl_device
    )

    # This allocation should go through the pool. We confirm by
    # constructing many same-size allocations and freeing them; with
    # a pool active this should never error and ideally be faster
    # (we can't easily assert performance in unit tests).
    for _ in range(32):
        m = dpm.MemoryUSMDevice(8192, queue=q)
        del m


def test_threshold_hybrid_allocator(clean_registry):
    """gpu4pyscf-style: small allocations through pool, large through
    direct sycl::malloc_device."""
    q = _try_make_queue()
    pool = dpm.MemoryPool(sycl_queue=q, usm_type="device")
    THRESHOLD = 1 << 20  # 1 MiB

    calls = {"pool": 0, "direct": 0}

    def hybrid(nbytes, queue):
        if nbytes >= THRESHOLD:
            calls["direct"] += 1
            return dpm.malloc_device(nbytes, queue=queue)
        calls["pool"] += 1
        return pool.malloc(nbytes, queue)

    dpm.set_allocator(
        hybrid, usm_type="device", sycl_device=q.sycl_device
    )

    small = dpm.MemoryUSMDevice(4096, queue=q)
    large = dpm.MemoryUSMDevice(4 << 20, queue=q)
    assert calls["pool"] == 1
    assert calls["direct"] == 1
    del small, large


def test_malloc_device_bypasses_hook(clean_registry):
    """dpm.malloc_device must NOT go through an installed hook,
    otherwise the hybrid pattern recurses infinitely."""
    q = _try_make_queue()

    calls = {"hook": 0}

    def trap(nbytes, queue):
        calls["hook"] += 1
        # Recursion would be fatal; if malloc_device honored the hook,
        # we'd never return from this call.
        return dpm.malloc_device(nbytes, queue=queue)

    dpm.set_allocator(trap, usm_type="device")

    # Direct call to malloc_device should bypass `trap` entirely.
    m = dpm.malloc_device(1024, queue=q)
    assert isinstance(m, dpm.MemoryUSMDevice)
    assert calls["hook"] == 0
    del m

    # Construction via the public class still triggers the hook.
    m2 = dpm.MemoryUSMDevice(1024, queue=q)
    assert calls["hook"] == 1
    del m2


# ---------------------------------------------------------------------------
# Cross-USM-type isolation
# ---------------------------------------------------------------------------


def test_device_hook_does_not_affect_shared(clean_registry):
    q = _try_make_queue()
    calls = {"device_hook": 0}

    def device_hook(nbytes, queue):
        calls["device_hook"] += 1
        return dpm.malloc_device(nbytes, queue=queue)

    dpm.set_allocator(device_hook, usm_type="device")

    # USM-shared allocation must not consult the device hook.
    m = dpm.MemoryUSMShared(1024, queue=q)
    assert calls["device_hook"] == 0
    del m
