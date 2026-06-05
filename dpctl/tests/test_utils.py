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

"""Defines unit test cases for utility functions."""

import pytest

import dpctl
import dpctl.utils


@pytest.mark.filterwarnings("ignore:.*:RuntimeWarning")
def test_onetrace_enabled():
    import os

    v_name = "PTI_ENABLE_COLLECTION"
    v_v = os.getenv(v_name, None)
    with dpctl.utils.onetrace_enabled():
        assert os.getenv(v_name, None) == "1"
    assert os.getenv(v_name, None) == v_v


def test_intel_device_info():
    try:
        d = dpctl.select_default_device()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Default device could not be created")
    descr = dpctl.utils.intel_device_info(d)
    assert isinstance(descr, dict)
    assert ("device_id" in descr) or not descr
    allowed_names = [
        "device_id",
        "gpu_slices",
        "gpu_eu_count",
        "gpu_eu_simd_width",
        "gpu_hw_threads_per_eu",
        "gpu_subslices_per_slice",
        "gpu_eu_count_per_subslice",
        "max_mem_bandwidth",
        "free_memory",
        "memory_clock_rate",
        "memory_bus_width",
    ]
    for descriptor_name in descr.keys():
        test = descriptor_name in allowed_names
        err_msg = f"Key '{descriptor_name}' is not recognized"
        assert test, err_msg


def test_intel_device_info_validation():
    invalid_device = {}
    with pytest.raises(TypeError):
        dpctl.utils.intel_device_info(invalid_device)


def test_order_manager():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created for default-selected device")
    _som = dpctl.utils.SequentialOrderManager
    _mngr = _som[q]
    assert isinstance(_mngr.num_host_task_events, int)
    assert isinstance(_mngr.num_submitted_events, int)
    assert isinstance(_mngr.submitted_events, list)
    assert isinstance(_mngr.host_task_events, list)
    _mngr.add_event_pair(dpctl.SyclEvent(), dpctl.SyclEvent())
    _mngr.add_event_pair([dpctl.SyclEvent()], dpctl.SyclEvent())
    _mngr.add_event_pair(dpctl.SyclEvent(), [dpctl.SyclEvent()])
    _mngr.wait()
    cpy = _mngr.__copy__()
    _som.clear()
    del cpy

    try:
        _passed = False
        _som[None]
    except TypeError:
        _passed = True
    finally:
        assert _passed


def test_no_op_order_manager():
    try:
        q = dpctl.SyclQueue(property=("in_order",))
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created for default-selected device")
    assert q.is_in_order
    _mngr = dpctl.utils.SequentialOrderManager[q]
    assert _mngr.num_host_task_events == 0
    assert _mngr.num_submitted_events == 0
    assert isinstance(_mngr.submitted_events, list)
    assert len(_mngr.submitted_events) == 0
    assert isinstance(_mngr.host_task_events, list)
    assert len(_mngr.host_task_events) == 0
    _mngr.add_event_pair(dpctl.SyclEvent(), dpctl.SyclEvent())
    _mngr.add_event_pair([dpctl.SyclEvent()], dpctl.SyclEvent())
    _mngr.add_event_pair(dpctl.SyclEvent(), [dpctl.SyclEvent()])
    # waits on queue
    _mngr.wait()
    cpy = _mngr.__copy__()
    del cpy


def test_no_op_order_manager_is_cached():
    try:
        q = dpctl.SyclQueue(property=("in_order",))
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created for default-selected device")
    _som = dpctl.utils.SequentialOrderManager
    # the no-op manager is cached on the queue and reused across accesses
    assert _som[q] is _som[q]
    # clear must wait on in-order queues without raising
    _som.clear()


def test_order_manager_dispatch():
    try:
        q_io = dpctl.SyclQueue(property=("in_order",))
        q_default = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created for default-selected device")
    from dpctl.utils._order_manager import _NoOpOrderManager

    # dpctl queues are always in-order, so every queue dispatches to the
    # no-op order manager.
    _som = dpctl.utils.SequentialOrderManager
    assert isinstance(_som[q_io], _NoOpOrderManager)
    assert isinstance(_som[q_default], _NoOpOrderManager)
    _som.clear()
