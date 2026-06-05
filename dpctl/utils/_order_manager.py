import weakref
from collections import defaultdict
from contextvars import ContextVar
from typing import List, Protocol, runtime_checkable

from .._sycl_event import SyclEvent
from .._sycl_queue import SyclQueue
from ._seq_order_keeper import _OrderManager


@runtime_checkable
class OrderManagerProtocol(Protocol):
    """Interface implemented by order managers returned from
    :class:`.SyclQueueToOrderManagerMap`."""

    def add_event_pair(self, host_task_ev, comp_ev) -> None: ...

    @property
    def num_host_task_events(self) -> int: ...

    @property
    def num_submitted_events(self) -> int: ...

    @property
    def host_task_events(self) -> List[SyclEvent]: ...

    @property
    def submitted_events(self) -> List[SyclEvent]: ...

    def wait(self) -> None: ...

    def __copy__(self) -> "OrderManagerProtocol": ...


class _SequentialOrderManager:
    """
    Class to orchestrate default sequential order
    of the tasks offloaded from Python.
    """

    def __init__(self):
        self._state = _OrderManager(16)

    def __dealloc__(self):
        _local = self._state
        SyclEvent.wait_for(_local.get_submitted_events())
        SyclEvent.wait_for(_local.get_host_task_events())

    def add_event_pair(self, host_task_ev, comp_ev):
        _local = self._state
        if isinstance(host_task_ev, SyclEvent) and isinstance(
            comp_ev, SyclEvent
        ):
            _local.add_to_both_events(host_task_ev, comp_ev)
        else:
            if not isinstance(host_task_ev, (list, tuple)):
                host_task_ev = (host_task_ev,)
            if not isinstance(comp_ev, (list, tuple)):
                comp_ev = (comp_ev,)
            _local.add_vector_to_both_events(host_task_ev, comp_ev)

    @property
    def num_host_task_events(self):
        _local = self._state
        return _local.get_num_host_task_events()

    @property
    def num_submitted_events(self):
        _local = self._state
        return _local.get_num_submitted_events()

    @property
    def host_task_events(self):
        _local = self._state
        return _local.get_host_task_events()

    @property
    def submitted_events(self):
        _local = self._state
        return _local.get_submitted_events()

    def wait(self):
        _local = self._state
        return _local.wait()

    def __copy__(self):
        res = _SequentialOrderManager.__new__(_SequentialOrderManager)
        res._state = _OrderManager(self._state)
        return res


class _NoOpOrderManager:
    """Dummy order manager used when queue is in-order."""

    def __init__(self, q: SyclQueue):
        self._queue = q

    def add_event_pair(self, _host_task_ev, _comp_ev):
        pass

    @property
    def num_host_task_events(self):
        return 0

    @property
    def num_submitted_events(self):
        return 0

    @property
    def host_task_events(self):
        return []

    @property
    def submitted_events(self):
        return []

    def wait(self):
        # to imitate SequentialOrderManager, wait on the in-order queue
        self._queue.wait()

    def __copy__(self):
        return _NoOpOrderManager(self._queue)


class SyclQueueToOrderManagerMap:
    """Utility class used to ensure sequential ordering of offloaded tasks
    when passed to order manager."""

    def __init__(self):
        self._map = ContextVar(
            "global_order_manager_map",
            default=defaultdict(_SequentialOrderManager),
        )
        # No-op managers for in-order queues are cached on the queues
        # themselves; we keep weak references here so that ``clear`` can
        # still wait on the associated in-order queues at finalization.
        self._in_order_managers = weakref.WeakSet()

    def __getitem__(self, q: SyclQueue) -> OrderManagerProtocol:
        """Get order manager for given SyclQueue"""
        if not isinstance(q, SyclQueue):
            raise TypeError(f"Expected `dpctl.SyclQueue`, got {type(q)}")
        if q.is_in_order:
            # The NoOpOrderManager is stateless, so a single instance can be
            # reused for the lifetime of the queue. Cache it on the queue to
            # avoid allocating one on every access.
            mngr = q._no_op_order_manager
            if mngr is None:
                mngr = _NoOpOrderManager(q)
                q._no_op_order_manager = mngr
            self._in_order_managers.add(mngr)
            return mngr
        _local = self._map.get()
        if q in _local:
            return _local[q]
        else:
            v = _local[q]
            _local[q] = v
            return v

    def clear(self):
        """Clear content of internal dictionary"""
        _local = self._map.get()
        for v in _local.values():
            v.wait()
        _local.clear()
        # Wait on in-order queues too; their no-op managers are not stored in
        # ``_local`` but still need to be synchronized at finalization.
        for m in tuple(self._in_order_managers):
            m.wait()
        self._in_order_managers.clear()


SequentialOrderManager = SyclQueueToOrderManagerMap()


def _callback(som):
    som.clear()


f = weakref.finalize(SequentialOrderManager, _callback, SequentialOrderManager)
