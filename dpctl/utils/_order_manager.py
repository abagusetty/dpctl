import weakref
from typing import List, Protocol, runtime_checkable

from .._sycl_event import SyclEvent
from .._sycl_queue import SyclQueue


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


class _NoOpOrderManager:
    """Order manager used for in-order queues.

    dpctl queues are always in-order, so no SYCL event book-keeping is
    required: the queue itself guarantees that tasks execute in submission
    order. All event accessors are therefore empty and ``add_event_pair`` is a
    no-op.
    """

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
        # to imitate prior behavior, wait on the in-order queue
        self._queue.wait()

    def __copy__(self):
        return _NoOpOrderManager(self._queue)


class SyclQueueToOrderManagerMap:
    """Utility class returning an order manager for a given queue.

    dpctl queues are always in-order, so a stateless no-op order manager is
    returned for every queue and cached on the queue itself.
    """

    def __init__(self):
        # No-op managers are cached on the queues themselves; weak references
        # are kept here so that ``clear`` can still wait on the associated
        # queues at finalization.
        self._in_order_managers = weakref.WeakSet()

    def __getitem__(self, q: SyclQueue) -> OrderManagerProtocol:
        """Get order manager for given SyclQueue"""
        if not isinstance(q, SyclQueue):
            raise TypeError(f"Expected `dpctl.SyclQueue`, got {type(q)}")
        # The NoOpOrderManager is stateless, so a single instance can be reused
        # for the lifetime of the queue. Cache it on the queue to avoid
        # allocating one on every access.
        mngr = q._no_op_order_manager
        if mngr is None:
            mngr = _NoOpOrderManager(q)
            q._no_op_order_manager = mngr
        self._in_order_managers.add(mngr)
        return mngr

    def clear(self):
        """Wait on the queues tracked by this map."""
        for m in tuple(self._in_order_managers):
            m.wait()
        self._in_order_managers.clear()


SequentialOrderManager = SyclQueueToOrderManagerMap()


def _callback(som):
    som.clear()


f = weakref.finalize(SequentialOrderManager, _callback, SequentialOrderManager)
