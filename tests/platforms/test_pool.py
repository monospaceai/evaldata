"""Tests for `ConnectionPool`."""

import inspect
import threading
from typing import Any, cast

import pytest

import evaldata.platforms.pool as platform_pool
from evaldata.platforms.base import TypeResolvingAdapter, execute_within_budget
from evaldata.platforms.pool import ConnectionPool, PoolUnavailableError
from evaldata.types import (
    DuckDBPlatformRef,
    ExecutionError,
    ExecutionFailure,
    ExecutionResult,
    ExecutionSuccess,
    PlatformRef,
    PoolPolicy,
)

pytestmark = pytest.mark.unit


class _FakeAdapter:
    def __init__(self) -> None:
        self.close_count = 0
        self.close_done = threading.Event()

    def execute(self, sql: str) -> ExecutionResult:  # pragma: no cover - never executed in these tests
        raise NotImplementedError

    def cancel(self) -> None:  # pragma: no cover - never executed in these tests
        raise NotImplementedError

    def close(self) -> None:
        self.close_count += 1
        self.close_done.set()


class _FakeParent:
    def __init__(self) -> None:
        self.closed = False
        self.close_done = threading.Event()

    def close(self) -> None:
        self.closed = True
        self.close_done.set()


class _FailingParent:
    def close(self) -> None:
        message = "parent close failed"
        raise RuntimeError(message)


class _PingAdapter(_FakeAdapter):
    def __init__(self, healthy: bool = True) -> None:
        super().__init__()
        self.healthy = healthy

    def ping(self) -> bool:
        return self.healthy


class _HangingAdapter(_FakeAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.finish = threading.Event()
        self.cancel_started = threading.Event()
        self.finish_cancel = threading.Event()

    def execute(self, sql: str) -> ExecutionResult:
        self.started.set()
        self.finish.wait()
        return ExecutionSuccess(rows=[], schema=None, latency_seconds=0.0)

    def cancel(self) -> None:
        self.cancel_started.set()
        self.finish_cancel.wait()


class _TypeAdapter(_FakeAdapter):
    def type_probe_sql(self, sql: str) -> str:
        return sql

    def types_from_probe(self, rows: list[dict[str, object]]) -> list[object] | ExecutionError:
        return []


class _ClassifierAdapter(_FakeAdapter):
    def __init__(self, disconnected: bool | Exception) -> None:
        super().__init__()
        self.disconnected = disconnected

    def is_disconnect(self, error: ExecutionError) -> bool:
        if isinstance(self.disconnected, Exception):
            raise self.disconnected
        return self.disconnected


class _NativeTimeoutAdapter(_FakeAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.timeouts: list[float] = []

    def execute(self, sql: str) -> ExecutionResult:
        msg = "pool watchdog must dispatch through execute_with_timeout"
        raise AssertionError(msg)

    def execute_with_timeout(self, sql: str, timeout_seconds: float) -> ExecutionResult:
        self.timeouts.append(timeout_seconds)
        return ExecutionSuccess(rows=[], schema=None, latency_seconds=0.0)


class _ReusableStateAdapter(_FakeAdapter):
    def __init__(self, reusable: bool | Exception) -> None:
        super().__init__()
        self.reusable = reusable

    def is_reusable(self) -> bool:
        if isinstance(self.reusable, Exception):
            raise self.reusable
        return self.reusable


class _FailingCloseAdapter(_FakeAdapter):
    def close(self) -> None:
        self.close_count += 1
        message = "close failed"
        raise RuntimeError(message)


class _RaisingPingAdapter(_FakeAdapter):
    def ping(self) -> bool:
        message = "ping failed"
        raise RuntimeError(message)


class _RaisingExecuteAdapter(_FakeAdapter):
    def execute(self, sql: str) -> ExecutionResult:
        message = "execute failed"
        raise RuntimeError(message)


class _RaisingCancelAdapter(_HangingAdapter):
    def cancel(self) -> None:
        self.cancel_started.set()
        message = "cancel failed"
        raise RuntimeError(message)


class _CompletingCancelAdapter(_HangingAdapter):
    def cancel(self) -> None:
        self.cancel_started.set()
        self.finish.set()


class _TrackingAdapter(_FakeAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_count = 0

    def cancel(self) -> None:
        self.cancel_count += 1

    def execute(self, sql: str) -> ExecutionResult:
        return ExecutionSuccess(rows=[], schema=None, latency_seconds=0.0)


class _BlockingCloseAdapter(_TrackingAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.close_started = threading.Event()
        self.allow_close = threading.Event()

    def close(self) -> None:
        self.close_started.set()
        self.allow_close.wait()
        super().close()


class _BlockingPingAdapter(_PingAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.ping_started = threading.Event()
        self.allow_ping = threading.Event()

    def ping(self) -> bool:
        self.ping_started.set()
        self.allow_ping.wait()
        return self.healthy


class _Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _ref() -> PlatformRef:
    return DuckDBPlatformRef(name="pool-test")


def _pool(max_size: int = 2, *, parent: _FakeParent | None = None) -> tuple[ConnectionPool, list[_FakeAdapter]]:
    built: list[_FakeAdapter] = []

    def factory() -> _FakeAdapter:
        member = _FakeAdapter()
        built.append(member)
        return member

    return ConnectionPool(_ref(), factory, max_size, parent=parent), built


class TestAcquireRelease:
    def test_lazily_builds_one_member_then_reuses_it(self) -> None:
        pool, built = _pool()
        first = pool.acquire()
        assert len(built) == 1
        pool.release(first)
        second = pool.acquire()
        assert second is first
        assert len(built) == 1

    def test_lease_delegates_adapter_attributes(self) -> None:
        pool, _ = _pool(max_size=1)
        lease = pool.acquire()

        assert lease.close_count == 0

    def test_grows_up_to_max_size(self) -> None:
        pool, built = _pool(max_size=2)
        a = pool.acquire()
        b = pool.acquire()
        assert a is not b
        assert len(built) == 2

    def test_blocks_until_a_member_is_released(self) -> None:
        pool, _ = _pool(max_size=1)
        held = pool.acquire()
        acquired_second: list[object] = []
        started = threading.Event()

        def waiter() -> None:
            started.set()
            acquired_second.append(pool.acquire())

        t = threading.Thread(target=waiter)
        t.start()
        started.wait()
        assert not t.join(timeout=0.1) and t.is_alive()
        pool.release(held)
        t.join(timeout=2)
        assert not t.is_alive()
        assert acquired_second == [held]


class TestUtility:
    def test_builds_utility_once_and_reuses_it(self) -> None:
        pool, built = _pool()
        u = pool.utility()
        assert pool.utility() is u
        assert built == [u]

    def test_utility_is_never_a_checkout_member(self) -> None:
        pool, _ = _pool()
        u = pool.utility()
        member = pool.acquire()
        assert member is not u

    def test_utility_waits_for_an_in_progress_health_check(self) -> None:
        adapter = _BlockingPingAdapter()
        pool = ConnectionPool(_ref(), lambda: adapter, policy=PoolPolicy(max_size=1, pre_ping=True))
        first: list[object] = []
        second: list[object] = []
        first_thread = threading.Thread(target=lambda: first.append(pool.utility()))
        first_thread.start()
        assert adapter.ping_started.wait(1)
        second_thread = threading.Thread(target=lambda: second.append(pool.utility()))
        second_thread.start()
        assert second_thread.is_alive()
        adapter.allow_ping.set()
        first_thread.join(1)
        second_thread.join(1)
        assert first == [adapter]
        assert second == [adapter]

    def test_unhealthy_utility_is_closed_and_rebuilt(self) -> None:
        first = _PingAdapter()
        second = _PingAdapter()
        adapters = [first, second]
        pool = ConnectionPool(_ref(), lambda: adapters.pop(0), policy=PoolPolicy(max_size=1, pre_ping=True))
        assert pool.utility() is first
        first.healthy = False
        assert pool.utility() is second
        assert first.close_count == 1

    def test_close_during_utility_creation_closes_the_unpublished_adapter(self) -> None:
        parent = _FakeParent()
        started = threading.Event()
        finish = threading.Event()
        built: list[_FakeAdapter] = []

        def factory() -> _FakeAdapter:
            started.set()
            finish.wait()
            adapter = _FakeAdapter()
            built.append(adapter)
            return adapter

        pool = ConnectionPool(_ref(), factory, max_size=1, parent=parent)
        errors: list[RuntimeError] = []

        def build_utility() -> None:
            try:
                pool.utility()
            except RuntimeError as error:
                errors.append(error)

        thread = threading.Thread(target=build_utility)
        thread.start()
        assert started.wait(1)
        pool.close()
        assert parent.closed is False
        finish.set()
        thread.join(1)
        assert not thread.is_alive()
        assert len(errors) == 1
        assert built[0].close_count == 1
        assert parent.closed is True


class TestClose:
    def test_closes_a_released_member(self) -> None:
        pool, built = _pool()
        lease = pool.acquire()
        pool.release(lease)

        pool.close()

        assert built[0].close_count == 1

    def test_closes_members_utility_and_parent(self) -> None:
        parent = _FakeParent()
        pool, built = _pool(parent=parent)
        member = pool.acquire()
        pool.utility()
        pool.close()
        assert built[0].close_count == 0
        assert built[1].close_count == 1
        assert parent.closed is False
        pool.release(member)
        assert built[0].close_done.wait(1)
        assert parent.close_done.wait(1)
        assert built[0].close_count == 1
        assert parent.closed is True

    def test_close_without_utility_or_parent(self) -> None:
        pool, built = _pool()
        member = pool.acquire()
        pool.close()
        assert built[0].close_count == 0
        pool.release(member)
        assert built[0].close_done.wait(1)
        assert built[0].close_count == 1

    def test_close_dedupes_a_shared_member_and_utility(self) -> None:
        shared = _FakeAdapter()
        pool = ConnectionPool(_ref(), lambda: shared, max_size=1)
        pool.acquire()
        pool.utility()
        pool.close()
        assert shared.close_count == 1

    def test_acquire_on_closed_pool_raises(self) -> None:
        pool, _ = _pool()
        pool.close()
        with pytest.raises(RuntimeError, match="is closed"):
            pool.acquire()

    def test_utility_on_closed_pool_raises(self) -> None:
        pool, built = _pool()
        pool.close()
        with pytest.raises(RuntimeError, match="is closed"):
            pool.utility()
        assert built == []

    def test_release_after_close_drops_member_without_double_close(self) -> None:
        pool, built = _pool(max_size=1)
        member = pool.acquire()
        pool.close()
        assert built[0].close_count == 0
        pool.release(member)
        assert built[0].close_done.wait(1)
        assert built[0].close_count == 1

    def test_checked_out_lease_rejects_execution_after_close(self) -> None:
        adapter = _TrackingAdapter()
        pool = ConnectionPool(_ref(), lambda: adapter, max_size=1)
        lease = pool.acquire()
        pool.close()
        result = lease.execute("SELECT 1")
        assert isinstance(result, ExecutionFailure)
        assert result.error.kind == "platform_unavailable"
        pool.release(lease)
        assert adapter.close_done.wait(1)

    def test_release_cleanup_does_not_wait_for_a_blocking_close(self) -> None:
        adapter = _BlockingCloseAdapter()
        pool = ConnectionPool(_ref(), lambda: adapter, max_size=1)
        lease = pool.acquire()
        cast(Any, lease)._member.poisoned = True
        returned = threading.Event()

        def release() -> None:
            pool.release(lease)
            returned.set()

        thread = threading.Thread(target=release)
        thread.start()
        assert adapter.close_started.wait(1)
        assert returned.wait(1)
        adapter.allow_close.set()
        assert adapter.close_done.wait(1)
        thread.join(1)

    def test_release_closes_adapter_before_deferred_parent(self) -> None:
        adapter = _BlockingCloseAdapter()
        parent = _FakeParent()
        pool = ConnectionPool(_ref(), lambda: adapter, max_size=1, parent=parent)
        lease = pool.acquire()
        pool.close()
        pool.release(lease)
        assert adapter.close_started.wait(1)
        assert parent.closed is False
        adapter.allow_close.set()
        assert adapter.close_done.wait(1)
        assert parent.close_done.wait(1)

    def test_closing_an_idle_lease_after_pool_close_closes_the_parent_in_order(self) -> None:
        adapter = _TrackingAdapter()
        parent = _FakeParent()
        pool = ConnectionPool(_ref(), lambda: adapter, max_size=1, parent=parent)
        lease = pool.acquire()
        pool.close()
        lease.close()
        assert adapter.close_count == 1
        assert parent.closed is True

    def test_deferred_parent_waits_for_every_asynchronous_member_cleanup(self) -> None:
        adapters = [_BlockingCloseAdapter(), _BlockingCloseAdapter()]
        parent = _FakeParent()
        pool = ConnectionPool(_ref(), lambda: adapters.pop(0), max_size=2, parent=parent)
        first = pool.acquire()
        second = pool.acquire()
        first_adapter = cast(Any, first)._member.adapter
        second_adapter = cast(Any, second)._member.adapter
        pool.close()
        pool.release(first)
        pool.release(second)
        assert first_adapter.close_started.wait(1)
        assert second_adapter.close_started.wait(1)
        second_adapter.allow_close.set()
        assert second_adapter.close_done.wait(1)
        assert parent.closed is False
        first_adapter.allow_close.set()
        assert first_adapter.close_done.wait(1)
        assert parent.close_done.wait(1)

    def test_close_wakes_a_blocked_waiter_with_an_error(self) -> None:
        pool, _ = _pool(max_size=1)
        pool.acquire()
        errors: list[Exception] = []
        started = threading.Event()

        def waiter() -> None:
            started.set()
            try:
                pool.acquire()
            except RuntimeError as e:
                errors.append(e)

        t = threading.Thread(target=waiter)
        t.start()
        started.wait()
        assert not t.join(timeout=0.1) and t.is_alive()
        pool.close()
        t.join(timeout=2)
        assert not t.is_alive()
        assert len(errors) == 1
        assert "is closed" in str(errors[0])

    def test_close_is_idempotent_and_tolerates_parent_cleanup_failure(self) -> None:
        pool = ConnectionPool(_ref(), _FakeAdapter, max_size=1, parent=_FailingParent())
        pool.close()
        pool.close()


class TestLifecycle:
    def test_pool_watchdog_dispatches_native_timeout_once(self) -> None:
        adapter = _NativeTimeoutAdapter()
        pool = ConnectionPool(_ref(), lambda: adapter, max_size=1)
        result = execute_within_budget(pool.acquire(), "SELECT 1", 0.125)
        assert isinstance(result, ExecutionSuccess)
        assert adapter.timeouts == [0.125]

    @pytest.mark.parametrize("reusable", [False, RuntimeError("state unavailable")])
    def test_release_retires_an_adapter_with_unreusable_local_state(self, reusable: bool | Exception) -> None:
        adapter = _ReusableStateAdapter(reusable)
        pool = ConnectionPool(_ref(), lambda: adapter, max_size=1)
        lease = pool.acquire()
        pool.release(lease)
        assert adapter.close_done.wait(1)
        assert adapter.close_count == 1

    def test_failed_pre_ping_retires_member_and_builds_replacement(self) -> None:
        adapters = [_PingAdapter(healthy=False), _PingAdapter(healthy=True)]
        pool = ConnectionPool(_ref(), lambda: adapters.pop(0), policy=PoolPolicy(max_size=1, pre_ping=True))
        pool.acquire()
        assert adapters == []

    def test_acquire_timeout_quarantines_a_blocked_pre_ping_until_it_stops(self) -> None:
        class Adapter(_BlockingPingAdapter):
            def __init__(self) -> None:
                super().__init__()
                self.closed = threading.Event()

            def close(self) -> None:
                super().close()
                self.closed.set()

        adapter = Adapter()
        pool = ConnectionPool(
            _ref(),
            lambda: adapter,
            policy=PoolPolicy(max_size=1, pre_ping=True, acquire_timeout_seconds=0.01),
        )
        errors: list[PoolUnavailableError] = []
        returned = threading.Event()

        def acquire() -> None:
            try:
                pool.acquire()
            except PoolUnavailableError as error:
                errors.append(error)
            finally:
                returned.set()

        thread = threading.Thread(target=acquire)
        thread.start()
        assert adapter.ping_started.wait(1)
        assert returned.wait(1)
        assert len(errors) == 1
        assert adapter.close_count == 0
        record = pool._members[0]  # noqa: SLF001
        assert record.state == "quarantined"
        adapter.allow_ping.set()
        assert adapter.closed.wait(1)
        thread.join(1)
        assert adapter.close_count == 1

    def test_lease_forwards_type_resolution_capability(self) -> None:
        pool = ConnectionPool(_ref(), _TypeAdapter, max_size=1)
        member = pool.acquire()
        assert isinstance(member, TypeResolvingAdapter)
        assert inspect.getattr_static(member, "type_probe_sql") is not None
        assert member.type_probe_sql("SELECT 1") == "SELECT 1"
        assert member.types_from_probe([]) == []

    def test_plain_lease_does_not_advertise_type_resolution(self) -> None:
        pool, _ = _pool(max_size=1)
        assert not isinstance(pool.acquire(), TypeResolvingAdapter)

    def test_timeout_quarantines_until_worker_and_cancel_finish(self) -> None:
        built: list[_HangingAdapter] = []

        def factory() -> _HangingAdapter:
            adapter = _HangingAdapter()
            built.append(adapter)
            return adapter

        pool = ConnectionPool(
            _ref(), factory, policy=PoolPolicy(max_size=1, max_quarantined=2, cancel_grace_seconds=0.01)
        )
        member = pool.acquire()
        result = execute_within_budget(member, "SELECT 1", 0.01, cancel_grace_seconds=0.01)
        first = built[0]
        assert isinstance(result, ExecutionFailure)
        assert result.error.kind == "budget_exceeded"
        assert first.started.is_set()
        assert first.cancel_started.is_set()
        pool.release(member)
        replacement = pool.acquire()
        assert replacement is not member
        assert len(built) == 2
        first.finish.set()
        assert first.close_count == 0
        first.finish_cancel.set()
        for _ in range(20):
            if first.close_count:
                break
            threading.Event().wait(0.01)
        assert first.close_count == 1

    def test_close_defers_shared_parent_until_quarantined_worker_finishes(self) -> None:
        adapter = _HangingAdapter()
        parent = _FakeParent()
        pool = ConnectionPool(
            _ref(), lambda: adapter, policy=PoolPolicy(max_size=1, cancel_grace_seconds=0.01), parent=parent
        )
        member = pool.acquire()
        execute_within_budget(member, "SELECT 1", 0.01, cancel_grace_seconds=0.01)
        pool.close()
        assert parent.closed is False
        adapter.finish.set()
        adapter.finish_cancel.set()
        for _ in range(20):
            if parent.closed:
                break
            threading.Event().wait(0.01)
        assert adapter.close_count == 1
        assert parent.closed is True

    def test_policy_grace_is_used_when_the_caller_omits_it(self) -> None:
        adapter = _HangingAdapter()
        pool = ConnectionPool(_ref(), lambda: adapter, policy=PoolPolicy(max_size=1, cancel_grace_seconds=0.0))
        results: list[ExecutionResult] = []
        returned = threading.Event()

        def run() -> None:
            results.append(execute_within_budget(pool.acquire(), "SELECT 1", 0.01))
            returned.set()

        thread = threading.Thread(target=run)
        thread.start()
        assert adapter.started.wait(1)
        assert adapter.cancel_started.wait(1)
        assert returned.wait(1)
        assert adapter.finish_cancel.is_set() is False
        assert isinstance(results[0], ExecutionFailure)
        adapter.finish.set()
        adapter.finish_cancel.set()
        thread.join(1)

    def test_close_cancels_active_non_watchdog_execution(self) -> None:
        adapter = _HangingAdapter()
        pool = ConnectionPool(_ref(), lambda: adapter, max_size=1)
        member = pool.acquire()
        thread = threading.Thread(target=lambda: member.execute("SELECT 1"))
        thread.start()
        assert adapter.started.wait(1)
        pool.close()
        assert adapter.cancel_started.wait(1)
        assert adapter.close_count == 0
        adapter.finish.set()
        adapter.finish_cancel.set()
        thread.join(1)
        assert not thread.is_alive()
        for _ in range(20):
            if adapter.close_count:
                break
            threading.Event().wait(0.01)
        assert adapter.close_count == 1

    def test_close_defers_parent_while_factory_is_creating_a_member(self) -> None:
        parent = _FakeParent()
        started = threading.Event()
        finish = threading.Event()
        built: list[_FakeAdapter] = []

        def factory() -> _FakeAdapter:
            started.set()
            finish.wait()
            adapter = _FakeAdapter()
            built.append(adapter)
            return adapter

        pool = ConnectionPool(_ref(), factory, max_size=1, parent=parent)
        errors: list[RuntimeError] = []

        def acquire() -> None:
            try:
                pool.acquire()
            except RuntimeError as error:
                errors.append(error)

        thread = threading.Thread(target=acquire)
        thread.start()
        assert started.wait(1)
        pool.close()
        assert parent.closed is False
        finish.set()
        thread.join(1)
        assert not thread.is_alive()
        assert len(errors) == 1
        assert built[0].close_count == 1
        assert parent.closed is True

    def test_close_during_member_creation_without_a_parent_rejects_publication(self) -> None:
        started = threading.Event()
        finish = threading.Event()
        built: list[_FakeAdapter] = []

        def factory() -> _FakeAdapter:
            started.set()
            finish.wait()
            adapter = _FakeAdapter()
            built.append(adapter)
            return adapter

        pool = ConnectionPool(_ref(), factory, max_size=1)
        errors: list[RuntimeError] = []

        def acquire() -> None:
            try:
                pool.acquire()
            except RuntimeError as error:
                errors.append(error)

        thread = threading.Thread(target=acquire)
        thread.start()
        assert started.wait(1)
        pool.close()
        finish.set()
        thread.join(1)
        assert len(errors) == 1
        assert built[0].close_count == 1

    def test_factory_failure_closes_a_parent_deferred_by_shutdown(self) -> None:
        parent = _FakeParent()
        started = threading.Event()
        fail = threading.Event()

        def factory() -> _FakeAdapter:
            started.set()
            fail.wait()
            message = "connection failed"
            raise RuntimeError(message)

        pool = ConnectionPool(_ref(), factory, max_size=1, parent=parent)
        errors: list[PoolUnavailableError] = []

        def acquire() -> None:
            try:
                pool.acquire()
            except PoolUnavailableError as error:
                errors.append(error)

        thread = threading.Thread(target=acquire)
        thread.start()
        assert started.wait(1)
        pool.close()
        assert parent.closed is False
        fail.set()
        thread.join(1)
        assert len(errors) == 1
        assert parent.closed is True

    def test_constructor_rejects_missing_or_conflicting_size(self) -> None:
        with pytest.raises(ValueError, match="requires max_size"):
            ConnectionPool(_ref(), _FakeAdapter)
        with pytest.raises(ValueError, match="must match"):
            ConnectionPool(_ref(), _FakeAdapter, max_size=1, policy=PoolPolicy(max_size=2))

    def test_factory_failure_is_typed_operational_error(self) -> None:
        def factory() -> _FakeAdapter:
            message = "connection failed"
            raise RuntimeError(message)

        pool = ConnectionPool(_ref(), factory, max_size=1)
        with pytest.raises(PoolUnavailableError, match="could not create") as error:
            pool.acquire()
        assert isinstance(error.value.__cause__, RuntimeError)

    def test_factory_returning_none_is_a_typed_operational_error(self) -> None:
        pool = ConnectionPool(_ref(), lambda: cast(Any, None), max_size=1)
        with pytest.raises(PoolUnavailableError, match="could not create") as error:
            pool.acquire()
        assert isinstance(error.value.__cause__, TypeError)

    def test_factory_thread_start_failure_releases_its_capacity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FailingThread:
            def __init__(self, **_: object) -> None:
                return

            def start(self) -> None:
                message = "thread unavailable"
                raise RuntimeError(message)

        monkeypatch.setattr(platform_pool.threading, "Thread", FailingThread)
        pool = ConnectionPool(_ref(), _FakeAdapter, max_size=1)
        with pytest.raises(PoolUnavailableError, match="could not create") as error:
            pool.acquire()
        assert isinstance(error.value.__cause__, RuntimeError)
        assert pool._creating == 0  # noqa: SLF001

    def test_factory_thread_start_failure_closes_a_deferred_parent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parent = _FakeParent()
        pool = ConnectionPool(_ref(), _FakeAdapter, max_size=1, parent=parent)

        class FailingThread:
            def __init__(self, **_: object) -> None:
                return

            def start(self) -> None:
                pool.close()
                message = "thread unavailable"
                raise RuntimeError(message)

        monkeypatch.setattr(platform_pool.threading, "Thread", FailingThread)

        with pytest.raises(PoolUnavailableError, match="could not create"):
            pool.acquire()

        assert parent.closed is True

    def test_cleanup_thread_start_failure_falls_back_to_inline_cleanup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FailingThread:
            def __init__(self, **_: object) -> None:
                return

            def start(self) -> None:
                message = "thread unavailable"
                raise RuntimeError(message)

        adapter = _TrackingAdapter()
        pool = ConnectionPool(_ref(), lambda: adapter, max_size=1)
        lease = pool.acquire()
        cast(Any, lease)._member.poisoned = True
        monkeypatch.setattr(platform_pool.threading, "Thread", FailingThread)
        pool.release(lease)
        assert adapter.close_count == 1
        assert pool._cleanup_pending == 0  # noqa: SLF001

    def test_factory_completion_wins_a_wait_timeout_race(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_event = threading.Event
        factory_done = real_event()

        class RacingEvent:
            def __init__(self) -> None:
                self._event = real_event()

            def is_set(self) -> bool:
                return self._event.is_set()

            def set(self) -> None:
                self._event.set()
                factory_done.set()

            def wait(self, timeout: float | None = None) -> bool:
                assert factory_done.wait(1)
                return False

        monkeypatch.setattr(platform_pool.threading, "Event", RacingEvent)
        pool = ConnectionPool(_ref(), _FakeAdapter, max_size=1)
        assert pool.acquire() is not None
        assert pool._creating == 0  # noqa: SLF001

    def test_acquire_timeout_is_typed_operational_error(self) -> None:
        pool, _ = _pool(max_size=1)
        pool.acquire()
        pool._policy = PoolPolicy(max_size=1, acquire_timeout_seconds=0.01)  # noqa: SLF001
        with pytest.raises(PoolUnavailableError, match="unavailable"):
            pool.acquire()

    def test_acquire_timeout_abandons_a_blocked_factory_without_releasing_capacity(self) -> None:
        class Adapter(_FakeAdapter):
            def __init__(self) -> None:
                super().__init__()
                self.closed = threading.Event()

            def close(self) -> None:
                super().close()
                self.closed.set()

        started = threading.Event()
        finish = threading.Event()
        built_ready = threading.Event()
        returned = threading.Event()
        attempts = 0
        built: list[Adapter] = []
        errors: list[PoolUnavailableError] = []

        def factory() -> Adapter:
            nonlocal attempts
            attempts += 1
            started.set()
            finish.wait()
            adapter = Adapter()
            built.append(adapter)
            built_ready.set()
            return adapter

        pool = ConnectionPool(_ref(), factory, policy=PoolPolicy(max_size=1, acquire_timeout_seconds=0.01))

        def acquire() -> None:
            try:
                pool.acquire()
            except PoolUnavailableError as error:
                errors.append(error)
            finally:
                returned.set()

        thread = threading.Thread(target=acquire)
        thread.start()
        assert started.wait(1)
        assert returned.wait(1)
        assert len(errors) == 1
        assert pool._creating == 1  # noqa: SLF001
        with pytest.raises(PoolUnavailableError, match="unavailable"):
            pool.acquire()
        assert attempts == 1
        finish.set()
        assert built_ready.wait(1)
        assert built[0].closed.wait(1)
        thread.join(1)
        assert built[0].close_count == 1
        assert pool._creating == 0  # noqa: SLF001
        assert pool._members == []  # noqa: SLF001

    def test_late_factory_completion_after_the_deadline_is_rejected(self) -> None:
        class Adapter(_FakeAdapter):
            def __init__(self) -> None:
                super().__init__()
                self.closed = threading.Event()

            def close(self) -> None:
                super().close()
                self.closed.set()

        clock = _Clock()
        started = threading.Event()
        finish = threading.Event()
        returned = threading.Event()
        built: list[Adapter] = []
        errors: list[PoolUnavailableError] = []

        def factory() -> Adapter:
            started.set()
            finish.wait()
            clock.value = 2.0
            adapter = Adapter()
            built.append(adapter)
            return adapter

        pool = ConnectionPool(_ref(), factory, policy=PoolPolicy(max_size=1, acquire_timeout_seconds=1.0), clock=clock)

        def acquire() -> None:
            try:
                pool.acquire()
            except PoolUnavailableError as error:
                errors.append(error)
            finally:
                returned.set()

        thread = threading.Thread(target=acquire)
        thread.start()
        assert started.wait(1)
        finish.set()
        assert returned.wait(1)
        assert len(errors) == 1
        assert built[0].closed.wait(1)
        thread.join(1)
        assert pool._members == []  # noqa: SLF001

    def test_close_defers_parent_until_an_abandoned_factory_failure_finishes(self) -> None:
        parent = _FakeParent()
        started = threading.Event()
        finish = threading.Event()
        returned = threading.Event()
        failed = threading.Event()
        errors: list[PoolUnavailableError] = []

        def factory() -> _FakeAdapter:
            started.set()
            finish.wait()
            failed.set()
            message = "connection failed"
            raise RuntimeError(message)

        pool = ConnectionPool(
            _ref(), factory, policy=PoolPolicy(max_size=1, acquire_timeout_seconds=0.01), parent=parent
        )

        def acquire() -> None:
            try:
                pool.acquire()
            except PoolUnavailableError as error:
                errors.append(error)
            finally:
                returned.set()

        thread = threading.Thread(target=acquire)
        thread.start()
        assert started.wait(1)
        assert returned.wait(1)
        assert len(errors) == 1
        pool.close()
        assert parent.closed is False
        finish.set()
        assert failed.wait(1)
        thread.join(1)
        assert parent.closed is True

    def test_lifetime_and_idle_expiry_retire_members(self) -> None:
        clock = _Clock()
        built: list[_FakeAdapter] = []

        def factory() -> _FakeAdapter:
            adapter = _FakeAdapter()
            built.append(adapter)
            return adapter

        pool = ConnectionPool(
            _ref(), factory, policy=PoolPolicy(max_size=1, max_lifetime_seconds=2.0, max_idle_seconds=1.0), clock=clock
        )
        member = pool.acquire()
        pool.release(member)
        clock.value = 1.0
        second = pool.acquire()
        assert built[0].close_count == 1
        pool.release(second)
        clock.value = 3.0
        assert pool.acquire() is not second
        assert len(built) == 3

    @pytest.mark.parametrize("disconnected", [True, False, RuntimeError("classifier failed")])
    def test_disconnect_classifier_only_retires_confirmed_disconnects(self, disconnected: bool | Exception) -> None:
        adapter = _ClassifierAdapter(disconnected)
        pool = ConnectionPool(_ref(), lambda: adapter, max_size=1)
        member = pool.acquire()
        result = ExecutionFailure(latency_seconds=0.0, error=ExecutionError(kind="query_failed", message="x"))
        pool._record_execution_result(cast(Any, member)._member, result)  # noqa: SLF001
        pool.release(member)
        assert adapter.close_count == int(disconnected is True)

    def test_raising_ping_retires_member(self) -> None:
        adapters = [_RaisingPingAdapter(), _FakeAdapter()]
        pool = ConnectionPool(_ref(), lambda: adapters.pop(0), policy=PoolPolicy(max_size=1, pre_ping=True))
        pool.acquire()
        assert adapters == []

    def test_unexpected_worker_exception_is_a_query_error(self) -> None:
        result = execute_within_budget(_RaisingExecuteAdapter(), "SELECT 1", 1)
        assert isinstance(result, ExecutionFailure)
        assert result.error.kind == "query_failed"
        assert result.error.message == "execute failed"

    def test_pool_watchdog_converts_worker_exceptions_and_returns_budget_error_after_grace(self) -> None:
        error_pool = ConnectionPool(_ref(), _RaisingExecuteAdapter, max_size=1)
        error = execute_within_budget(error_pool.acquire(), "SELECT 1", 1)
        assert isinstance(error, ExecutionFailure)
        assert error.error.kind == "query_failed"

        adapter = _CompletingCancelAdapter()
        pool = ConnectionPool(_ref(), lambda: adapter, max_size=1)
        result = execute_within_budget(pool.acquire(), "SELECT 1", 0.01, cancel_grace_seconds=1)
        assert adapter.cancel_started.wait(1)
        assert isinstance(result, ExecutionFailure)
        assert result.error.kind == "budget_exceeded"

    def test_pool_completion_at_or_after_deadline_returns_budget_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        now = {"value": 0.0}
        monkeypatch.setattr(platform_pool.time, "monotonic", lambda: now["value"])
        started = threading.Event()
        finish = threading.Event()

        class Adapter(_TrackingAdapter):
            def execute(self, sql: str) -> ExecutionResult:
                started.set()
                finish.wait()
                now["value"] = 1.0
                return ExecutionSuccess(rows=[], schema=None, latency_seconds=0.0)

        pool = ConnectionPool(_ref(), Adapter, max_size=1)
        lease = pool.acquire()
        results: list[ExecutionResult] = []
        returned = threading.Event()

        def execute() -> None:
            results.append(execute_within_budget(lease, "SELECT 1", 1.0, cancel_grace_seconds=0.0))
            returned.set()

        thread = threading.Thread(target=execute)
        thread.start()
        assert started.wait(1)
        finish.set()
        assert returned.wait(1)
        thread.join(1)
        assert isinstance(results[0], ExecutionFailure)
        assert results[0].error.kind == "budget_exceeded"

    def test_active_or_quarantined_lease_rejects_another_execution(self) -> None:
        adapter = _HangingAdapter()
        pool = ConnectionPool(_ref(), lambda: adapter, max_size=1)
        lease = pool.acquire()
        first_results: list[ExecutionResult] = []

        thread = threading.Thread(target=lambda: first_results.append(lease.execute("SELECT 1")))
        thread.start()
        assert adapter.started.wait(1)
        active = lease.execute("SELECT 2")
        assert isinstance(active, ExecutionFailure)
        assert active.error.kind == "platform_unavailable"
        adapter.finish.set()
        thread.join(1)
        assert isinstance(first_results[0], ExecutionSuccess)

        adapter = _HangingAdapter()
        pool = ConnectionPool(_ref(), lambda: adapter, max_size=1)
        lease = pool.acquire()
        execute_within_budget(lease, "SELECT 1", 0.01, cancel_grace_seconds=0.0)
        quarantined = execute_within_budget(lease, "SELECT 2", 1.0)
        assert isinstance(quarantined, ExecutionFailure)
        assert quarantined.error.kind == "platform_unavailable"
        adapter.finish.set()
        adapter.finish_cancel.set()
        assert adapter.close_done.wait(1)

    @pytest.mark.parametrize("action", ["release", "close"])
    def test_relinquishing_an_active_lease_quarantines_it(self, action: str) -> None:
        adapter = _HangingAdapter()
        pool = ConnectionPool(_ref(), lambda: adapter, max_size=1)
        lease = pool.acquire()
        thread = threading.Thread(target=lambda: lease.execute("SELECT 1"))
        thread.start()
        assert adapter.started.wait(1)
        if action == "release":
            pool.release(lease)
        else:
            lease.close()
        assert adapter.cancel_started.wait(1)
        assert cast(Any, lease)._member.state == "quarantined"
        adapter.finish.set()
        adapter.finish_cancel.set()
        thread.join(1)
        assert adapter.close_done.wait(1)

    def test_failed_pre_ping_cleanup_does_not_delay_replacement(self) -> None:
        class UnhealthyAdapter(_BlockingCloseAdapter):
            def ping(self) -> bool:
                return False

        unhealthy = UnhealthyAdapter()
        healthy = _PingAdapter()
        adapters: list[_FakeAdapter] = [unhealthy, healthy]
        pool = ConnectionPool(
            _ref(),
            lambda: adapters.pop(0),
            policy=PoolPolicy(max_size=1, max_quarantined=2, pre_ping=True),
        )
        assert pool.acquire() is not None
        assert unhealthy.close_started.wait(1)
        assert unhealthy.close_done.is_set() is False
        unhealthy.allow_close.set()
        assert unhealthy.close_done.wait(1)

    def test_blocked_cleanup_counts_toward_the_quarantine_ceiling(self) -> None:
        class UnhealthyAdapter(_BlockingCloseAdapter):
            def ping(self) -> bool:
                return False

        adapter = UnhealthyAdapter()
        pool = ConnectionPool(
            _ref(),
            lambda: adapter,
            policy=PoolPolicy(max_size=1, max_quarantined=1, pre_ping=True, acquire_timeout_seconds=0.01),
        )
        with pytest.raises(PoolUnavailableError, match="unavailable"):
            pool.acquire()
        assert adapter.close_started.wait(1)
        assert pool._cleanup_pending == 1  # noqa: SLF001
        adapter.allow_close.set()
        assert adapter.close_done.wait(1)

    def test_close_and_cancellation_failures_do_not_strand_state(self) -> None:
        adapter = _RaisingCancelAdapter()
        pool = ConnectionPool(_ref(), lambda: adapter, policy=PoolPolicy(max_size=1, cancel_grace_seconds=0.0))
        result = execute_within_budget(pool.acquire(), "SELECT 1", 0.01)
        assert isinstance(result, ExecutionFailure)
        adapter.finish.set()
        for _ in range(20):
            if adapter.close_count:
                break
            threading.Event().wait(0.01)
        assert adapter.close_count == 1
        failing = _FailingCloseAdapter()
        close_pool = ConnectionPool(_ref(), lambda: failing, max_size=1)
        failing_lease = close_pool.acquire()
        failing_lease.close()

    def test_double_and_foreign_release_are_noops(self) -> None:
        pool, built = _pool(max_size=1)
        member = pool.acquire()
        pool.release(member)
        pool.release(member)
        pool.release(_FakeAdapter())
        assert pool.acquire() is member
        assert built == [cast(Any, member)._member.adapter]  # noqa: SLF001

    def test_public_lease_cancel_forwards_and_close_retires_the_adapter(self) -> None:
        adapters = [_TrackingAdapter(), _TrackingAdapter()]
        pool = ConnectionPool(_ref(), lambda: adapters.pop(0), max_size=1)
        lease = pool.acquire()
        first = cast(Any, lease)._member.adapter
        lease.cancel()
        lease.close()
        pool.release(lease)
        replacement = pool.acquire()
        assert first.cancel_count == 1
        assert first.close_count == 1
        assert replacement is not lease

    def test_prepare_handles_close_and_state_change_races(self) -> None:
        adapter = _BlockingPingAdapter()
        pool = ConnectionPool(_ref(), lambda: adapter, policy=PoolPolicy(max_size=1, pre_ping=True))
        errors: list[RuntimeError] = []

        def acquire() -> None:
            try:
                pool.acquire()
            except RuntimeError as error:
                errors.append(error)

        thread = threading.Thread(target=acquire)
        thread.start()
        assert adapter.ping_started.wait(1)
        pool.close()
        adapter.allow_ping.set()
        thread.join(1)
        assert len(errors) == 1

        pool, _ = _pool(max_size=1)
        lease = pool.acquire()
        record = cast(Any, lease)._member
        record.state = "free"
        assert pool._prepare(record, pool._clock() + 1) is False  # noqa: SLF001

        adapter = _PingAdapter()
        pool = ConnectionPool(_ref(), lambda: adapter, policy=PoolPolicy(max_size=1, pre_ping=True))
        lease = pool.acquire()
        record = cast(Any, lease)._member
        pool.close()
        assert pool._ping_within_deadline(record, pool._clock() + 1) is False  # noqa: SLF001
        pool.release(lease)
        assert adapter.close_done.wait(1)

    def test_pre_ping_thread_start_failure_marks_validation_unhealthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = _PingAdapter()
        pool = ConnectionPool(_ref(), lambda: adapter, max_size=1)
        lease = pool.acquire()
        record = cast(Any, lease)._member
        pool._policy = PoolPolicy(max_size=1, pre_ping=True)  # noqa: SLF001

        class FailingThread:
            def __init__(self, **_: object) -> None:
                return

            def start(self) -> None:
                message = "thread unavailable"
                raise RuntimeError(message)

        monkeypatch.setattr(platform_pool.threading, "Thread", FailingThread)

        assert pool._ping_within_deadline(record, pool._clock() + 1) is False  # noqa: SLF001
        assert record.active is False

    def test_utility_health_check_handles_close_and_detached_state(self) -> None:
        adapter = _BlockingPingAdapter()
        adapter.allow_ping.set()
        pool = ConnectionPool(_ref(), lambda: adapter, policy=PoolPolicy(max_size=1, pre_ping=True))
        assert pool.utility() is adapter
        adapter.ping_started.clear()
        adapter.allow_ping.clear()
        errors: list[RuntimeError] = []

        def get_utility() -> None:
            try:
                pool.utility()
            except RuntimeError as error:
                errors.append(error)

        thread = threading.Thread(target=get_utility)
        thread.start()
        assert adapter.ping_started.wait(1)
        pool.close()
        adapter.allow_ping.set()
        thread.join(1)
        assert len(errors) == 1

        replacement = _PingAdapter()
        pool = ConnectionPool(_ref(), lambda: replacement, policy=PoolPolicy(max_size=1, pre_ping=True))
        lease = pool.acquire()
        record = cast(Any, lease)._member
        replacement.healthy = False
        pool._utility = None  # noqa: SLF001
        assert pool._prepare_utility(record) is False  # noqa: SLF001

    def test_retiring_a_free_member_removes_it_from_every_collection(self) -> None:
        pool, _ = _pool(max_size=1)
        lease = pool.acquire()
        pool.release(lease)
        record = cast(Any, lease)._member
        pool._retire(record)  # noqa: SLF001
        assert record not in pool._members  # noqa: SLF001
        assert record not in pool._free  # noqa: SLF001

    def test_duplicate_completion_after_shutdown_is_harmless(self) -> None:
        adapter = _TrackingAdapter()
        pool = ConnectionPool(_ref(), lambda: adapter, max_size=1)
        lease = pool.acquire()
        record = cast(Any, lease)._member
        worker_done = threading.Event()
        cancel_done = threading.Event()
        worker_done.set()
        cancel_done.set()
        record.worker_done = worker_done
        record.cancel_done = cancel_done
        record.state = "quarantined"
        pool.close()
        pool._execution_completed(record)  # noqa: SLF001
        pool._execution_completed(record)  # noqa: SLF001
        assert record not in pool._members  # noqa: SLF001

    def test_quarantine_closes_after_cancellation_then_worker_completion(self) -> None:
        adapter = _HangingAdapter()
        pool = ConnectionPool(_ref(), lambda: adapter, max_size=1)
        lease = pool.acquire()
        execute_within_budget(lease, "SELECT 1", 0.01, cancel_grace_seconds=0.0)
        assert adapter.cancel_started.wait(1)
        adapter.finish_cancel.set()
        assert adapter.close_count == 0
        adapter.finish.set()
        for _ in range(20):
            if adapter.close_count:
                break
            threading.Event().wait(0.01)
        assert adapter.close_count == 1

    def test_internal_cancellation_can_retire_a_deferred_checked_out_member(self) -> None:
        adapter = _TrackingAdapter()
        pool = ConnectionPool(_ref(), lambda: adapter, max_size=1)
        lease = pool.acquire()
        record = cast(Any, lease)._member
        pool.close()
        pool._start_cancellation(record)  # noqa: SLF001
        assert adapter.close_done.wait(1)
        assert adapter.cancel_count == 1

    def test_duplicate_cancellation_request_is_ignored(self) -> None:
        adapter = _TrackingAdapter()
        pool = ConnectionPool(_ref(), lambda: adapter, max_size=1)
        lease = pool.acquire()
        record = cast(Any, lease)._member
        scheduled = threading.Event()
        record.cancel_done = scheduled

        pool._start_cancellation(record)  # noqa: SLF001

        assert record.cancel_done is scheduled
        assert adapter.cancel_count == 0

    def test_quarantine_ceiling_bounds_replacement_creation(self) -> None:
        adapters = [_HangingAdapter(), _HangingAdapter()]
        pool = ConnectionPool(
            _ref(),
            lambda: adapters.pop(0),
            policy=PoolPolicy(max_size=1, max_quarantined=1, acquire_timeout_seconds=0.01),
        )
        first = pool.acquire()
        execute_within_budget(first, "SELECT 1", 0.01, cancel_grace_seconds=0.0)
        with pytest.raises(PoolUnavailableError, match="unavailable"):
            pool.acquire()
        cast(Any, first)._member.adapter.finish.set()  # noqa: SLF001
        cast(Any, first)._member.adapter.finish_cancel.set()  # noqa: SLF001
