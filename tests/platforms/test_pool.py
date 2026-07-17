"""Unit tests for `ConnectionPool` — acquire/release/reuse, lazy growth, blocking, and close."""

import threading

import pytest

from evaldata.platforms.pool import ConnectionPool
from evaldata.types import ExecutionResult, PlatformRef

pytestmark = pytest.mark.unit


class _FakeAdapter:
    """A stand-in `PlatformAdapter` that only tracks how often it is closed."""

    def __init__(self) -> None:
        self.close_count = 0

    def execute(self, sql: str) -> ExecutionResult:  # pragma: no cover - never executed in these tests
        raise NotImplementedError

    def cancel(self) -> None:  # pragma: no cover - never executed in these tests
        raise NotImplementedError

    def close(self) -> None:
        self.close_count += 1


class _FakeParent:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _ref() -> PlatformRef:
    return PlatformRef(name="pool-test", kind="duckdb")


def _pool(max_size: int = 2, *, parent: _FakeParent | None = None) -> tuple[ConnectionPool, list[_FakeAdapter]]:
    """Build a pool whose factory records every member it builds."""
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
        assert len(built) == 1  # released member reused; no new build

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
        # The waiter is blocked on acquire (pool exhausted); releasing wakes it deterministically.
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


class TestClose:
    def test_closes_members_utility_and_parent(self) -> None:
        parent = _FakeParent()
        pool, built = _pool(parent=parent)
        member = pool.acquire()
        utility = pool.utility()
        pool.close()
        assert member.close_count == 1
        assert utility.close_count == 1
        assert parent.closed is True

    def test_close_without_utility_or_parent(self) -> None:
        pool, _ = _pool()
        member = pool.acquire()
        pool.close()
        assert member.close_count == 1

    def test_close_dedupes_a_shared_member_and_utility(self) -> None:
        shared = _FakeAdapter()
        pool = ConnectionPool(_ref(), lambda: shared, max_size=1)
        pool.acquire()
        pool.utility()
        pool.close()
        assert shared.close_count == 1  # closed once despite being both member and utility

    def test_acquire_on_closed_pool_raises(self) -> None:
        pool, _ = _pool()
        pool.close()
        with pytest.raises(RuntimeError, match="is closed"):
            pool.acquire()

    def test_close_wakes_a_blocked_waiter_with_an_error(self) -> None:
        pool, _ = _pool(max_size=1)
        pool.acquire()  # exhaust the pool
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
