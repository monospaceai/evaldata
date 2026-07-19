"""`ConnectionPool`: bounded platform sessions with explicit lifecycle state."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias, cast, runtime_checkable

from evaldata.platforms.base import (
    NativeTimeoutAdapter,
    PlatformAdapter,
    ReusableStateAdapter,
    TypeResolvingAdapter,
    execution_error,
)
from evaldata.types import ExecutionError, ExecutionResult, PlatformRef, PoolPolicy, SqlType


class _Closeable(Protocol):
    """A resource the pool closes on teardown (e.g. a shared parent connection)."""

    def close(self) -> None:
        """Release the underlying resource."""
        ...


class PoolUnavailableError(RuntimeError):
    """Failure to acquire a pool member before the deadline."""

    def __init__(self, platform_name: str, message: str) -> None:
        """Initialize the acquisition failure.

        Args:
            platform_name: The unavailable platform's configured name.
            message: Explanation of the acquisition failure.
        """
        super().__init__(message)
        self.platform_name = platform_name


@runtime_checkable
class ConnectionHealthAdapter(Protocol):
    """Capability for validating an idle adapter before it is reused."""

    def ping(self) -> bool:
        """Return whether the adapter is ready for another statement."""
        ...


@runtime_checkable
class DisconnectClassifier(Protocol):
    """Capability for identifying errors that make a session unsafe to reuse."""

    def is_disconnect(self, error: ExecutionError) -> bool:
        """Return whether `error` makes this session unsafe to reuse without performing I/O."""
        ...


_MemberState: TypeAlias = Literal["free", "checked_out", "quarantined", "retired"]


@dataclass
class _PoolMember:
    """One adapter and the pool state required to reuse or retire it safely."""

    adapter: PlatformAdapter
    created_at: float
    idle_since: float
    state: _MemberState = "free"
    poisoned: bool = False
    lease: _PoolLease | None = None
    active: bool = False
    worker_done: threading.Event | None = None
    cancel_done: threading.Event | None = None


@dataclass(frozen=True)
class _CreationSucceeded:
    """A connection factory result that is ready for publication."""

    adapter: PlatformAdapter
    completed_at: float


@dataclass(frozen=True)
class _CreationFailed:
    """A connection factory failure captured for the acquiring thread."""

    error: BaseException
    completed_at: float


_CreationOutcome: TypeAlias = _CreationSucceeded | _CreationFailed


@dataclass
class _PendingCreation:
    """One factory call whose result may arrive after its acquirer has timed out."""

    done: threading.Event
    outcome: _CreationOutcome | None = None
    abandoned: bool = False


class _PoolLease:
    """Checked-out adapter view with pool-managed lifecycle state."""

    def __init__(self, pool: ConnectionPool, member: _PoolMember) -> None:
        self._pool = pool
        self._member = member

    def execute(self, sql: str) -> ExecutionResult:
        done = threading.Event()
        if not self._pool._begin_execution(self._member, done):
            return self._pool._lease_unavailable()
        try:
            result = self._member.adapter.execute(sql)
            self._pool._record_execution_result(self._member, result)
            return result
        finally:
            self._pool._worker_finished(self._member, done)

    def cancel(self) -> None:
        """Request cancellation on the checked-out adapter."""
        self._member.adapter.cancel()

    def close(self) -> None:
        """Retire and close the checked-out adapter through its owning pool."""
        self._pool._close_lease(self._member)

    def execute_within_budget(
        self, sql: str, max_seconds: float, *, cancel_grace_seconds: float | None
    ) -> ExecutionResult:
        grace = self._pool._policy.cancel_grace_seconds if cancel_grace_seconds is None else cancel_grace_seconds
        return self._pool._execute_within_budget(self._member, sql, max_seconds, grace)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._member.adapter, name)


class _TypeResolvingPoolLease(_PoolLease):
    """A lease with statically discoverable type-resolution methods."""

    def type_probe_sql(self, sql: str) -> str:
        adapter = cast(TypeResolvingAdapter, self._member.adapter)
        return adapter.type_probe_sql(sql)

    def types_from_probe(self, rows: list[dict[str, Any]]) -> list[SqlType] | ExecutionError:
        adapter = cast(TypeResolvingAdapter, self._member.adapter)
        return adapter.types_from_probe(rows)


class ConnectionPool:
    """Bounded platform-session pool with quarantine-aware lifecycle handling."""

    def __init__(
        self,
        ref: PlatformRef,
        factory: Callable[[], PlatformAdapter],
        max_size: int | None = None,
        *,
        policy: PoolPolicy | None = None,
        parent: _Closeable | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize a pool.

        Args:
            ref: The platform reference this pool serves.
            factory: Builds a session adapter.
            max_size: Compatibility shorthand for `policy.max_size`.
            policy: Lifecycle limits, which must agree with `max_size` when both are set.
            parent: Optional shared connection.
            clock: Clock for lifecycle and acquisition deadlines.

        Raises:
            ValueError: If no size is supplied or supplied sizes disagree.
        """
        if policy is None:
            if max_size is None:
                msg = "ConnectionPool requires max_size or policy"
                raise ValueError(msg)
            policy = PoolPolicy(max_size=max_size)
        elif max_size is not None and max_size != policy.max_size:
            msg = "max_size must match policy.max_size"
            raise ValueError(msg)
        self.ref = ref
        self._factory = factory
        self._policy = policy
        self._max_quarantined = policy.max_quarantined or policy.max_size
        self._parent = parent
        self._clock = clock
        self._cond = threading.Condition()
        self._members: list[_PoolMember] = []
        self._free: list[_PoolMember] = []
        self._creating = 0
        self._cleanup_pending = 0
        self._utility: _PoolMember | None = None
        self._utility_busy = False
        self._closed = False
        self._parent_close_pending = False

    def acquire(self) -> PlatformAdapter:
        """Check out a healthy member.

        Returns:
            A lease reserved for the caller until `release`.

        Raises:
            RuntimeError: If the pool is closed.
        """
        deadline = self._clock() + self._policy.acquire_timeout_seconds
        while True:
            candidate = self._reserve_member(deadline)
            if candidate is None:
                candidate = self._create_member(deadline)
            if self._prepare(candidate, deadline):
                lease = candidate.lease
                if lease is None:  # pragma: no cover - members always receive their lease at publication
                    msg = "connection pool member has no lease"
                    raise RuntimeError(msg)
                return lease

    def release(self, member: PlatformAdapter) -> None:
        """Release or retire a lease.

        Args:
            member: The lease previously returned by `acquire`.
        """
        record = self._member_for(member)
        if record is None:
            return
        reusable = self._is_reusable(record.adapter)
        close: PlatformAdapter | None = None
        parent: _Closeable | None = None
        cancel = False
        with self._cond:
            if record.state == "quarantined":
                return
            if record.state != "checked_out":
                return
            if record.active:
                cancel = True
            elif self._closed or record.poisoned or not reusable or self._expired(record, self._clock()):
                self._remove_member_locked(record)
                close = record.adapter
                self._cleanup_pending += 1
                parent = self._take_deferred_parent_locked()
                self._cond.notify_all()
            else:
                record.state = "free"
                record.idle_since = self._clock()
                self._free.append(record)
                self._cond.notify()
        if close is not None:
            self._start_cleanup(close, parent)
        if cancel:
            self._start_cancellation(record)

    def utility(self) -> PlatformAdapter:
        """Return the dedicated utility adapter after lifecycle validation."""
        while True:
            with self._cond:
                if self._closed:
                    self._raise_closed()
                if self._utility_busy:
                    self._cond.wait()
                    continue
                member = self._utility
                self._utility_busy = True
            try:
                if member is None:
                    adapter = self._factory()
                    now = self._clock()
                    member = _PoolMember(adapter=adapter, created_at=now, idle_since=now, state="checked_out")
                    with self._cond:
                        if self._closed:
                            close = adapter
                        else:
                            self._utility = member
                            close = None
                    if close is not None:
                        self._close_safely(close)
                        self._raise_closed()
                if self._prepare_utility(member):
                    return member.adapter
            finally:
                self._finish_utility_reservation()

    def close(self) -> None:
        """Close idle and ordinary members; defer quarantined resources until workers exit."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._cond.notify_all()
            deferred = [
                member for member in self._members if member.state in {"checked_out", "quarantined"} or member.active
            ]
            adapters = [member.adapter for member in self._members if member not in deferred]
            for member in self._members:
                if member not in deferred:
                    member.state = "retired"
            if self._utility is not None:
                adapters.append(self._utility.adapter)
            self._members = deferred
            self._free.clear()
            self._utility = None
            parent = self._parent
            if deferred or self._creating or self._utility_busy:
                self._parent_close_pending = parent is not None
                parent = None
            else:
                self._parent = None
        for adapter in dict.fromkeys(adapters):
            self._close_safely(adapter)
        for member in deferred:
            if (member.active or member.state == "quarantined") and member.cancel_done is None:
                self._start_cancellation(member)
        if parent is not None:
            self._close_parent(parent)

    def _reserve_member(self, deadline: float) -> _PoolMember | None:
        with self._cond:
            while True:
                if self._closed:
                    self._raise_closed()
                if self._free:
                    member = self._free.pop()
                    member.state = "checked_out"
                    return member
                quarantined = sum(member.state == "quarantined" for member in self._members)
                active = sum(member.state != "quarantined" for member in self._members)
                unavailable = quarantined + self._cleanup_pending
                if unavailable < self._max_quarantined and active + self._creating < self._policy.max_size:
                    self._creating += 1
                    return None
                remaining = deadline - self._clock()
                if remaining <= 0:
                    msg = f"connection pool for platform {self.ref.name!r} is unavailable"
                    raise PoolUnavailableError(self.ref.name, msg)
                self._cond.wait(remaining)

    def _create_member(self, deadline: float) -> _PoolMember:
        pending = _PendingCreation(done=threading.Event())

        def build() -> None:
            try:
                adapter = self._factory()
                if adapter is None:
                    msg = "connection factory returned None"
                    raise TypeError(msg)
            except BaseException as exc:  # noqa: BLE001 - a factory thread must always release its reservation
                with self._cond:
                    outcome: _CreationOutcome = _CreationFailed(exc, self._clock())
                    pending.outcome = outcome
                    abandoned = pending.abandoned
                    pending.done.set()
            else:
                with self._cond:
                    outcome = _CreationSucceeded(adapter, self._clock())
                    pending.outcome = outcome
                    abandoned = pending.abandoned
                    pending.done.set()
            if abandoned:
                if isinstance(outcome, _CreationSucceeded):
                    self._close_safely(outcome.adapter)
                with self._cond:
                    self._creating -= 1
                    parent = self._take_deferred_parent_locked()
                    self._cond.notify_all()
                if parent is not None:
                    self._close_parent(parent)

        try:
            threading.Thread(target=build, daemon=True).start()
        except BaseException as error:  # noqa: BLE001 - a failed start must not strand pool capacity
            with self._cond:
                self._creating -= 1
                parent = self._take_deferred_parent_locked()
                self._cond.notify_all()
            if parent is not None:
                self._start_parent_cleanup(parent)
            msg = f"connection pool for platform {self.ref.name!r} could not create a connection"
            raise PoolUnavailableError(self.ref.name, msg) from error
        if not pending.done.wait(max(deadline - self._clock(), 0.0)):
            with self._cond:
                if not pending.done.is_set():
                    pending.abandoned = True
                    timed_out = True
                else:
                    timed_out = False
            if timed_out:
                msg = f"connection pool for platform {self.ref.name!r} is unavailable"
                raise PoolUnavailableError(self.ref.name, msg)

        with self._cond:
            outcome = cast(_CreationOutcome, pending.outcome)
            self._creating -= 1
            timed_out = outcome.completed_at >= deadline
            if timed_out or isinstance(outcome, _CreationFailed) or self._closed:
                self._cond.notify_all()
                publish = False
            else:
                adapter = outcome.adapter
                now = self._clock()
                member = _PoolMember(adapter=adapter, created_at=now, idle_since=now, state="checked_out")
                lease_type = _TypeResolvingPoolLease if isinstance(adapter, TypeResolvingAdapter) else _PoolLease
                member.lease = lease_type(self, member)
                self._members.append(member)
                self._cond.notify_all()
                publish = True
            cleanup_adapter = outcome.adapter if not publish and isinstance(outcome, _CreationSucceeded) else None
            if cleanup_adapter is not None:
                self._cleanup_pending += 1
            parent = self._take_deferred_parent_locked()
        if not publish:
            if cleanup_adapter is not None:
                self._start_cleanup(cleanup_adapter, parent)
            elif parent is not None:
                self._start_parent_cleanup(parent)
            if timed_out:
                msg = f"connection pool for platform {self.ref.name!r} is unavailable"
                raise PoolUnavailableError(self.ref.name, msg)
            if isinstance(outcome, _CreationFailed):
                msg = f"connection pool for platform {self.ref.name!r} could not create a connection"
                raise PoolUnavailableError(self.ref.name, msg) from outcome.error
            self._raise_closed()
        return member

    def _prepare(self, member: _PoolMember, deadline: float) -> bool:
        if self._expired(member, self._clock()):
            self._retire(member, asynchronous=True)
            return False
        if not self._ping_within_deadline(member, deadline):
            with self._cond:
                quarantined = member.state == "quarantined"
            if not quarantined:
                self._retire(member, asynchronous=True)
            return False
        with self._cond:
            if self._closed:
                closed = True
            elif member.state == "checked_out":
                closed = False
            else:
                return False
        if closed:
            self._retire(member, asynchronous=True)
            self._raise_closed()
        return True

    def _ping_within_deadline(self, member: _PoolMember, deadline: float) -> bool:
        if not self._policy.pre_ping or not isinstance(member.adapter, ConnectionHealthAdapter):
            return True
        done = threading.Event()
        healthy: list[bool] = []
        completed_at: list[float] = []
        if not self._begin_execution(member, done):
            return False

        def ping() -> None:
            try:
                healthy.append(self._ping(member.adapter))
            finally:
                completed_at.append(self._clock())
                self._worker_finished(member, done)

        try:
            threading.Thread(target=ping, daemon=True).start()
        except BaseException:  # noqa: BLE001 - a failed validation cannot prove the member healthy
            self._worker_finished(member, done)
            return False
        if done.wait(max(deadline - self._clock(), 0.0)) and completed_at[0] < deadline:
            return healthy[0]
        self._start_cancellation(member)
        return False

    def _prepare_utility(self, member: _PoolMember) -> bool:
        if not self._expired(member, self._clock()) and self._ping(member.adapter):
            with self._cond:
                if not self._closed and self._utility is member:
                    return True
            self._close_safely(member.adapter)
            self._raise_closed()
        with self._cond:
            if self._utility is member:
                self._utility = None
            self._cond.notify_all()
        self._close_safely(member.adapter)
        return False

    def _finish_utility_reservation(self) -> None:
        """Release the utility creation or validation reservation and close a deferred parent."""
        with self._cond:
            self._utility_busy = False
            parent = self._take_deferred_parent_locked()
            self._cond.notify_all()
        if parent is not None:
            self._close_parent(parent)

    def _execute_within_budget(
        self, member: _PoolMember, sql: str, max_seconds: float, cancel_grace_seconds: float
    ) -> ExecutionResult:
        start = time.monotonic()
        deadline = start + max_seconds
        done = threading.Event()
        outcome: list[ExecutionResult] = []
        completed_at: list[float] = []
        if not self._begin_execution(member, done):
            return self._lease_unavailable()

        def run() -> None:
            try:
                if isinstance(member.adapter, NativeTimeoutAdapter):
                    result = member.adapter.execute_with_timeout(sql, max_seconds)
                else:
                    result = member.adapter.execute(sql)
            except Exception as e:  # noqa: BLE001 - retain errors-as-values when an adapter violates its contract
                result = ExecutionResult(
                    rows=[], schema=None, latency_seconds=time.monotonic() - start, error=execution_error(e)
                )
            outcome.append(result)
            self._record_execution_result(member, result)
            completed_at.append(time.monotonic())
            self._worker_finished(member, done)

        threading.Thread(target=run, daemon=True).start()
        if done.wait(max(deadline - time.monotonic(), 0.0)) and completed_at[0] < deadline:
            return outcome[0]
        self._start_cancellation(member)
        grace_deadline = deadline + max(cancel_grace_seconds, 0.0)
        if done.wait(max(grace_deadline - time.monotonic(), 0.0)):
            return self._budget_exceeded(start, max_seconds)
        return self._budget_exceeded(start, max_seconds)

    def _start_cancellation(self, member: _PoolMember) -> None:
        """Atomically quarantine a timed-out member before its asynchronous cancellation starts."""
        cancel_done = threading.Event()
        with self._cond:
            if member.state not in {"checked_out", "quarantined"} or member.cancel_done is not None:
                return
            member.cancel_done = cancel_done
            member.state = "quarantined"
            self._cond.notify_all()
        self._start_cancel(member.adapter, member, cancel_done)

    def _worker_finished(self, member: _PoolMember, done: threading.Event) -> None:
        """Publish worker completion only after its member state is no longer active."""
        with self._cond:
            member.active = False
        done.set()
        self._execution_completed(member)

    def _begin_execution(self, member: _PoolMember, worker_done: threading.Event | None = None) -> bool:
        with self._cond:
            if self._closed or member.state != "checked_out" or member.active or member not in self._members:
                return False
            member.active = True
            member.worker_done = worker_done
            return True

    def _execution_completed(self, member: _PoolMember) -> None:
        """Close deferred members after all active execution and cancellation has stopped."""
        with self._cond:
            worker_done = member.worker_done
            cancel_done = member.cancel_done
            worker_finished = worker_done is None or worker_done.is_set()
            cancel_finished = cancel_done is None or cancel_done.is_set()
            if not worker_finished or not cancel_finished:
                return
            member.active = False
            if member.state != "quarantined" and not self._closed:
                return
            self._remove_member_locked(member)
            parent = self._take_deferred_parent_locked()
            self._cond.notify_all()
        self._close_safely(member.adapter)
        if parent is not None:
            self._close_parent(parent)

    def _record_execution_result(self, member: _PoolMember, result: ExecutionResult) -> None:
        """Mark a member poisoned only when its adapter classifies the result as a disconnect."""
        error = result.error
        if error is None or not isinstance(member.adapter, DisconnectClassifier):
            return
        try:
            disconnected = member.adapter.is_disconnect(error)
        except Exception:  # noqa: BLE001 - an unavailable classifier cannot prove the session unsafe
            return
        if disconnected:
            with self._cond:
                member.poisoned = True

    def _ping(self, adapter: PlatformAdapter) -> bool:
        if not self._policy.pre_ping or not isinstance(adapter, ConnectionHealthAdapter):
            return True
        try:
            return adapter.ping()
        except Exception:  # noqa: BLE001 - failed validation means the adapter is not reusable
            return False

    @staticmethod
    def _is_reusable(adapter: PlatformAdapter) -> bool:
        if not isinstance(adapter, ReusableStateAdapter):
            return True
        try:
            return adapter.is_reusable()
        except Exception:  # noqa: BLE001 - an unavailable local-state check cannot prove reuse safety
            return False

    def _expired(self, member: _PoolMember, now: float) -> bool:
        """Return whether lifecycle age or idle limits require retirement."""
        lifetime = self._policy.max_lifetime_seconds
        idle = self._policy.max_idle_seconds
        return (lifetime is not None and now - member.created_at >= lifetime) or (
            idle is not None and now - member.idle_since >= idle
        )

    def _retire(self, member: _PoolMember, *, asynchronous: bool = False) -> None:
        """Remove and close a member without taking the condition lock during close."""
        with self._cond:
            if member.state == "retired":
                return
            self._remove_member_locked(member)
            if asynchronous:
                self._cleanup_pending += 1
            parent = self._take_deferred_parent_locked()
            self._cond.notify_all()
        if asynchronous:
            self._start_cleanup(member.adapter, parent)
        else:
            self._close_safely(member.adapter)
            if parent is not None:
                self._close_parent(parent)

    def _close_lease(self, member: _PoolMember) -> None:
        """Retire an idle lease or quarantine and cancel one whose worker is still active."""
        with self._cond:
            active = member.active and member.state == "checked_out"
        if active:
            self._start_cancellation(member)
        else:
            self._retire(member)

    def _remove_member_locked(self, member: _PoolMember) -> None:
        """Remove `member` from tracked collections while holding the condition lock."""
        member.state = "retired"
        if member in self._members:
            self._members.remove(member)
        if member in self._free:
            self._free.remove(member)

    def _member_for(self, adapter: PlatformAdapter) -> _PoolMember | None:
        if isinstance(adapter, _PoolLease) and adapter._pool is self:
            return adapter._member
        with self._cond:
            return next((member for member in self._members if member.adapter is adapter), None)

    def _take_deferred_parent_locked(self) -> _Closeable | None:
        """Return the deferred parent once no quarantined worker can still use it."""
        if (
            not self._parent_close_pending
            or self._members
            or self._creating
            or self._cleanup_pending
            or self._utility_busy
        ):
            return None
        self._parent_close_pending = False
        parent = self._parent
        self._parent = None
        return parent

    def _raise_closed(self) -> None:
        msg = f"connection pool for platform {self.ref.name!r} is closed"
        raise RuntimeError(msg)

    def _lease_unavailable(self) -> ExecutionResult:
        return ExecutionResult(
            rows=[],
            schema=None,
            latency_seconds=0.0,
            error=ExecutionError(
                kind="platform_unavailable",
                message=f"connection pool lease for platform {self.ref.name!r} is unavailable",
            ),
        )

    @staticmethod
    def _budget_exceeded(start: float, max_seconds: float) -> ExecutionResult:
        return ExecutionResult(
            rows=[],
            schema=None,
            latency_seconds=time.monotonic() - start,
            error=ExecutionError(
                kind="budget_exceeded", message=f"exceeded cost budget: query did not complete within {max_seconds}s"
            ),
        )

    @staticmethod
    def _close_safely(adapter: PlatformAdapter) -> None:
        """Close one adapter without allowing teardown failures to strand pool state."""
        try:
            adapter.close()
        except Exception:  # noqa: BLE001 - closing a broken driver must not block pool cleanup
            return

    @staticmethod
    def _close_parent(parent: _Closeable) -> None:
        """Close the shared parent after all dependent members are gone."""
        try:
            parent.close()
        except Exception:  # noqa: BLE001 - parent cleanup is best effort
            return

    def _start_cleanup(
        self,
        adapter: PlatformAdapter,
        parent: _Closeable | None = None,
    ) -> None:
        """Close detached resources asynchronously, preserving adapter-before-parent ordering."""

        def cleanup() -> None:
            self._close_safely(adapter)
            with self._cond:
                self._cleanup_pending -= 1
                deferred_parent = self._take_deferred_parent_locked()
                self._cond.notify_all()
            for resource in dict.fromkeys(item for item in (parent, deferred_parent) if item is not None):
                self._close_parent(resource)

        try:
            threading.Thread(target=cleanup, daemon=True).start()
        except RuntimeError:
            cleanup()

    def _start_parent_cleanup(self, parent: _Closeable) -> None:
        """Close a detached shared parent asynchronously."""
        try:
            threading.Thread(target=self._close_parent, args=(parent,), daemon=True).start()
        except RuntimeError:
            self._close_parent(parent)

    def _start_cancel(self, adapter: PlatformAdapter, member: _PoolMember, done: threading.Event) -> None:
        """Run cancellation independently so a blocking driver cannot delay the caller."""

        def cancel() -> None:
            try:
                adapter.cancel()
            except Exception:  # noqa: BLE001 - cancellation is best effort
                pass
            finally:
                done.set()
                self._execution_completed(member)

        threading.Thread(target=cancel, daemon=True).start()
