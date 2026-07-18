"""`ConnectionPool`: a per-platform bounded pool of adapter sessions for concurrent execution."""

import threading
from collections.abc import Callable
from typing import Protocol

from evaldata.platforms.base import PlatformAdapter
from evaldata.types import PlatformRef


class _Closeable(Protocol):
    """A resource the pool closes on teardown (e.g. a shared parent connection)."""

    def close(self) -> None:
        """Release the underlying resource."""
        ...


class ConnectionPool:
    """A bounded pool of `PlatformAdapter` sessions for one platform name.

    A case acquires a member for its whole solve->execute->score pipeline and releases it
    afterwards. Members are built lazily by `factory` as concurrency demands them, so a serial
    caller acquires, releases, and reuses one member. `utility` is a dedicated adapter that is
    never handed out for checkout — it backs seeding and direct `resolve` use. `close` releases
    every member, the utility, and any shared parent connection.
    """

    def __init__(
        self,
        ref: PlatformRef,
        factory: Callable[[], PlatformAdapter],
        max_size: int,
        *,
        parent: _Closeable | None = None,
    ) -> None:
        """Bind the pool to a platform and its member factory.

        Args:
            ref: The platform reference this pool serves.
            factory: Builds one fresh session adapter — a new independent connection, or a
                cursor of the shared `parent`.
            max_size: The most members that may exist at once.
            parent: A shared connection every member and the utility wrap, closed by `close`,
                or `None` when each member owns an independent connection.
        """
        self.ref = ref
        self._factory = factory
        self._max_size = max_size
        self._parent = parent
        self._cond = threading.Condition()
        self._members: list[PlatformAdapter] = []
        self._free: list[PlatformAdapter] = []
        self._utility: PlatformAdapter | None = None
        self._closed = False

    def acquire(self) -> PlatformAdapter:
        """Check out an idle member, building one lazily up to `max_size`, else wait for a release.

        Returns:
            A member adapter reserved for the caller until it is passed back to `release`.

        Raises:
            RuntimeError: If the pool is closed, including while a caller is blocked waiting.
        """
        with self._cond:
            while True:
                if self._closed:
                    msg = f"connection pool for platform {self.ref.name!r} is closed"
                    raise RuntimeError(msg)
                if self._free:
                    return self._free.pop()
                if len(self._members) < self._max_size:
                    member = self._factory()
                    self._members.append(member)
                    return member
                self._cond.wait()

    def release(self, member: PlatformAdapter) -> None:
        """Return `member` to the free list and wake one waiter; drop it if the pool is closed.

        Args:
            member: The member previously returned by `acquire`. A member released after `close`
                is dropped rather than requeued — `close` already closed it.
        """
        with self._cond:
            if self._closed:
                return
            self._free.append(member)
            self._cond.notify()

    def utility(self) -> PlatformAdapter:
        """Return the dedicated utility adapter, building it once on first use.

        Returns:
            The utility adapter for direct use, never a checkout member.

        Raises:
            RuntimeError: If the pool is closed.
        """
        with self._cond:
            if self._closed:
                msg = f"connection pool for platform {self.ref.name!r} is closed"
                raise RuntimeError(msg)
            if self._utility is None:
                self._utility = self._factory()
            return self._utility

    def close(self) -> None:
        """Close every member, the utility, and any shared parent; wake blocked waiters.

        Idempotent, and safe only when no case holds a checked-out member.
        """
        with self._cond:
            self._closed = True
            self._cond.notify_all()
            adapters = [*self._members, self._utility]
            parent = self._parent
            self._members.clear()
            self._free.clear()
            self._utility = None
            self._parent = None
        for adapter in dict.fromkeys(a for a in adapters if a is not None):
            adapter.close()
        if parent is not None:
            parent.close()
