"""Generic async event stream with final-result promise.

Provides push/pull semantics for producer-consumer patterns where a stream
of typed events is emitted and a final result is extracted from a terminal
event.

Upstream reference: ``packages/ai/src/utils/event-stream.ts``
"""

# ruff: noqa: UP046 — Generic[T, R] required when inheriting AsyncIterator[T]

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Generic, TypeVar, override

from otter_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
)

T = TypeVar("T")
R = TypeVar("R")


class EventStream(Generic[T, R], AsyncIterator[T]):
    """Async-iterable event stream with a final-result promise.

    Producers call :meth:`push` to emit events and :meth:`end` to signal
    completion.  Consumers ``async for`` over the stream or call
    :meth:`result` to await the final extracted value.

    Parameters
    ----------
    is_complete:
        Predicate that returns ``True`` when an event is terminal
        (``"done"`` or ``"error"``).
    extract_result:
        Extracts the final ``R`` value from a terminal event.
    """

    def __init__(
        self,
        is_complete: Callable[[T], bool],
        extract_result: Callable[[T], R],
    ) -> None:
        self._is_complete = is_complete
        self._extract_result = extract_result
        self._queue: list[T] = []
        self._waiters: list[_Waiter[T]] = []
        self._done = False
        self._final_result: R | None = None
        self._final_result_set = False
        self._final_result_event: asyncio.Event | None = None

    # -- Producer API -------------------------------------------------------

    def push(self, event: T) -> None:
        """Emit an event to the stream.

        If the event is terminal (per ``is_complete``), the stream is marked
        as done and the final result is extracted.
        """
        if self._done:
            return

        if self._is_complete(event):
            self._done = True
            self._final_result = self._extract_result(event)
            self._final_result_set = True
            if self._final_result_event is not None:
                self._final_result_event.set()

        waiter = self._waiters.pop(0) if self._waiters else None
        if waiter is not None:
            waiter.resolve(event)
        else:
            self._queue.append(event)

    def end(self, result: R | None = None) -> None:
        """Signal that the stream is finished, optionally providing a result.

        Any consumers waiting for the next event are woken and receive
        ``StopAsyncIteration``.  The final-result promise resolves to
        *result* if provided.
        """
        self._done = True
        if result is not None:
            self._final_result = result
            self._final_result_set = True
            if self._final_result_event is not None:
                self._final_result_event.set()

        while self._waiters:
            self._waiters.pop(0).resolve_none()

    # -- Consumer API -------------------------------------------------------

    async def result(self) -> R:
        """Await the final result extracted from the terminal event.

        Returns once a terminal event has been pushed or :meth:`end` has
        been called.
        """
        if self._final_result_set:
            return self._final_result  # type: ignore[return-value]

        if self._final_result_event is None:
            self._final_result_event = asyncio.Event()

        await self._final_result_event.wait()
        return self._final_result  # type: ignore[return-value]

    # -- AsyncIterator ------------------------------------------------------

    @override
    async def __anext__(self) -> T:
        while True:
            if self._queue:
                return self._queue.pop(0)

            if self._done:
                raise StopAsyncIteration

            waiter: _Waiter[T] = _Waiter()
            self._waiters.append(waiter)
            await waiter.wait()
            if not waiter.has_value:
                raise StopAsyncIteration
            return waiter.value

    @override
    def __aiter__(self) -> AsyncIterator[T]:
        return self


class _Waiter[T]:
    """One-shot async holder for a single pushed value."""

    def __init__(self) -> None:
        self._event: asyncio.Event = asyncio.Event()
        self._value: T | None = None
        self._has_value: bool = False

    def resolve(self, value: T) -> None:
        self._value = value
        self._has_value = True
        self._event.set()

    def resolve_none(self) -> None:
        self._has_value = False
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()

    @property
    def has_value(self) -> bool:
        return self._has_value

    @property
    def value(self) -> T:
        return self._value  # type: ignore[return-value]


# ============================================================================
# AssistantMessageEventStream
# ============================================================================


class AssistantMessageEventStream(EventStream[AssistantMessageEvent, AssistantMessage]):
    """Concrete ``EventStream`` for LLM assistant message events.

    Terminal events:
    * ``"done"``  — extracts ``event.message``
    * ``"error"`` — extracts ``event.error``
    """

    def __init__(self) -> None:
        super().__init__(
            is_complete=lambda e: e.type in ("done", "error"),
            extract_result=self._extract,
        )

    @staticmethod
    def _extract(event: AssistantMessageEvent) -> AssistantMessage:
        if event.type == "done":
            return event.message
        if event.type == "error":
            return event.error
        raise RuntimeError("Unexpected event type for final result")


def create_assistant_message_event_stream() -> AssistantMessageEventStream:
    """Factory function for :class:`AssistantMessageEventStream`."""
    return AssistantMessageEventStream()
