"""Plain-text, per-run diagnostics for the sync CLI and its captured logs."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import Event, Lock, Thread
from time import perf_counter


class SyncProgress:
    def __init__(self, emit: Callable[[str], None], interval: float = 15.0) -> None:
        self.emit = emit
        self.interval = interval
        self.seconds: dict[str, float] = defaultdict(float)
        self._lock = Lock()
        self._stop = Event()
        self._thread = Thread(target=self._heartbeat, name="sync-progress", daemon=True)
        self._active: tuple[str, float] | None = None
        self._operation: str | None = None
        self._done = 0
        self._total: int | None = None

    def __enter__(self) -> SyncProgress:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def note(self, message: str) -> None:
        with self._lock:
            self.emit(f"[index sync] {message}")

    @contextmanager
    def phase(self, label: str, *, total: int | None = None) -> Iterator[None]:
        started = perf_counter()
        with self._lock:
            self._active = (label, started)
            self._done = 0
            self._total = total
            self.emit(f"[index sync] {label}: starting")
        status = "failed"
        try:
            yield
            status = "done"
        finally:
            elapsed = perf_counter() - started
            with self._lock:
                self._active = None
                self.seconds[label] += elapsed
                self.emit(f"[index sync] {label}: {status} in {elapsed:.1f}s{self._count_text()}")

    @contextmanager
    def measure(self, operation: str) -> Iterator[None]:
        started = perf_counter()
        with self._lock:
            previous, self._operation = self._operation, operation
        try:
            yield
        finally:
            elapsed = perf_counter() - started
            with self._lock:
                self.seconds[operation] += elapsed
                self._operation = previous

    def advance(self) -> None:
        with self._lock:
            self._done += 1

    def summary(self) -> None:
        with self._lock:
            timings = "; ".join(f"{label} {seconds:.1f}s" for label, seconds in self.seconds.items())
            self.emit(f"[index sync] Timings (apply includes its sub-operations): {timings}")

    def _count_text(self) -> str:
        return "" if self._total is None else f" · {self._done}/{self._total} changes"

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.interval):
            with self._lock:
                if self._active is not None:
                    label, started = self._active
                    detail = f" · {self._operation}" if self._operation else ""
                    self.emit(f"[index sync] {label}: {perf_counter() - started:.1f}s elapsed"
                              f"{self._count_text()}{detail}")
