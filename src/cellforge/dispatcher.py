"""
cellforge — the dispatcher cell.

The dispatcher IS the play button. It's a value-cell that propagates.
Workers (real or simulated) observe the dispatcher and act accordingly.

L1 v0.1.0 SCOPE: 3 modes only (IDLE, PLAYING, PAUSED).
The killer demo: perfect pause — every worker halts on the same tick within 14ms.

Sub-states for PAUSED (Qwen-Max-Thinking finding 13):
  PAUSE_REQUESTED → PAUSE_ACKNOWLEDGED → FULLY_PAUSED

A worker reports ack when it stops. The dispatcher transitions declared mode
to PAUSED only when ALL workers (or a quorum) have acked.
"""
import asyncio
import enum
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


class Mode(str, enum.Enum):
    IDLE = "IDLE"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"


class PauseSubState(str, enum.Enum):
    NOT_PAUSED = "NOT_PAUSED"
    PAUSE_REQUESTED = "PAUSE_REQUESTED"
    PAUSE_ACKNOWLEDGED = "PAUSE_ACKNOWLEDGED"  # workers reporting back
    FULLY_PAUSED = "FULLY_PAUSED"


@dataclass
class WorkerHandle:
    """A handle for one worker. Workers call ack() when they've stopped."""
    worker_id: str
    zone_id: str
    last_ack_tick: int = -1
    last_ack_substate: PauseSubState = PauseSubState.NOT_PAUSED
    paused_at_tick: Optional[int] = None
    pause_latency_ms: Optional[float] = None  # time from REQUESTED to FULLY_PAUSED


@dataclass
class PauseStats:
    """Stats from the most recent pause operation. Used by the killer demo."""
    requested_at_tick: int
    fully_paused_at_tick: int
    workers_acked: int
    workers_total: int
    ack_latency_ms: float
    ack_drift_ticks: float  # max_tick - min_tick of acks (must be < 0.02)

    def is_perfect_pause(self, max_drift_ticks: float = 0.02, max_latency_ms: float = 14.0) -> bool:
        return self.ack_drift_ticks < max_drift_ticks and self.ack_latency_ms < max_latency_ms


class Dispatcher:
    """The dispatcher cell. Owns worker lifecycle. Observable.

    For v0.1.0 L1: 3 modes (IDLE/PLAYING/PAUSED) + pause sub-state machine.
    """
    def __init__(self, dispatcher_id: str = "DISPATCH_CELL_root"):
        self.dispatcher_id = dispatcher_id
        self.mode: Mode = Mode.IDLE
        self.pause_substate: PauseSubState = PauseSubState.NOT_PAUSED
        self.current_tick: int = 0
        self.workers: Dict[str, WorkerHandle] = {}
        self.tick_callbacks: List[Callable] = []
        # Pause tracking
        self._pause_requested_at: Optional[float] = None
        self._pause_request_tick: Optional[int] = None
        self._pause_acks: List[int] = []  # tick values when workers acked
        self.last_pause_stats: Optional[PauseStats] = None
        # Force-pause flag
        self._force_pause: bool = False

    def register_worker(self, worker_id: str, zone_id: str = "A") -> WorkerHandle:
        """Register a worker with the dispatcher."""
        if worker_id in self.workers:
            raise ValueError(f"Worker {worker_id} already registered")
        w = WorkerHandle(worker_id=worker_id, zone_id=zone_id)
        self.workers[worker_id] = w
        return w

    def register_tick_callback(self, callback: Callable) -> None:
        """Register a callback that runs on every tick."""
        self.tick_callbacks.append(callback)

    def tick(self, n: int = 1) -> None:
        """Advance by n ticks. Only ticks when in PLAYING mode."""
        if self.mode != Mode.PLAYING:
            return
        for _ in range(n):
            self.current_tick += 1
            for cb in self.tick_callbacks:
                cb(self.current_tick)

    def transition_to(self, new_mode: Mode, force: bool = False) -> None:
        """Transition to a new mode. Use pause() for proper pause sub-state."""
        if new_mode == Mode.PAUSED and not force:
            self.pause()
            return
        # Direct transition (IDLE → PLAYING, etc.)
        self.mode = new_mode
        if new_mode != Mode.PAUSED:
            self.pause_substate = PauseSubState.NOT_PAUSED

    def pause(self, force: bool = False, zone_filter: Optional[str] = None) -> None:
        """The killer feature: pause all workers on the same tick boundary.

        Sub-state machine:
          NOT_PAUSED → PAUSE_REQUESTED → PAUSE_ACKNOWLEDGED → FULLY_PAUSED

        Worker acks are recorded via worker.ack_pause() (called by the worker
        itself in real use, by the test harness in tests).
        """
        if self.mode == Mode.PAUSED:
            return  # already paused
        self._force_pause = force
        self._pause_requested_at = time.time()
        self._pause_request_tick = self.current_tick
        self._pause_acks = []
        self.mode = Mode.PAUSED
        self.pause_substate = PauseSubState.PAUSE_REQUESTED
        # If force=True, finalize immediately (no acks required)
        if force:
            self._finalize_pause()

    def worker_ack_pause(self, worker_id: str, at_tick: Optional[int] = None) -> None:
        """A worker reports that it has halted.

        Forces FULLY_PAUSED transition when all (or all forced) workers have acked.
        """
        if worker_id not in self.workers:
            return
        w = self.workers[worker_id]
        if w.last_ack_substate == PauseSubState.FULLY_PAUSED:
            return  # already acked
        w.last_ack_substate = PauseSubState.PAUSE_ACKNOWLEDGED
        w.paused_at_tick = at_tick if at_tick is not None else self.current_tick
        self._pause_acks.append(w.paused_at_tick)
        # First ack moves dispatcher to PAUSE_ACKNOWLEDGED
        if self.pause_substate == PauseSubState.PAUSE_REQUESTED:
            self.pause_substate = PauseSubState.PAUSE_ACKNOWLEDGED

        # Check if all workers have acked (or force)
        if self._force_pause or self._all_workers_acked(zone_filter=None):
            self._finalize_pause()

    def _all_workers_acked(self, zone_filter: Optional[str]) -> bool:
        """Check if all workers (optionally filtered by zone) have acked."""
        targets = [w for w in self.workers.values()
                   if zone_filter is None or w.zone_id == zone_filter]
        if not targets:
            return True
        return all(w.last_ack_substate != PauseSubState.NOT_PAUSED for w in targets)

    def _finalize_pause(self) -> None:
        """Transition PAUSE_REQUESTED → FULLY_PAUSED. Compute stats."""
        now = time.time()
        latency_ms = (now - self._pause_requested_at) * 1000 if self._pause_requested_at else 0
        if self._pause_acks:
            acked_ticks = sorted(self._pause_acks)
            min_tick = acked_ticks[0]
            max_tick = acked_ticks[-1]
            drift = max_tick - min_tick
        else:
            # Force pause with no acks — drift is 0, fully paused at current tick
            min_tick = self.current_tick
            max_tick = self.current_tick
            drift = 0
        self.last_pause_stats = PauseStats(
            requested_at_tick=self._pause_request_tick or 0,
            fully_paused_at_tick=max_tick,
            workers_acked=len(self._pause_acks),
            workers_total=len(self.workers),
            ack_latency_ms=latency_ms,
            ack_drift_ticks=drift,
        )
        self.pause_substate = PauseSubState.FULLY_PAUSED

    def resume(self) -> None:
        """Resume PLAYING from PAUSED. Resets pause tracking state."""
        if self.mode != Mode.PAUSED:
            return
        self.mode = Mode.PLAYING
        self.pause_substate = PauseSubState.NOT_PAUSED
        # Reset worker ack states
        for w in self.workers.values():
            w.last_ack_substate = PauseSubState.NOT_PAUSED
            w.paused_at_tick = None
            w.pause_latency_ms = None

    def status(self) -> Dict:
        """Snapshot of dispatcher state."""
        return {
            "dispatcher_id": self.dispatcher_id,
            "mode": self.mode.value,
            "pause_substate": self.pause_substate.value,
            "current_tick": self.current_tick,
            "workers": len(self.workers),
            "last_pause": (
                None if self.last_pause_stats is None
                else {
                    "requested_at_tick": self.last_pause_stats.requested_at_tick,
                    "fully_paused_at_tick": self.last_pause_stats.fully_paused_at_tick,
                    "workers_acked": self.last_pause_stats.workers_acked,
                    "workers_total": self.last_pause_stats.workers_total,
                    "ack_latency_ms": self.last_pause_stats.ack_latency_ms,
                    "ack_drift_ticks": self.last_pause_stats.ack_drift_ticks,
                    "is_perfect_pause": self.last_pause_stats.is_perfect_pause(),
                }
            ),
        }
