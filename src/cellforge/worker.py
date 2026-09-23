"""
cellforge — workers.

A worker is a transient computational engine that observes the dispatcher cell
state. For v0.1.0, we ship a mock linear-regression worker (no PyTorch).

Workers are HOT-SWAPPABLE in principle — but v0.1.0 ships only one worker type.
v0.2.0+ will add pytorch_worker, jax_worker, custom_worker.

The WorkerContract is what the dispatcher uses to verify workers before
they're allowed to mutate cell state.
"""
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .cells import CellKind, Retention, Zone, WitnessEvent, Workbook
from .dispatcher import Dispatcher, Mode, PauseSubState


@dataclass
class WorkerContract:
    """The contract a worker must satisfy to mutate cell state."""
    name: str
    zone_id: str
    capabilities: List[str] = field(default_factory=list)  # ["read_weights", "write_gradients"]
    dtype_policy: str = "float32"  # all writes must match this dtype
    hot_swap_cost_seconds: float = 0.0


class MockWorker:
    """A simulated worker that pretends to train a linear regression model.

    It doesn't actually do math — it just:
      - observes dispatcher's mode
      - if PLAYING: ticks at random (simulating compute time)
      - if PAUSED: immediately acks and halts
      - records witness events for its operations
    """
    def __init__(self, worker_id: str, dispatcher: Dispatcher, workbook: Workbook,
                 zone_id: str = "A", compute_latency_ms: float = 0.1):
        self.worker_id = worker_id
        self.dispatcher = dispatcher
        self.workbook = workbook
        self.zone_id = zone_id
        self.compute_latency_ms = compute_latency_ms
        self._tick_at_pause: Optional[int] = None
        self._acked = False
        self.contract = WorkerContract(
            name=f"mock_worker_{worker_id}",
            zone_id=zone_id,
            capabilities=["read_weights", "write_gradients"],
        )
        # Register with dispatcher
        self.dispatcher.register_worker(worker_id, zone_id)
        # Register tick callback
        self.dispatcher.register_tick_callback(self._on_tick)

    def _on_tick(self, tick: int) -> None:
        """Called on every tick. Simulate compute, then check for pause."""
        # Simulate compute time
        if self.compute_latency_ms > 0:
            time.sleep(self.compute_latency_ms / 1000.0)
        # Record witness event
        self.workbook.record_witness(
            zone_id=self.zone_id,
            payload={"worker": self.worker_id, "tick": tick, "event": "tick"},
        )
        # Check pause state
        if self.dispatcher.pause_substate in (PauseSubState.PAUSE_REQUESTED,):
            # Immediately ack the pause
            self.ack_pause(at_tick=tick)

    def ack_pause(self, at_tick: Optional[int] = None) -> None:
        """Acknowledge pause request. Stops work."""
        self.dispatcher.worker_ack_pause(self.worker_id, at_tick)
        self._acked = True
        self._tick_at_pause = at_tick
