"""
cellforge — the substrate cell kind definitions.

The cell matrix is the permanent substrate. Every cell has:
  - id (unique within workbook)
  - kind (one of the 8 originals + FORK_VERSION_VECTOR)
  - retention policy (full_ledger / rolling_window / last_tick_only / ttl_bearing)
  - zone (A / B / C / cross-zone)
  - payload (the actual value(s))

THIS MODULE: pure data + dataclass definitions. No execution logic.
"""
import dataclasses
import datetime
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CellKind(str, Enum):
    WEIGHT = "WEIGHT_CELL"                # permanent model parameters
    ACTIVATION = "ACTIVATION_CELL"        # forward-pass intermediates (rolling)
    GRADIENT = "GRADIENT_CELL"            # gradients (full_ledger)
    TOKEN = "TOKEN_CELL"                  # current token/batch (last_tick_only)
    INFLUENCE = "INFLUENCE_CELL"          # human/agent guidance (ttl_bearing)
    WITNESS = "WITNESS_CELL"              # hash chain (NO DELETION)
    DISPATCH = "DISPATCH_CELL"            # owns worker lifecycle (first-class)
    ZONE_BOUNDARY = "ZONE_BOUNDARY_CELL"  # declares inter-zone permissions
    FORK_VERSION_VECTOR = "FORK_VERSION_VECTOR"  # new in v0.1.0


class Zone(str, Enum):
    A = "A"  # Fast Reflex (100Hz)
    B = "B"  # Slow Critic (1Hz, reads A only)
    C = "C"  # Nudge Foundry (human/agent-paced, TTL writes to A)


class Retention(str, Enum):
    FULL_LEDGER = "full_ledger"        # every value stored, append-only
    ROLLING_WINDOW = "rolling_window"  # keep last N values
    LAST_TICK_ONLY = "last_tick_only"  # only current value
    TTL_BEARING = "ttl_bearing"        # value expires at tick


@dataclass
class Cell:
    """Base cell. All cells have these fields."""
    id: str
    kind: CellKind
    zone: Zone
    retention: Retention
    payload: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at_tick: int = 0
    last_modified_tick: int = 0

    def __post_init__(self):
        if not self.id:
            raise ValueError("Cell.id required")
        # Validate kind/zone/retention compatibility
        if self.kind == CellKind.WITNESS and self.retention != Retention.FULL_LEDGER:
            raise ValueError("WITNESS_CELL must have full_ledger retention (no deletion)")


@dataclass
class WitnessEvent:
    """A single witness chain entry. Vector-clock consistent."""
    tick: int
    zone_id: str
    vector_clock: Dict[str, int]
    parent_hashes: List[str]
    content_hash: str
    payload: Any
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")

    @staticmethod
    def make(tick: int, zone_id: str, vector_clock: Dict[str, int],
             parent_hashes: List[str], payload: Any) -> "WitnessEvent":
        # Compute content hash
        h = hashlib.sha256()
        h.update(f"{tick}|{zone_id}|{vector_clock}|{parent_hashes}|{payload}".encode())
        return WitnessEvent(
            tick=tick,
            zone_id=zone_id,
            vector_clock=dict(vector_clock),
            parent_hashes=list(parent_hashes),
            content_hash=h.hexdigest()[:16],
            payload=payload,
        )


@dataclass
class ForkVersionVector:
    """For v0.1.0: minimal fork DAG metadata. Future versions extend."""
    fork_id: str
    parent_fork: Optional[str] = None
    sibling_forks: List[str] = field(default_factory=list)
    zone_id: str = "A"
    created_at_tick: int = 0


@dataclass
class ReplayCell:
    """v0.2.0: A snapshot of the witness chain to be replayed.

    Holds a sequence of WitnessEvents captured at some point in the past.
    The REPLAY_CELL can be loaded into a Dispatcher's rewind history,
    letting the user traverse what happened.
    """
    id: str
    zone_id: str = "A"
    events: List[WitnessEvent] = field(default_factory=list)
    captured_at_tick: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def integrity_check(self) -> bool:
        """Verify the witness chain within this REPLAY_CELL is intact.

        Each event's parent_hashes should reference earlier events in the list.
        """
        for i, ev in enumerate(self.events):
            for ph in ev.parent_hashes:
                # Find this parent in earlier events
                found = any(e.content_hash == ph for e in self.events[:i + 1])
                if not found:
                    return False
        return True

    def size(self) -> int:
        return len(self.events)

    def event_at_tick(self, tick: int) -> Optional[WitnessEvent]:
        for ev in self.events:
            if ev.tick == tick:
                return ev
        return None


@dataclass
class PredictionCell:
    """v0.3.0: A predicted future cell state with uncertainty.

    Carries a probability distribution over possible values, NOT a point
    estimate. Per Qwen-Max-Thinking Theme 22: predictions are epistemic,
    not ontic. The dispatcher treats them as DISTRIBUTIONS, not facts.

    fields:
      - prediction_id: unique id
      - source_cell: which canon cell is being predicted
      - distribution: {value: probability} mapping (sums to ~1.0)
      - mean: expected value (computed from distribution)
      - std_dev: uncertainty magnitude
      - horizon: how many ticks ahead this prediction is for
      - confidence: derived from multi-worker polyformality (Jaccard)
      - worker_agreement: set of (worker_id, value) tuples that voted
      - tick: tick at which prediction was made
    """
    prediction_id: str
    source_cell: str
    distribution: Dict[str, float] = field(default_factory=dict)
    horizon: int = 1
    confidence: float = 0.0
    worker_agreement: List[tuple] = field(default_factory=list)
    tick: int = 0
    zone_id: str = "A"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "PredictionCell":
        """Return a copy with distribution normalized to sum=1.0."""
        total = sum(self.distribution.values())
        if total == 0:
            return self
        new_dist = {k: v / total for k, v in self.distribution.items()}
        return PredictionCell(
            prediction_id=self.prediction_id,
            source_cell=self.source_cell,
            distribution=new_dist,
            horizon=self.horizon,
            confidence=self.confidence,
            worker_agreement=list(self.worker_agreement),
            tick=self.tick,
            zone_id=self.zone_id,
            metadata=dict(self.metadata),
        )

    def entropy(self) -> float:
        """Shannon entropy of the distribution (higher = more uncertain)."""
        import math
        h = 0.0
        for p in self.distribution.values():
            if p > 0:
                h -= p * math.log2(p)
        return h

    def to_canonical(self) -> Optional[str]:
        """Return the highest-probability value if confidence > threshold."""
        if not self.distribution or self.confidence < 0.7:
            return None
        return max(self.distribution.items(), key=lambda kv: kv[1])[0]


@dataclass
class Workbook:
    """The grid. The town. The cell matrix.

    v0.1.1: write-lock when dispatcher is PLAYING.
    v0.2.0: + write-lock in REWINDING mode. + REPLAY_CELL support.
    Modifying the canon (cells, forks) requires PAUSE.
    Witness log writes are ALWAYS allowed (they ARE the canon).
    """
    name: str
    cells: Dict[str, Cell] = field(default_factory=dict)
    forks: Dict[str, ForkVersionVector] = field(default_factory=dict)
    replays: Dict[str, ReplayCell] = field(default_factory=dict)
    # Witness chain (only WITNESS_CELL events live here; FULL_LEDGER cells
    # keep their own internal ledger)
    witness_log: List[WitnessEvent] = field(default_factory=list)
    # Vector clock per zone
    vector_clock: Dict[str, int] = field(default_factory=lambda: {"A": 0, "B": 0, "C": 0, "master": 0})
    # Optional reference to a Dispatcher for write-lock enforcement
    _dispatcher: Optional[Any] = field(default=None, init=False, repr=False)

    def bind_dispatcher(self, dispatcher) -> None:
        """Bind a dispatcher so add_cell enforces write-lock when PLAYING."""
        self._dispatcher = dispatcher

    def _check_write_lock(self, force: bool) -> None:
        """Internal: enforce write-lock unless force=True or not bound to dispatcher."""
        if force or self._dispatcher is None:
            return
        mode = getattr(self._dispatcher, "mode", None)
        from .dispatcher import Mode
        if mode == Mode.PLAYING:
            raise PermissionError(
                "Cannot modify canon while dispatcher is PLAYING. "
                "Pause first, or pass force=True to bypass."
            )
        if mode == Mode.REWINDING:
            raise PermissionError(
                "Cannot modify canon while in REWINDING mode. "
                "Resume or fork first."
            )

    def add_cell(self, cell: Cell, force: bool = False) -> None:
        """Add a cell. Write-lock enforces (force=True bypass)."""
        self._check_write_lock(force)
        if cell.id in self.cells:
            raise ValueError(f"Cell {cell.id} already exists")
        self.cells[cell.id] = cell

    def get_cell(self, cell_id: str) -> Cell:
        return self.cells[cell_id]

    def add_fork(self, fv: ForkVersionVector, force: bool = False) -> None:
        """Add a fork. Write-lock enforces."""
        self._check_write_lock(force)
        self.forks[fv.fork_id] = fv
        if fv.parent_fork:
            parent = self.forks.get(fv.parent_fork)
            if parent and fv.fork_id not in parent.sibling_forks:
                parent.sibling_forks.append(fv.fork_id)

    def add_replay(self, replay: ReplayCell, force: bool = False) -> None:
        """v0.2.0: Add a REPLAY_CELL holding historical witness chain."""
        self._check_write_lock(force)
        self.replays[replay.id] = replay

    def capture_replay(self, replay_id: str, zone_id: str = "A",
                      force: bool = False) -> ReplayCell:
        """v0.2.0: Capture current witness_log into a REPLAY_CELL."""
        # REPLAY_CELL creation requires PLAYING (witness is being recorded)
        # But the REPLAY_CELL write is in itself a write — needs force or PAUSE
        if self._dispatcher is not None and not force:
            from .dispatcher import Mode
            if self._dispatcher.mode == Mode.PAUSED:
                pass  # OK to capture while paused
            elif self._dispatcher.mode == Mode.PLAYING:
                raise PermissionError(
                    "Cannot capture replay while PLAYING. Pause first, or use force=True."
                )
        replay = ReplayCell(
            id=replay_id,
            zone_id=zone_id,
            events=list(self.witness_log),  # snapshot
            captured_at_tick=self._dispatcher.current_tick if self._dispatcher else 0,
        )
        self.replays[replay_id] = replay
        return replay

    def record_witness(self, zone_id: str, payload: Any) -> WitnessEvent:
        """Append a witness event. ALWAYS allowed (witness IS the canon)."""
        self.vector_clock[zone_id] = self.vector_clock.get(zone_id, 0) + 1
        parent_hashes = [e.content_hash for e in self.witness_log[-3:]]
        ev = WitnessEvent.make(
            tick=self.vector_clock["master"],
            zone_id=zone_id,
            vector_clock=dict(self.vector_clock),
            parent_hashes=parent_hashes,
            payload=payload,
        )
        self.witness_log.append(ev)
        return ev

    def validate(self) -> List[str]:
        """Validate the workbook. Returns list of issues (empty if valid)."""
        issues = []
        for c in self.cells.values():
            if c.kind == CellKind.WITNESS and c.retention != Retention.FULL_LEDGER:
                issues.append(f"WITNESS_CELL {c.id} must have full_ledger retention")
        for ev in self.witness_log:
            for z in ev.vector_clock:
                if ev.vector_clock[z] > self.vector_clock.get(z, 0) + 1:
                    issues.append(f"Witness at tick {ev.tick} has inconsistent vector clock")
        return issues
