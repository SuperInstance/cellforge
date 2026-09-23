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
class Workbook:
    """The grid. The town. The cell matrix."""
    name: str
    cells: Dict[str, Cell] = field(default_factory=dict)
    forks: Dict[str, ForkVersionVector] = field(default_factory=dict)
    # Witness chain (only WITNESS_CELL events live here; FULL_LEDGER cells
    # keep their own internal ledger)
    witness_log: List[WitnessEvent] = field(default_factory=list)
    # Vector clock per zone
    vector_clock: Dict[str, int] = field(default_factory=lambda: {"A": 0, "B": 0, "C": 0, "master": 0})

    def add_cell(self, cell: Cell) -> None:
        if cell.id in self.cells:
            raise ValueError(f"Cell {cell.id} already exists")
        self.cells[cell.id] = cell

    def get_cell(self, cell_id: str) -> Cell:
        return self.cells[cell_id]

    def add_fork(self, fv: ForkVersionVector) -> None:
        self.forks[fv.fork_id] = fv
        # Update siblings' lists
        if fv.parent_fork:
            parent = self.forks.get(fv.parent_fork)
            if parent and fv.fork_id not in parent.sibling_forks:
                parent.sibling_forks.append(fv.fork_id)

    def record_witness(self, zone_id: str, payload: Any) -> WitnessEvent:
        """Append a witness event with vector-clock consistency."""
        self.vector_clock[zone_id] = self.vector_clock.get(zone_id, 0) + 1
        parent_hashes = [e.content_hash for e in self.witness_log[-3:]]  # last 3 for resilience
        ev = WitnessEvent.make(
            tick=self.vector_clock["master"],  # global tick = max of all zones
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
        # WITNESS_CELL retention check
        for c in self.cells.values():
            if c.kind == CellKind.WITNESS and c.retention != Retention.FULL_LEDGER:
                issues.append(f"WITNESS_CELL {c.id} must have full_ledger retention")
        # Vector clock consistency
        for ev in self.witness_log:
            for z in ev.vector_clock:
                if ev.vector_clock[z] > self.vector_clock.get(z, 0) + 1:
                    issues.append(f"Witness at tick {ev.tick} has inconsistent vector clock")
        return issues
