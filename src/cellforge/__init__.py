"""
cellforge — minimal public API.

The whole L1 v0.1.0 surface area, in one easy import.

Example:
    from cellforge import Workbook, Dispatcher, MockWorker, Cell, CellKind, Zone, Retention

    wb = Workbook(name="my-first-forge")
    wb.add_cell(Cell(id="w1", kind=CellKind.WEIGHT, zone=Zone.A, retention=Retention.FULL_LEDGER))

    d = Dispatcher()
    for i in range(100):
        MockWorker(worker_id=f"w{i:03d}", dispatcher=d, workbook=wb)

    d.transition_to(Mode.PLAYING)
    d.tick(50)
    d.pause()
    # ... workers ack
    d.status()  # see pause stats
"""
from .cells import (
    Cell, CellKind, ForkVersionVector, Retention, WitnessEvent, Workbook, Zone,
)
from .dispatcher import Dispatcher, Mode, PauseStats, PauseSubState, WorkerHandle
from .worker import MockWorker, WorkerContract

__all__ = [
    "Cell", "CellKind", "ForkVersionVector", "Retention", "WitnessEvent", "Workbook", "Zone",
    "Dispatcher", "Mode", "PauseStats", "PauseSubState", "WorkerHandle",
    "MockWorker", "WorkerContract",
]

__version__ = "0.1.0"
