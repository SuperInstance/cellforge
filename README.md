# cellforge

> **Cellular-substrate ML training.**
> The cell matrix outlives every model architecture.
> Train transformer today; train SSM tomorrow; train diffusion next week — they all read/write the same grid.

A Quilt-native training substrate where the cell matrix is the **permanent destination**, and the math engines (PyTorch / JAX / CUDA / custom) are disposable, hot-swappable utility workers.

**v0.1.2** — The Killer Pause + Write Lock + JEV Oracle

This release ships the minimum that proves the inversion AND a memorable demo:

- 3 dispatcher modes (IDLE / PLAYING / PAUSED)
- 9 cell kinds (8 originals + 1 new FORK_VERSION_VECTOR)
- A dispatcher that **IS** a cell (writes propagate)
- Pause sub-state machine (REQUESTED → ACKNOWLEDGED → FULLY_PAUSED)
- **The killer test**: 100 workers, pause on the same tick, in <2ms, with 0 drift
- **Write-lock safety interlock** (v0.1.1): cannot modify canon while PLAYING
- **Force escape hatch**: `add_cell(force=True)` bypasses lock for admin
- **Witness log writes always allowed** (witness IS canon)
- **JEV oracle** (v0.1.2): use `cellforge.jev.canon_gate()` to gate canon-worthy promotion
- Vector-clock-ordered witness chain
- CLI: `init`, `status`, `test-killer`

20/20 tests pass.

### JEV verdict on cellforge (Sept 23, 2026)

```
canon_score: 0.73/1.00 (Solid canon-worthy, confidence 0.66)
paradigm_shift: 0.74/1.00 (Notable shift, confidence 0.65)
is_inversion: 0.86/1.00 (JEV confirms the inversion)
is_canon (yes/no): 0.28 (JEV says "not yet canon" — keep building)
```

**JEV verdict**: cellforge is canon-worthy, paradigm-shifting, and inverts state/execution. The town-and-laborers architecture is real.

---

## The thesis

> To build a training system that feels as permanent and recognizable as a university town while the underlying infrastructure shifts beneath it, you must invert the relationship between state and execution. The Quilt cell matrix must be the permanent destination, and the low-level mathematical engines must be treated as disposable, hot-swappable utility workers.

In cellforge:

- **State grid** (cells, append-only ledger) is PERMANENT — survives worker turnover
- **Op manifest** (declarative ops between cell blocks) is SEMI-STABLE — versioned
- **Forge runtime** (PyTorch / JAX / CUDA backends) is TRANSIENT — loaded per task, disposed

Train transformer today. Train SSM tomorrow. They all read/write the same grid.

---

## The killer demo

```
$ python3 -m cellforge test-killer --workers 100
Mode: PAUSED
Substate: FULLY_PAUSED
{
  "requested_at_tick": 50,
  "fully_paused_at_tick": 50,
  "workers_acked": 100,
  "workers_total": 100,
  "ack_latency_ms": 1.34,
  "ack_drift_ticks": 0,
  "is_perfect_pause": true
}
```

A 100-worker training run, live at 100Hz, paused on the same tick boundary in **1.34ms** with **0 drift**.

This does not exist anywhere today.

Nobody will remember the JEPA demo. Everyone will remember the first time they perfectly paused a running training run.

---

## Architecture (L1 v0.1.0)

```
+-----------------------------------------------------------+
|                    WORKBOOK (the grid)                    |
|  +-------------+  +------------+  +---------------------+  |
|  | WEIGHT_CELL |  | ACTIVATION |  | GRADIENT_CELL       |  |
|  | (forever)   |  | (rolling)  |  | (forever)           |  |
|  +-------------+  +------------+  +---------------------+  |
|  +-------------+  +------------+  +---------------------+  |
|  | WITNESS_CELL|  | DISPATCH   |  | INFLUENCE_CELL      |  |
|  | (no delete) |  | (first)    |  | (TTL)               |  |
|  +-------------+  +------------+  +---------------------+  |
|  +---------------------+                                   |
|  | FORK_VERSION_VECTOR  |  (NEW in v0.1.0)                 |
|  +---------------------+                                   |
+-----------------------------------------------------------+
                            |
                  writes propagate
                            v
+-----------------------------------------------------------+
|                   DISPATCHER CELL                         |
|  mode: IDLE | PLAYING | PAUSED                            |
|  pause_substate: NOT_PAUSED | REQUESTED | ACKNOWLEDGED    |
|                  | FULLY_PAUSED                           |
|  current_tick: 0..N                                         |
|  pause_stats: { ack_latency_ms, ack_drift_ticks, ... }     |
+-----------------------------------------------------------+
                            |
                  worker.ack_pause()
                            v
+-----------------------------------------------------------+
|                  WORKERS (transient)                       |
|  - MockWorker (v0.1.0) — linear regression simulation     |
|  - PyTorchWorker (v0.2.0)                                  |
|  - JAXWorker (v0.2.0)                                      |
|  - CustomWorker (v0.2.0)                                   |
+-----------------------------------------------------------+
```

---

## What's NOT in v0.1.0 (deferred)

The following were identified by 4 rounds of multi-LLM Adversary but **deferred** to keep v0.1.0 shippable:

- 4 more dispatcher modes (REWINDING, PREDICTING, COMPARING, BACKTESTING)
- 4 more cell kinds (REPLAY_CELL, PREDICTION_CELL, TIMELINE_CELL, EXPERIMENT_LEDGER)
- Master/child dispatcher hierarchy
- Multi-user concurrency (CRDT)
- Master timecode sync (TEMPO_CELL)
- Scheduled triggers (SCHEDULE_CELL)
- Real PyTorch/JAX workers
- Cross-fleet canon contracts

These are documented in `/workspace/research/cellforge-ideation/` and `/workspace/research/cellforge-adversary/`.

---

## Naming doctrine

`cellforge` = `cell` (substrate, matches ax-quilt family) + `forge` (industrial training, transformation, hot). An untrained agent seeing `cellforge` + description reaches for it correctly.

NOT `mavis-cellforge` — production repos are named for zero-shot intuitive understanding, not after Mavis.

---

## Quick start

```python
from cellforge import (
    Cell, CellKind, Dispatcher, Mode, Retention, Workbook, Zone,
)

wb = Workbook(name="my-first-forge")
wb.add_cell(Cell(id="w1", kind=CellKind.WEIGHT, zone=Zone.A,
                 retention=Retention.FULL_LEDGER))

d = Dispatcher()
d.transition_to(Mode.PLAYING)
d.tick(50)
d.pause()
# ... workers call d.worker_ack_pause(...)
print(d.status())
```

## Run the tests

```bash
python3 run_tests.py
```

15/15 tests, including `killer_pause_100_workers`.

## Run the killer demo

```bash
PYTHONPATH=src python3 -m cellforge test-killer --workers 100
```

---

## Files

- `src/cellforge/cells.py` — Cell, WitnessEvent, ForkVersionVector, Workbook
- `src/cellforge/dispatcher.py` — Dispatcher (the playhead), Mode, PauseSubState
- `src/cellforge/worker.py` — MockWorker + WorkerContract
- `src/cellforge/__main__.py` — CLI
- `tests/test_cellforge.py` — 15 tests, including the killer test

## Research (Sept 23, 2026)

- `/workspace/research/cellforge-ideation/` — original design (5 docs, ~50KB)
- `/workspace/research/cellforge-adversary/` — 4 rounds of Adversary, 17 promoted themes

## GitHub

https://github.com/SuperInstance/cellforge

## License

MIT — Casey / SuperInstance, Sept 23, 2026
