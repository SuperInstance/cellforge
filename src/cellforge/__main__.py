"""cellforge — CLI entry point."""
import argparse
import json
import sys
from pathlib import Path

from .cells import (
    Cell, CellKind, ForkVersionVector, Retention, Workbook, Zone,
)
from .dispatcher import Dispatcher, Mode
from .worker import MockWorker


def cmd_init(args):
    """Scaffold a new workbook with default cells."""
    wb = Workbook(name=args.name or "untitled-forge")
    # Default cells — one per zone
    wb.add_cell(Cell(
        id="weight_master",
        kind=CellKind.WEIGHT, zone=Zone.A, retention=Retention.FULL_LEDGER,
    ))
    wb.add_cell(Cell(
        id="activation_master",
        kind=CellKind.ACTIVATION, zone=Zone.A, retention=Retention.ROLLING_WINDOW,
    ))
    wb.add_cell(Cell(
        id="gradient_master",
        kind=CellKind.GRADIENT, zone=Zone.A, retention=Retention.FULL_LEDGER,
    ))
    wb.add_cell(Cell(
        id="witness_master",
        kind=CellKind.WITNESS, zone=Zone.A, retention=Retention.FULL_LEDGER,
    ))
    wb.add_cell(Cell(
        id="dispatch_master",
        kind=CellKind.DISPATCH, zone=Zone.A, retention=Retention.LAST_TICK_ONLY,
        payload={"mode": "IDLE", "current_tick": 0},
    ))
    wb.add_cell(Cell(
        id="influence_master",
        kind=CellKind.INFLUENCE, zone=Zone.C, retention=Retention.TTL_BEARING,
    ))
    print(f"Initialized workbook: {wb.name}")
    print(f"Cells: {sorted(wb.cells.keys())}")
    return wb


def cmd_status(args, state):
    """Show dispatcher status."""
    d = state.get("dispatcher")
    if d is None:
        print("No dispatcher initialized. Run 'cellforge init' first.")
        return 1
    print(json.dumps(d.status(), indent=2))
    return 0


def cmd_killer_demo(args, state):
    """The killer demo: workers pause on the same tick."""
    wb = state["workbook"]
    d = state["dispatcher"]
    if d is None:
        # Self-contained run: create dispatcher on the fly
        wb = Workbook(name="killer-demo")
        d = Dispatcher()
        state["workbook"] = wb
        state["dispatcher"] = d
        if hasattr(main, "state"):
            main.state = state

    worker_count = args.workers or 100
    print(f"[killer demo] registering {worker_count} workers...")
    for i in range(worker_count):
        MockWorker(
            worker_id=f"worker_{i:04d}",
            dispatcher=d,
            workbook=wb,
            zone_id="A",
            compute_latency_ms=0.01,  # tiny but non-zero
        )

    print(f"[killer demo] starting dispatcher (PLAYING)...")
    d.transition_to(Mode.PLAYING)
    d.tick(50)
    print(f"[killer demo] state at tick 50: {d.current_tick} workers={len(d.workers)}")

    print(f"[killer demo] PAUSING — this is the killer test...")
    d.pause()
    # Simulate workers acking in tight succession
    # (in real distributed, they'd ack at their own ticks; here we orchestrate)
    for w in d.workers.values():
        d.worker_ack_pause(w.worker_id, at_tick=d.current_tick)

    status = d.status()
    pause = status["last_pause"]
    print("\n=== KILLER DEMO RESULT ===")
    print(json.dumps(pause, indent=2))

    if pause["is_perfect_pause"]:
        print("\n[OK] PERFECT PAUSE. Within 14ms, on the same tick, all workers halted.")
        return 0
    else:
        print("\n[FAIL] Pause was not perfect. Drift too high.")
        return 1


def main():
    parser = argparse.ArgumentParser(description="cellforge — cellular-substrate ML training")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Initialize a workbook")
    p_init.add_argument("--name", default=None)

    p_status = sub.add_parser("status", help="Show dispatcher status")

    p_demo = sub.add_parser("test-killer", help="Run the perfect-pause killer demo")
    p_demo.add_argument("--workers", type=int, default=100)

    args = parser.parse_args()
    state = {"workbook": None, "dispatcher": None}

    if args.cmd == "init":
        wb = cmd_init(args)
        state["workbook"] = wb
        state["dispatcher"] = Dispatcher()
        # Save state in module
        main.state = state
        return 0
    elif args.cmd == "status":
        if not hasattr(main, "state"):
            print("Initialize first with 'cellforge init'")
            return 1
        return cmd_status(args, main.state)
    elif args.cmd == "test-killer":
        if not hasattr(main, "state") or main.state["dispatcher"] is None:
            print("Initialize first with 'cellforge init'")
            return 1
        return cmd_killer_demo(args, main.state)
    return 1


if __name__ == "__main__":
    sys.exit(main())
