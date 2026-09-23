"""Test runner for cellforge v0.1.0."""
import sys
import traceback
from pathlib import Path

# Add src to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

# Import tests as a module
sys.path.insert(0, str(ROOT / "tests"))

import test_cellforge as t


def main():
    tests = [
        ("tick_advances_state_when_playing", t.test_tick_advances_state_when_playing),
        ("tick_does_not_advance_when_paused", t.test_tick_does_not_advance_when_paused),
        ("register_worker", t.test_register_worker),
        ("pause_requires_worker_ack", t.test_pause_requires_worker_ack),
        ("pause_is_perfect_with_quick_acks", t.test_pause_is_perfect_with_quick_acks),
        ("force_pause", t.test_force_pause),
        ("resume_continues_from_current_tick", t.test_resume_continues_from_current_tick),
        ("resume_resets_worker_ack_state", t.test_resume_resets_worker_ack_state),
        ("pause_when_already_paused_is_noop", t.test_pause_when_already_paused_is_noop),
        ("zone_isolation", t.test_zone_isolation),
        ("fork_version_vector_basic", t.test_fork_version_vector_basic),
        ("witness_chain_integrity", t.test_witness_chain_integrity),
        ("witness_cell_retention_constraint", t.test_witness_cell_retention_constraint),
        ("killer_pause_100_workers", t.test_killer_pause_100_workers),
        ("status_returns_useful_dict", t.test_status_returns_useful_dict),
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed}/{len(tests)} tests passed ({failed} failed)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
