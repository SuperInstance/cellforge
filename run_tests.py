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
        # v0.1.1 — write lock safety interlock
        ("cannot_modify_canon_while_playing", t.test_cannot_modify_canon_while_playing),
        ("force_write_bypasses_lock", t.test_force_write_bypasses_lock),
        ("witness_log_writes_always_allowed", t.test_witness_log_writes_always_allowed),
        # JEV oracle helper
        ("jev_helper_smoke", t.test_jev_helper_smoke),
        ("canon_gate_shape", t.test_canon_gate_shape),
        # v0.2.0 — REWINDING mode + REPLAY_CELL
        ("rewind_to_tick_n", t.test_rewind_to_tick_n),
        ("rewind_blocked_when_no_ticks_happened", t.test_rewind_blocked_when_no_ticks_happened),
        ("rewind_blocks_writes_to_canon", t.test_rewind_blocks_writes_to_canon),
        ("replay_cell_integrity", t.test_replay_cell_integrity),
        ("pause_rewind_change_resume_diverges", t.test_pause_rewind_change_resume_diverges),
        # v0.3.0 — PREDICTING + PredictionCell + JEPA stub
        ("predicting_creates_fork_id", t.test_predicting_creates_fork_id),
        ("predicting_exits_to_paused", t.test_predicting_exits_to_paused),
        ("prediction_cell_normalized", t.test_prediction_cell_normalized),
        ("prediction_cell_entropy", t.test_prediction_cell_entropy),
        ("jepa_predictor_basic", t.test_jepa_predictor_basic),
        ("jev_verifier_basic", t.test_jev_verifier_basic),
        # v0.4.0 — EXPERIMENTAL mode (PREDICTING + BACKTESTING)
        ("experimental_pure_predicting", t.test_experimental_pure_predicting),
        ("experimental_with_replay_is_backtesting", t.test_experimental_with_replay_is_backtesting),
        ("experimental_exits_to_paused", t.test_experimental_exits_to_paused),
        ("compare_scenarios_returns_dict", t.test_compare_scenarios_returns_dict),
        ("compare_scenarios_only_in_experimental", t.test_compare_scenarios_only_in_experimental),
        ("experimental_multi_scenario_promises", t.test_experimental_multi_scenario_promises),
        # v0.4.1 — Causal-consistency verdict on rewind (R10 Theme T5)
        ("rewind_returns_verdict_dict", t.test_rewind_returns_verdict_dict),
        ("rewind_without_bound_workbook_assumes_causal", t.test_rewind_without_bound_workbook_assumes_causal),
        ("rewind_with_witness_chain_detects_non_causal", t.test_rewind_with_witness_chain_detects_non_causal),
        ("rewind_with_properly_chained_witness_is_causal", t.test_rewind_with_properly_chained_witness_is_causal),
        ("bind_workbook_symmetric", t.test_bind_workbook_symmetric),
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
