#!/usr/bin/env python3
"""Verifier shim for the CloudCast multi-cloud broadcast task.

Bridges the CORAL `evaluator.py` (in-process Python, returns
``{"combined_score": float, "total_cost": float, ...}``) into the cloudcast
verifier protocol that hayekmas's CloudcastEnv expects:

    python3 compute_reward.py --app-dir <workspace> --output-dir <out>

writes ``<out>/reward.json`` with ``score``, ``reason``, and per-config
``subscores``.  ``combined_score`` is already in (0, 1] so we use it
directly as the normalized score; ``total_cost`` is reported alongside.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE / "cloudcast_eval"

# Seed-program baseline: running the unmodified initial_program.py against all
# five configs gives total_cost=1035.0 (combined_score≈0.000965).  We normalize
# the engine-facing score to the cost reduction relative to this baseline:
#
#     score = max(0, 1 - total_cost / SEED_BASELINE_COST)
#
# so the seed scores 0, halving the cost scores 0.5, and approaching free
# delivery scores 1.  This keeps the env_reward_scale / terminal_output_bonus
# defaults (5.0 / 2.0) well-behaved without per-task retuning.  The raw
# combined_score is preserved in the reward.json `raw_evaluator` block for
# downstream analysis.
SEED_BASELINE_COST = 1035.0


def _write_reward(out_dir: Path, score: float, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    body = {"score": float(score), "reward": float(score), **payload}
    (out_dir / "reward.json").write_text(json.dumps(body, indent=2))
    (out_dir / "reward.txt").write_text(f"{score}\n")


def _load_evaluator():
    """Load the bundled CORAL evaluator module."""
    if str(EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(EVAL_DIR))
    spec = importlib.util.spec_from_file_location(
        "cloudcast_evaluator", str(EVAL_DIR / "evaluator.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-dir", required=True, help="Agent workspace.")
    parser.add_argument("--output-dir", required=True, help="Where to write reward.json.")
    args = parser.parse_args()

    app_dir = Path(args.app_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    program = app_dir / "initial_program.py"
    if not program.is_file():
        _write_reward(out_dir, 0.0, {
            "reason": f"initial_program.py missing in workspace ({app_dir})",
            "subscores": [],
            "total_cost": None,
        })
        return 0

    # Put the workspace on sys.path so the evaluator can resolve any helper
    # modules the agent dropped alongside initial_program.py.
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))

    try:
        evaluator = _load_evaluator()
    except Exception as exc:  # noqa: BLE001
        _write_reward(out_dir, 0.0, {
            "reason": f"failed to load evaluator: {exc}",
            "trace": traceback.format_exc()[-2000:],
            "subscores": [],
            "total_cost": None,
        })
        return 0

    # The evaluator writes paths/ and evals/ relative to CWD; sandbox it.
    with tempfile.TemporaryDirectory(prefix="cloudcast_eval_") as tmpdir:
        cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            try:
                result = evaluator.evaluate(str(program))
            except Exception as exc:  # noqa: BLE001
                _write_reward(out_dir, 0.0, {
                    "reason": f"evaluate() crashed: {exc}",
                    "trace": traceback.format_exc()[-2000:],
                    "subscores": [],
                    "total_cost": None,
                })
                return 0
        finally:
            os.chdir(cwd)

    if not isinstance(result, dict):
        _write_reward(out_dir, 0.0, {
            "reason": f"evaluator returned non-dict: {type(result).__name__}",
            "subscores": [],
            "total_cost": None,
        })
        return 0

    err = result.get("error")
    combined = float(result.get("combined_score", 0.0) or 0.0)
    total_cost = result.get("total_cost")
    successful = int(result.get("successful_configs", 0) or 0)
    failed = int(result.get("failed_configs", 0) or 0)

    if err or failed > 0 or total_cost is None:
        # File parses-but-crashes / partial config breakage. We need a *negative*
        # score (not 0) so the engine's delta mechanism (env._record_checkpoint_score)
        # treats "I broke it" as a regression vs. a previously-working program.
        # Floor of 0 here used to silently absorb breakage as "no signal".
        score = -0.1
        improvement_pct = 0.0
    else:
        # No floor: cost above seed yields a *negative* score so the
        # checkpoint delta remains informative even when the agent is still
        # below baseline. Upper bound stays at 1.0 (cost=0).
        score = min(1.0, 1.0 - float(total_cost) / SEED_BASELINE_COST)
        improvement_pct = 100.0 * (1.0 - float(total_cost) / SEED_BASELINE_COST)

    reason = (
        f"score={score:.4f} (cost {total_cost} vs seed {SEED_BASELINE_COST}, "
        f"{improvement_pct:+.2f}%) | combined_score={combined:.6f} | "
        f"configs={successful}/{successful + failed}"
        + (f" | error={err}" if err else "")
    )

    _write_reward(out_dir, score, {
        "reason": reason,
        "total_cost": total_cost,
        "avg_cost": result.get("avg_cost"),
        "successful_configs": successful,
        "failed_configs": failed,
        "runs_successfully": result.get("runs_successfully"),
        "improvement_pct": improvement_pct,
        "seed_baseline_cost": SEED_BASELINE_COST,
        "subscores": [
            {"subtask": "cost_reduction_vs_seed", "score": round(score, 6),
             "stdout": reason, "stderr": ""},
            {"subtask": "raw_combined_score", "score": round(combined, 6),
             "stdout": f"combined_score={combined:.6f} (1/(1+total_cost))",
             "stderr": ""},
        ],
        "raw_evaluator": {k: v for k, v in result.items() if k not in {"error"}},
    })
    print(
        f"score={score:.4f}  improvement={improvement_pct:+.2f}%  "
        f"total_cost={total_cost}  configs={successful}/{successful + failed}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
