#!/usr/bin/env python3
"""
Standalone Timeloop/DOSA-Gemmini eval script for the agent workspace.

Reads `candidate.yaml` (a DOSA mapping string + rationale), parses it with DOSA's
`mapping_utils.process_mapping`, runs Timeloop via `GemminiConfig.run_mapping_from_dict`,
returns JSON to stdout, appends to history.jsonl, updates best.json if EDP improves,
decrements budget.json.

Usage:
    python eval.py                 # read candidate.yaml, evaluate
    python eval.py --check-budget  # print remaining budget
    python eval.py --reset         # reset budget/history/best
    python eval.py --reset --max-calls 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import yaml

WS = Path(__file__).resolve().parent
DOSA_ROOT_RAW = os.environ.get("DOSA_ROOT", "")
DOSA_ROOT = Path(DOSA_ROOT_RAW) if DOSA_ROOT_RAW else None
if DOSA_ROOT is not None and str(DOSA_ROOT) not in sys.path:
    sys.path.insert(0, str(DOSA_ROOT))

WORKLOAD_FILE = WS / "workload.yaml"
HARDWARE_FILE = WS / "hardware.yaml"
CANDIDATE_FILE = WS / "candidate.yaml"
HISTORY_FILE = WS / "history.jsonl"
BEST_FILE = WS / "best.json"
BUDGET_FILE = WS / "budget.json"


def load_budget() -> dict:
    if not BUDGET_FILE.exists():
        return {"max_calls": 200, "calls_used": 0}
    with open(BUDGET_FILE) as f:
        return json.load(f)


def save_budget(b: dict) -> None:
    with open(BUDGET_FILE, "w") as f:
        json.dump(b, f, indent=2)


def append_history(entry: dict) -> None:
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def update_best(result: dict) -> bool:
    if not result.get("valid"):
        return False
    if not BEST_FILE.exists():
        with open(BEST_FILE, "w") as f:
            json.dump(result, f, indent=2)
        return True
    with open(BEST_FILE) as f:
        cur = json.load(f)
    if result["edp"] < cur.get("edp", float("inf")):
        with open(BEST_FILE, "w") as f:
            json.dump(result, f, indent=2)
        return True
    return False


def validate_mapping_str(mapping_str: str) -> tuple[bool, str]:
    """Reject mappings with duplicated (dim, kind) at same level.

    Motivation: DOSA's `process_mapping` takes the LAST occurrence of a dim/kind at
    a level, silently overwriting earlier tokens. This lets a mapping like
    `L3[WIO] P1024 K16 P1` effectively set P_L3=1 while appearing to cover P=1024,
    shortening the computation. We reject such mappings here.
    """
    import re
    blocks = mapping_str.split(" - ")
    for blk in blocks:
        parts = blk.split()
        if not parts:
            return False, f"empty level block"
        lvl_tag = parts[0]
        seen = set()
        for tok in parts[1:]:
            m = re.match(r"([A-Z])([0-9]+)([XY]?)", tok)
            if not m:
                return False, f"malformed token '{tok}' in {lvl_tag}"
            dim = m.group(1)
            spatial = m.group(3) or "T"  # T for temporal, else X/Y
            key = (dim, spatial)
            if key in seen:
                return False, (
                    f"duplicate {dim}{'(spatial)' if spatial in ('X','Y') else '(temporal)'}"
                    f" at {lvl_tag} — tokens appear more than once"
                )
            seen.add(key)
    return True, "ok"


def run_one(mapping_str: str, rationale: str = "") -> dict:
    """Run a single mapping through Timeloop via DOSA's pipeline. Returns result dict."""
    if DOSA_ROOT is None:
        return {
            "valid": False,
            "reason": "DOSA_ROOT environment variable is required",
            "mapping_str": mapping_str,
        }

    from dataset.common import mapping_utils
    from dataset.workloads import Prob
    from dataset.hw import init_hw_config

    # Validate mapping string structure
    ok, msg = validate_mapping_str(mapping_str)
    if not ok:
        return {
            "valid": False,
            "reason": f"malformed mapping: {msg}",
            "mapping_str": mapping_str,
        }

    # Load workload
    with open(WORKLOAD_FILE) as f:
        wl_doc = yaml.safe_load(f)
    # Prob can take either a path or a dict
    prob = Prob(wl_doc)

    # Load hardware search spec (ranges for pe_dim, sp_size, acc_size)
    with open(HARDWARE_FILE) as f:
        hw_doc = yaml.safe_load(f)
    pe_range = hw_doc.get("pe_dim_range", [hw_doc.get("pe_dim_fixed", 16),
                                            hw_doc.get("pe_dim_fixed", 16)])
    sp_range = hw_doc.get("sp_size_range", [1, 2048])
    acc_range = hw_doc.get("acc_size_range", [1, 2048])

    # Read agent's chosen HW from candidate
    with open(CANDIDATE_FILE) as f:
        cand_doc = yaml.safe_load(f)
    hw_cfg_from_cand = cand_doc.get("hw") or {}
    pe_dim = int(hw_cfg_from_cand.get("pe_dim", hw_doc.get("default_pe_dim", 16)))
    sp_size = int(hw_cfg_from_cand.get("sp_size", hw_doc.get("default_sp_size", 128)))
    acc_size = int(hw_cfg_from_cand.get("acc_size", hw_doc.get("default_acc_size", 32)))

    # Clamp to spec
    pe_dim = max(int(pe_range[0]), min(int(pe_range[1]), pe_dim))
    sp_size = max(int(sp_range[0]), min(int(sp_range[1]), sp_size))
    acc_size = max(int(acc_range[0]), min(int(acc_range[1]), acc_size))
    hw_config = [pe_dim, sp_size, acc_size]

    # Scratch dir for arch/logs — use project scratch
    scratch_raw = os.environ.get("ARCHGYM_SCRATCH", "")
    if not scratch_raw:
        return {
            "valid": False,
            "reason": "ARCHGYM_SCRATCH environment variable is required",
            "mapping_str": mapping_str,
        }
    scratch_root = Path(scratch_raw)
    scratch_root.mkdir(parents=True, exist_ok=True)
    output_dir = scratch_root / f"eval_{os.getpid()}_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        flat = mapping_utils.process_mapping(mapping_str, prob.shape)
    except Exception as e:
        return {
            "valid": False,
            "reason": f"process_mapping failed: {e}",
            "mapping_str": mapping_str,
        }

    try:
        arch_config = init_hw_config("gemmini", hw_config, output_dir)
        mapping_dict = arch_config.flat_mapping_to_dict(prob.shape, flat)
        row = arch_config.run_mapping_from_dict(prob, mapping_dict)
    except Exception as e:
        return {
            "valid": False,
            "reason": f"Timeloop eval failed: {e}",
            "traceback": traceback.format_exc()[:1000],
            "mapping_str": mapping_str,
            "flat_mapping_len": int(len(flat)),
        }

    elapsed = time.time() - t0

    if not row:
        # Parse Timeloop's diagnostic log for the actual rejection reason
        reason = "Timeloop returned no row (mapper rejected mapping)"
        fail_diagnostic = None
        try:
            # Find the most recently-created timeloop-*/random.txt in scratch
            tl_logs = sorted(
                Path(output_dir).rglob("random.txt"),
                key=lambda p: p.stat().st_mtime,
            )
            if tl_logs:
                log_text = tl_logs[-1].read_text()
                # Extract "Level: X ... Fail reason: Y" blocks
                blocks = []
                lines = log_text.splitlines()
                for i, line in enumerate(lines):
                    if "Fail reason:" in line:
                        # Walk back a few lines to find the Level header
                        lvl = None
                        for j in range(max(0, i - 30), i):
                            if lines[j].strip().startswith("Level:"):
                                lvl = lines[j].strip()
                                break
                        blocks.append(f"{lvl or 'Level: ?'}  |  {line.strip()}")
                if blocks:
                    fail_diagnostic = " ; ".join(blocks[:3])
                elif "no valid mappings found" in log_text:
                    fail_diagnostic = "no valid mappings found within search criteria (check factorization, spatial constraints, or buffer capacity)"
        except Exception:
            pass
        return {
            "valid": False,
            "reason": reason,
            "timeloop_diagnostic": fail_diagnostic,
            "mapping_str": mapping_str,
            "eval_duration_s": elapsed,
        }

    try:
        cycles = float(row["target.cycle"])
        energy = float(row["target.energy"])
        area = float(row.get("target.area", 0.0))
    except Exception as e:
        return {
            "valid": False,
            "reason": f"Missing target fields: {e}",
            "row_keys": list(row.keys())[:20],
            "mapping_str": mapping_str,
        }

    if cycles <= 0 or energy <= 0:
        return {
            "valid": False,
            "reason": f"invalid cycles/energy: cycles={cycles}, energy={energy}",
            "mapping_str": mapping_str,
        }

    # Sanity: cycles must be at least (K*C*P*Q*R*S*N) / (pe_dim^2), else the mapping
    # is skipping work. Allow 10% slack for Timeloop's counting conventions.
    dims = prob.prob
    total_macs = 1
    for k in ("K", "C", "P", "Q", "R", "S", "N"):
        total_macs *= dims.get(k, 1)
    min_cycles = total_macs / (pe_dim * pe_dim)
    if cycles < 0.9 * min_cycles:
        return {
            "valid": False,
            "reason": (
                f"cycles ({cycles:.0f}) implausibly low: "
                f"total_MACs={total_macs}, pe_dim={pe_dim}, min_expected={min_cycles:.0f}. "
                f"Your factorization must cover every workload dimension exactly once "
                f"across the 4 levels."
            ),
            "mapping_str": mapping_str,
            "cycles_reported": cycles,
            "cycles_min_expected": min_cycles,
        }

    # Cross-check Timeloop's "Computes (total)" against expected total_macs.
    # Catches the silent-spatial-zero bug where Timeloop runs on 1/256 of the workload
    # because the spatial factor was placed at the wrong level (DOSA's flat_mapping_to_dict
    # lines 113-116 silently zero CX at L2 and KX at L1).
    try:
        # Find latest timeloop-mapper.stats.txt under our scratch tree
        import re as _re
        stats_files = sorted(Path(output_dir).rglob("timeloop-mapper.stats.txt"),
                             key=lambda p: p.stat().st_mtime)
        computes = None
        if stats_files:
            for line in stats_files[-1].read_text().splitlines():
                m = _re.match(r"^\s*Computes \(total\)\s*:\s*([0-9]+)", line)
                if m:
                    computes = int(m.group(1))
                    break
        if computes is not None and computes < 0.99 * total_macs:
            return {
                "valid": False,
                "reason": (
                    f"Timeloop computed only {computes} MAC ops but workload requires "
                    f"{total_macs} (coverage = {100.0*computes/total_macs:.2f}%). "
                    f"This usually means a spatial factor was silently zeroed because it "
                    f"was placed at the wrong level. Gemmini convention: KX only at L2 "
                    f"(Scratchpad), CX only at L1 (Accumulator). Did you put CX at L2 or "
                    f"KX at L1 by mistake?"
                ),
                "mapping_str": mapping_str,
                "computes_reported": computes,
                "computes_expected": total_macs,
                "coverage_pct": 100.0 * computes / total_macs,
            }
    except Exception as _ce:
        # If cross-check itself fails, log but don't reject (would block legitimate evals on infra error)
        pass

    edp = cycles * energy
    return {
        "valid": True,
        "mapping_str": mapping_str,
        "hw_config": {"pe_dim": pe_dim, "sp_size": sp_size, "acc_size": acc_size},
        "cycles": cycles,
        "energy_uJ": energy,            # NOTE: Timeloop's "Energy: X uJ" is uJ — was previously mislabeled energy_nJ
        "energy_nJ": energy,            # kept for backward-compat with already-saved best.json files (DEPRECATED — same value as energy_uJ)
        "edp": edp,                     # uJ × cycles (Timeloop's natural EDP unit)
        "edp_units": "uJ_x_cycles",
        "area": area,
        "eval_duration_s": elapsed,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check-budget", action="store_true")
    p.add_argument("--reset", action="store_true")
    p.add_argument("--max-calls", type=int, default=None)
    args = p.parse_args()

    if args.reset:
        mc = args.max_calls if args.max_calls is not None else 200
        save_budget({"max_calls": mc, "calls_used": 0})
        for f in (HISTORY_FILE, BEST_FILE):
            if f.exists():
                f.unlink()
        print(json.dumps({"reset": True, "max_calls": mc}))
        return 0

    budget = load_budget()
    if args.check_budget:
        print(
            json.dumps(
                {
                    "max_calls": budget["max_calls"],
                    "calls_used": budget["calls_used"],
                    "remaining": budget["max_calls"] - budget["calls_used"],
                }
            )
        )
        return 0

    if budget["calls_used"] >= budget["max_calls"]:
        print(
            json.dumps(
                {
                    "valid": False,
                    "reason": "BUDGET EXHAUSTED",
                    "max_calls": budget["max_calls"],
                    "calls_used": budget["calls_used"],
                }
            )
        )
        return 2

    if not CANDIDATE_FILE.exists():
        print(json.dumps({"valid": False, "reason": f"candidate.yaml not found at {CANDIDATE_FILE}"}))
        return 1

    with open(CANDIDATE_FILE) as f:
        cand = yaml.safe_load(f)
    mapping_str = cand.get("mapping", "").strip()
    rationale = cand.get("rationale", "")
    if not mapping_str:
        result = {"valid": False, "reason": "candidate.yaml has empty 'mapping' field"}
    else:
        result = run_one(mapping_str, rationale)

    budget["calls_used"] += 1
    save_budget(budget)
    is_new_best = update_best(result)

    entry = {
        "call_index": budget["calls_used"],
        "mapping": mapping_str,
        "rationale": rationale,
        "result": result,
        "is_new_best": is_new_best,
    }
    append_history(entry)

    out = {
        **result,
        "calls_used": budget["calls_used"],
        "remaining": budget["max_calls"] - budget["calls_used"],
        "is_new_best": is_new_best,
    }
    if is_new_best:
        out["message"] = "NEW BEST"
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
