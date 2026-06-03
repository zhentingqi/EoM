#!/usr/bin/env python3
"""
Local-search refinement tool for a mapping string.

Takes the agent's current best mapping and generates N perturbations:
 - permutation of the temporal loop order at each level
 - ± one divisor step on each temporal tile factor
 - ± one divisor step on spatial factors (respecting pe_dim cap)

Evaluates each perturbation with Timeloop. Returns the best found.

Each internal Timeloop call this tool runs counts as 1 budget unit (1 for the
seed evaluation plus 1 per perturbation sample), the same accounting as a
direct submit — refinement is budgeted, not free.

Usage:
    python refine_mapping.py --start-mapping "L3[WIO] ..." [--n-samples 50]

Returns JSON to stdout:
    {"best_mapping": "...", "best_edp": ..., "n_evaluated": N, "n_valid": M,
     "duration_s": ..., "improvement_ratio": ...}
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import yaml

WS = Path(__file__).resolve().parent
DOSA_ROOT_RAW = os.environ.get("DOSA_ROOT", "")
DOSA_ROOT = Path(DOSA_ROOT_RAW) if DOSA_ROOT_RAW else None
if DOSA_ROOT is not None and str(DOSA_ROOT) not in sys.path:
    sys.path.insert(0, str(DOSA_ROOT))

WORKLOAD_FILE = WS / "workload.yaml"
HARDWARE_FILE = WS / "hardware.yaml"
BUDGET_FILE = WS / "budget.json"


def _load_budget():
    if not BUDGET_FILE.exists():
        return {"max_calls": 200, "calls_used": 0}
    import json as _j
    return _j.load(open(BUDGET_FILE))


def _save_budget(b):
    import json as _j
    _j.dump(b, open(BUDGET_FILE, "w"), indent=2)


def _consume_one_budget(reason: str = "refine.sh internal Timeloop call"):
    """Charge 1 unit against budget.json. Returns (success, calls_used, max_calls).
    Each internal Timeloop call in refine charges against the same budget as a
    direct submit, so refinement cannot buy extra unbudgeted evaluations.
    """
    b = _load_budget()
    if b["calls_used"] >= b["max_calls"]:
        return False, b["calls_used"], b["max_calls"]
    b["calls_used"] += 1
    _save_budget(b)
    return True, b["calls_used"], b["max_calls"]

# --- Parser for mapping strings ---
_TOK = re.compile(r"^([A-Z])([0-9]+)([XY]?)$")


def parse_mapping(s: str) -> list[dict]:
    """Return [{level_tag, factors: [(dim, value, kind)]}, ...]"""
    blocks = []
    for blk in s.split(" - "):
        parts = blk.split()
        lvl = parts[0]
        facts = []
        for tok in parts[1:]:
            m = _TOK.match(tok)
            if not m:
                raise ValueError(f"malformed token: {tok}")
            dim = m.group(1)
            val = int(m.group(2))
            kind = m.group(3) or "T"  # T temporal, X spatial
            facts.append((dim, val, kind))
        blocks.append({"level": lvl, "factors": facts})
    return blocks


def blocks_to_str(blocks: list[dict]) -> str:
    parts = []
    for b in blocks:
        toks = [b["level"]]
        for dim, val, kind in b["factors"]:
            if kind == "T":
                toks.append(f"{dim}{val}")
            else:
                toks.append(f"{dim}{val}{kind}")
        parts.append(" ".join(toks))
    return " - ".join(parts)


def divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def neighbors_for_factor(v: int, max_val: int) -> list[int]:
    """Return divisors of max_val that are near v (prev, current, next)."""
    divs = divisors(max_val)
    if v not in divs:
        # Snap to nearest divisor
        v = min(divs, key=lambda d: abs(d - v))
    idx = divs.index(v)
    out = {v}
    for delta in (-2, -1, 1, 2):
        j = idx + delta
        if 0 <= j < len(divs):
            out.add(divs[j])
    return sorted(out)


def generate_perturbations(mapping_str: str, prob_dims: dict, pe_dim: int, n_samples: int, rng: random.Random) -> list[str]:
    """Return up to n_samples perturbed mapping strings derived from the input."""
    blocks = parse_mapping(mapping_str)
    out = set()

    # Strategy 1: swap temporal loop orders at a random level
    for _ in range(n_samples):
        new_blocks = copy.deepcopy(blocks)
        lvl_idx = rng.randrange(len(new_blocks))
        temp_idx = [i for i, f in enumerate(new_blocks[lvl_idx]["factors"]) if f[2] == "T"]
        if len(temp_idx) >= 2:
            rng.shuffle(temp_idx)  # just permute which ones move
            # Simpler: random shuffle of the temporal sub-sequence
            temp_facts = [new_blocks[lvl_idx]["factors"][i] for i in temp_idx]
            rng.shuffle(temp_facts)
            for i, t in zip(temp_idx, temp_facts):
                new_blocks[lvl_idx]["factors"][i] = t
        out.add(blocks_to_str(new_blocks))
        if len(out) >= n_samples // 2:
            break

    # Strategy 2: perturb a tile size to nearest divisor
    for _ in range(n_samples):
        new_blocks = copy.deepcopy(blocks)
        lvl_idx = rng.randrange(len(new_blocks))
        facts = new_blocks[lvl_idx]["factors"]
        if not facts:
            continue
        fi = rng.randrange(len(facts))
        dim, val, kind = facts[fi]
        # Cap: for spatial, must be ≤ pe_dim
        cap = pe_dim if kind in ("X", "Y") else prob_dims.get(dim, val)
        neighbors = neighbors_for_factor(val, max(1, cap))
        if len(neighbors) > 1:
            new_val = rng.choice([n for n in neighbors if n != val] or neighbors)
            facts[fi] = (dim, new_val, kind)
            out.add(blocks_to_str(new_blocks))
        if len(out) >= n_samples:
            break

    return list(out)[:n_samples]


def eval_one_via_dosa(mapping_str: str, prob, arch_config) -> dict:
    """Run one mapping through DOSA's Timeloop pipeline. Return {valid, edp, cycles, energy}."""
    from dataset.common import mapping_utils

    try:
        flat = mapping_utils.process_mapping(mapping_str, prob.shape)
        mapping_dict = arch_config.flat_mapping_to_dict(prob.shape, flat)
        row = arch_config.run_mapping_from_dict(prob, mapping_dict)
    except Exception as e:
        return {"valid": False, "reason": f"decode failed: {e}"}
    if not row:
        return {"valid": False, "reason": "Timeloop rejected"}
    try:
        cyc = float(row["target.cycle"])
        eng = float(row["target.energy"])
    except Exception as e:
        return {"valid": False, "reason": f"missing fields: {e}"}
    if cyc <= 0 or eng <= 0:
        return {"valid": False, "reason": "bad cycles/energy"}
    return {"valid": True, "edp": cyc * eng, "cycles": cyc, "energy_nJ": eng}


def validate_mapping(mapping_str: str, prob_dims: dict, pe_dim: int) -> tuple[bool, str]:
    """Quick local validity: duplicates + factor product == prob dim."""
    try:
        blocks = parse_mapping(mapping_str)
    except ValueError as e:
        return False, str(e)
    # Duplicates
    for b in blocks:
        seen = set()
        for dim, val, kind in b["factors"]:
            key = (dim, kind)
            if key in seen:
                return False, f"duplicate ({dim},{kind}) at {b['level']}"
            seen.add(key)
    # Factor product per dim
    by_dim = {d: 1 for d in ("R", "S", "P", "Q", "C", "K", "N")}
    for b in blocks:
        for dim, val, kind in b["factors"]:
            by_dim[dim] = by_dim.get(dim, 1) * val
    for dim, p in prob_dims.items():
        if dim not in by_dim:
            continue
        if by_dim[dim] != p:
            return False, f"factor product for {dim} = {by_dim[dim]}, expected {p}"
    # Spatial cap
    for b in blocks:
        sp = 1
        for dim, val, kind in b["factors"]:
            if kind in ("X", "Y"):
                sp *= val
        if sp > pe_dim:
            return False, f"spatial product {sp} > pe_dim={pe_dim} at {b['level']}"
    return True, "ok"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start-mapping", required=True, help="seed mapping string")
    p.add_argument("--n-samples", type=int, default=50, help="perturbations to evaluate")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    if DOSA_ROOT is None:
        print(json.dumps({"error": "DOSA_ROOT environment variable is required"}))
        return 2

    # Load workload + hw
    with open(WORKLOAD_FILE) as f:
        wl_doc = yaml.safe_load(f)
    with open(HARDWARE_FILE) as f:
        hw_doc = yaml.safe_load(f)
    from dataset.workloads import Prob
    from dataset.hw import init_hw_config

    prob = Prob(wl_doc)
    hw_config = [int(hw_doc["pe_dim"]), int(hw_doc["sp_size"]), int(hw_doc["acc_size"])]
    pe_dim = hw_config[0]

    # Scratch
    scratch_raw = os.environ.get("ARCHGYM_SCRATCH", "")
    if not scratch_raw:
        print(json.dumps({"error": "ARCHGYM_SCRATCH environment variable is required"}))
        return 2
    scratch_root = Path(scratch_raw) / f"refine_{os.getpid()}_{int(time.time())}"
    scratch_root.mkdir(parents=True, exist_ok=True)
    arch_config = init_hw_config("gemmini", hw_config, scratch_root)

    rng = random.Random(args.seed if args.seed is not None else int(time.time()))

    # Eval the starting point first as a reference. Each Timeloop call inside
    # refine charges against budget.json, so refinement cannot exceed the
    # configured per-task budget (budget_per_task).
    t0 = time.time()
    ok_b, used, mx = _consume_one_budget()
    if not ok_b:
        print(json.dumps({"error": "BUDGET EXHAUSTED — refine.sh now consumes budget per audit fix",
                          "calls_used": used, "max_calls": mx}))
        return 2
    start_result = eval_one_via_dosa(args.start_mapping, prob, arch_config)
    start_edp = start_result.get("edp", float("inf")) if start_result["valid"] else None

    # Generate perturbations
    samples = generate_perturbations(args.start_mapping, prob.prob, pe_dim, args.n_samples, rng)

    best_edp = start_edp if start_edp is not None else float("inf")
    best_mapping = args.start_mapping if start_edp is not None else None
    n_valid = 1 if start_result["valid"] else 0
    n_evaluated = 1
    budget_exhausted = False

    for cand in samples:
        n_evaluated += 1
        # Quick local validation (free; no Timeloop call)
        ok, _ = validate_mapping(cand, prob.prob, pe_dim)
        if not ok:
            continue
        # Charge budget for this Timeloop call
        ok_b, used, mx = _consume_one_budget()
        if not ok_b:
            budget_exhausted = True
            break
        r = eval_one_via_dosa(cand, prob, arch_config)
        if r.get("valid"):
            n_valid += 1
            if r["edp"] < best_edp:
                best_edp = r["edp"]
                best_mapping = cand

    duration = time.time() - t0

    improvement = None
    if start_edp is not None and best_edp < start_edp:
        improvement = start_edp / best_edp

    final_b = _load_budget()
    out = {
        "best_mapping": best_mapping,
        "best_edp": best_edp if best_edp != float("inf") else None,
        "start_edp": start_edp,
        "improvement_ratio": improvement,
        "n_evaluated": n_evaluated,
        "n_valid": n_valid,
        "duration_s": duration,
        "calls_used": final_b["calls_used"],
        "remaining": final_b["max_calls"] - final_b["calls_used"],
        "budget_exhausted": budget_exhausted,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
