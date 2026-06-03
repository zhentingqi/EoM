#!/usr/bin/env python3
"""Standalone local pre-check for the agent workspace.

Reads `candidate.yaml` (hw + mapping) and `workload.yaml` (problem dims) and
runs the SAME validity rules eval.py enforces post-Timeloop:
  1. Token grammar (4 ' - '-joined level blocks; tokens like K16, C16X).
  2. Per-dim factor product == workload prob.
  3. KX only at L2[WI]; CX only at L1[O]; L3 / L0 no spatial.
  4. L0[W]: only P/Q temporal factors > 1.
  5. Spatial product per level ≤ pe_dim.

Exits 0 always (so submit.sh can branch on the JSON output, not exit code).
Prints a JSON object to stdout:

  Locally OK:
      {"valid": true, "ok": true, "reason": null}

  Locally rejected (BEFORE budget is consumed):
      {
        "valid": false, "ok": false, "local_pre_check_failed": true,
        "reason": "<actionable error>",
        "calls_used": <unchanged>, "remaining": <unchanged>
      }

submit.sh, on seeing local_pre_check_failed=true, returns this JSON to the
agent without invoking eval.py — so no Timeloop call happens, no budget gets
charged, and the agent reads the same hint Hayek's Executor sees in retry mode.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

try:
    import yaml as _yaml  # PyYAML — preferred when available
    _HAVE_PYYAML = True
except Exception:
    _yaml = None
    _HAVE_PYYAML = False


def _safe_load_yaml(text):
    """Best-effort YAML loader. Uses PyYAML if available; otherwise falls back
    to a tiny parser sufficient for the candidate.yaml / workload.yaml schemas
    we control. The fallback handles two-level dicts with int/string scalars
    and quoted-string values, which is all our files contain."""
    if _HAVE_PYYAML:
        return _yaml.safe_load(text)
    # Mini YAML parser — safe for files of the form:
    #   problem:
    #     C: 256
    #     ...
    #   hw:
    #     pe_dim: 16
    #     ...
    #   mapping: "L3[WIO] ..."
    #   rationale: "..."
    out: dict = {}
    cur_key = None
    cur_section: dict | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        # Top-level key (no leading whitespace)
        if not line.startswith(" ") and not line.startswith("\t"):
            stripped = line.strip()
            if ":" in stripped:
                k, _, v = stripped.partition(":")
                k = k.strip()
                v = v.strip()
                if not v:
                    cur_key = k
                    cur_section = {}
                    out[k] = cur_section
                else:
                    # scalar value
                    out[k] = _scalar(v)
                    cur_key = None
                    cur_section = None
        else:
            # nested key
            if cur_section is None:
                continue
            stripped = line.strip()
            if ":" in stripped:
                k, _, v = stripped.partition(":")
                cur_section[k.strip()] = _scalar(v.strip())
    return out


def _scalar(v):
    if not v:
        return ""
    # Strip optional surrounding quotes
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    # Try int
    try:
        return int(v)
    except ValueError:
        pass
    # Try float
    try:
        return float(v)
    except ValueError:
        pass
    return v

WS = Path(__file__).resolve().parent
WORKLOAD_FILE = WS / "workload.yaml"
HARDWARE_FILE = WS / "hardware.yaml"
CANDIDATE_FILE = WS / "candidate.yaml"
BUDGET_FILE = WS / "budget.json"


# Mirrors env.py exactly.
_MAPPING_TOKEN_RE = re.compile(r"^([RSPQCKN])([0-9]+)([XY]?)$")
_VALID_DIMS = ("R", "S", "P", "Q", "C", "K", "N")
_SPATIAL_ALLOWED_AT_LEVEL = {
    "L3[WIO]": set(),
    "L2[WI]":  {"K"},
    "L1[O]":   {"C"},
    "L0[W]":   set(),
}
_TEMPORAL_ALLOWED_AT_LEVEL = {
    "L3[WIO]": set(_VALID_DIMS),
    "L2[WI]":  set(_VALID_DIMS),
    "L1[O]":   set(_VALID_DIMS),
    "L0[W]":   {"P", "Q"},
}
_LEVEL_ORDER = ("L3[WIO]", "L2[WI]", "L1[O]", "L0[W]")


def _parse_mapping_blocks(mapping_str):
    if not mapping_str or not isinstance(mapping_str, str):
        return None, "empty mapping string"
    blocks_raw = mapping_str.split(" - ")
    if len(blocks_raw) != 4:
        return None, (
            f"mapping must have 4 ' - '-joined level blocks, got {len(blocks_raw)}"
        )
    out = []
    for blk_idx, blk in enumerate(blocks_raw):
        parts = blk.split()
        if not parts:
            return None, f"block {blk_idx} is empty"
        lvl = parts[0]
        if lvl != _LEVEL_ORDER[blk_idx]:
            return None, (
                f"block {blk_idx} has level tag {lvl!r}, expected "
                f"{_LEVEL_ORDER[blk_idx]!r}"
            )
        factors = []
        seen = set()
        for tok in parts[1:]:
            m = _MAPPING_TOKEN_RE.match(tok)
            if not m:
                return None, f"malformed token {tok!r} at level {lvl}"
            dim = m.group(1)
            kind = m.group(3) or "T"
            try:
                val = int(m.group(2))
            except ValueError:
                return None, f"non-integer factor in token {tok!r} at level {lvl}"
            if val <= 0:
                return None, f"non-positive factor {val} in token {tok!r} at {lvl}"
            key = (dim, kind)
            if key in seen:
                return None, (
                    f"duplicate {dim}{'(spatial)' if kind in ('X','Y') else '(temporal)'}"
                    f" at {lvl}"
                )
            seen.add(key)
            factors.append((dim, val, kind))
        out.append({"level": lvl, "factors": factors})
    return out, "ok"


def validate_mapping_locally(prob, hw, mapping_str):
    pe_dim = int(hw.get("pe_dim", 16)) if hw else 16
    blocks, err = _parse_mapping_blocks(mapping_str)
    if blocks is None:
        return False, f"mapping parse error: {err}"

    by_dim = {d: 1 for d in _VALID_DIMS}
    for blk in blocks:
        for dim, val, _kind in blk["factors"]:
            by_dim[dim] = by_dim.get(dim, 1) * val
    for dim in _VALID_DIMS:
        expected = int(prob.get(dim, 1))
        got = by_dim.get(dim, 1)
        if got != expected:
            per_level = []
            for blk in blocks:
                for d2, v2, k2 in blk["factors"]:
                    if d2 == dim:
                        suffix = k2 if k2 in ("X", "Y") else "T"
                        per_level.append(f"{blk['level']}:{dim}{v2}{suffix}")
            breakdown = ", ".join(per_level) if per_level else f"(no {dim} factors)"
            return False, (
                f"factor product for {dim} = {got} but workload {dim} = {expected} "
                f"(off by factor {got}/{expected} = {got/expected:.4f}). "
                f"Your {dim} factors across levels: {breakdown}. "
                f"Fix: ensure {dim}_L3 * {dim}_L2 * {dim}_L1 * {dim}_L0 = {expected}."
            )

    for blk in blocks:
        lvl = blk["level"]
        sp_allowed = _SPATIAL_ALLOWED_AT_LEVEL[lvl]
        tmp_allowed = _TEMPORAL_ALLOWED_AT_LEVEL[lvl]
        sp_product = 1
        for dim, val, kind in blk["factors"]:
            if kind in ("X", "Y"):
                if dim not in sp_allowed:
                    valid_for_dim = (
                        "L2[WI]" if dim == "K" else
                        ("L1[O]" if dim == "C" else "(no level)")
                    )
                    return False, (
                        f"spatial factor {dim}{val}{kind} is not allowed at {lvl}. "
                        f"On Gemmini, KX is valid ONLY at L2[WI], CX is valid ONLY at L1[O]. "
                        f"Move {dim}{val}{kind} to {valid_for_dim}."
                    )
                sp_product *= val
            else:
                if val > 1 and dim not in tmp_allowed:
                    return False, (
                        f"L0[W] (registers) accepts only P/Q temporal factors > 1; "
                        f"got {dim}{val} at L0. Set {dim}=1 at L0 (or omit) and place "
                        f"the {dim}={val} factor at L3/L2/L1 instead."
                    )
        if sp_product > pe_dim:
            return False, (
                f"spatial product at {lvl} = {sp_product} exceeds pe_dim = {pe_dim}. "
                f"Reduce the spatial factor at {lvl} so the product ≤ pe_dim."
            )
    return True, "ok"


def _load_budget():
    if not BUDGET_FILE.exists():
        return {"max_calls": 200, "calls_used": 0}
    try:
        return json.loads(BUDGET_FILE.read_text())
    except Exception:
        return {"max_calls": 200, "calls_used": 0}


def main():
    if not CANDIDATE_FILE.exists():
        print(json.dumps({
            "valid": False, "ok": False,
            "reason": f"candidate.yaml not found at {CANDIDATE_FILE}",
        }))
        return 0
    if not WORKLOAD_FILE.exists():
        print(json.dumps({
            "valid": False, "ok": False,
            "reason": f"workload.yaml not found at {WORKLOAD_FILE}",
        }))
        return 0

    try:
        cand = _safe_load_yaml(CANDIDATE_FILE.read_text())
    except Exception as e:
        print(json.dumps({
            "valid": False, "ok": False,
            "reason": f"candidate.yaml parse error: {e}",
        }))
        return 0
    try:
        wl = _safe_load_yaml(WORKLOAD_FILE.read_text())
    except Exception as e:
        print(json.dumps({
            "valid": False, "ok": False,
            "reason": f"workload.yaml parse error: {e}",
        }))
        return 0

    prob = (wl or {}).get("problem", wl or {})
    if not isinstance(prob, dict):
        prob = {}
    prob = {k: int(prob.get(k, 1)) for k in _VALID_DIMS}

    hw = (cand or {}).get("hw") or {}
    if not isinstance(hw, dict):
        hw = {}
    mapping = (cand or {}).get("mapping", "") or ""
    if not isinstance(mapping, str):
        mapping = ""

    ok, msg = validate_mapping_locally(prob, hw, mapping)
    if ok:
        print(json.dumps({"valid": True, "ok": True, "reason": None}))
        return 0

    budget = _load_budget()
    print(json.dumps({
        "valid": False,
        "ok": False,
        "local_pre_check_failed": True,
        "reason": msg,
        # Budget is unchanged — we did NOT call eval.py.
        "calls_used": budget.get("calls_used", 0),
        "remaining": budget.get("max_calls", 0) - budget.get("calls_used", 0),
        "max_calls": budget.get("max_calls", 0),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
