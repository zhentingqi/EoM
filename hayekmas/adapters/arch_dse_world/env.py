"""
arch_dse_world adapter environment layer.

Each task = one ResNet-50 layer mapping-search problem on a Gemmini-style
accelerator. An ArchDseEnv wraps a per-task host-side workspace; agents
communicate through its public state (history, advice, direction buffers).

Within one episode, the agent population auctions through a fixed
information chain:

    history.jsonl  ←(auto)─  Executor  ←(direction)── Planner  ←(advice)── Historian

The Executor submits a (hardware, mapping) pair to the Timeloop +
Accelergy backend (DOSA paper's evaluation pipeline). The simulator
returns cycles, energy, and EDP (energy-delay product). We track the
running-best EDP per task and emit per-step reward whenever the latest
submit beats the previous best.

Timeloop / Gemmini notation used throughout this file (and the
prompts the agents see):

  Workload dimensions (per conv layer)
    K  output channels    P  output height   R  filter height
    C  input channels     Q  output width    S  filter width

  Hardware parameters
    pe_dim   square root of the PE-array dimension (pe_dim=128 means
             a 128×128 systolic-array tile).
    sp_size  scratchpad size at L2 (bytes).
    acc_size accumulator size at L1 (bytes).

  Memory hierarchy (closest to PEs at L0, furthest at L3)
    L0[W]    register file holding the Weight tile
    L1[O]    accumulator holding the Output tile
    L2       on-chip scratchpad (shared across PEs)
    L3[WIO]  off-chip DRAM holding Weights, Inputs, Outputs

  Mapping-token shorthand
    K_spatial_L2 = n   →  map factor n of K spatially at L2
    K_temporal_L3 = n  →  map factor n of K temporally at L3
    Same pattern for C, P, Q, R, S and for any L0..L3.
    PQ_at_L0 = [P, Q]  →  the entire (P, Q) tile sits at L0[W]
                          (weight-stationary).
    PQ_at_L1 = [P, Q]  →  the entire (P, Q) tile sits at L1[O]
                          (output-stationary).
    KX, CX             →  shorthand for "K mapped spatially at the
                          referenced level", "C mapped spatially at
                          the referenced level".
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml as _yaml_mod

from hayekmas.base.agent import BaseAction
from hayekmas.base.env import BaseEnv
from hayekmas.base.config import DEFAULT_HAYEK_CONFIG, RewardConfig


# Simulator dir bundled inside this adapter:
#   simulator/workloads/resnet50/*.yaml   → 24 layer descriptions
#   simulator/workspace/                  → eval.py + Timeloop/Accelergy hardware template
SIMULATOR_DIR = Path(__file__).resolve().parent / "simulator"
WORKSPACE_TEMPLATE = SIMULATOR_DIR / "workspace"
WORKLOADS_DIR = SIMULATOR_DIR / "workloads"
DEFAULT_RESNET50_DIR = WORKLOADS_DIR / "resnet50"

# External simulator dependencies are configured through environment variables.
# Required external dependencies that must point at real installs:
#   - the conda env with Timeloop + Accelergy + the DOSA repo installed.
#   - DOSA_ROOT: working copy of the DOSA codebase
#     (Hong et al., MICRO 2023 — https://github.com/ucb-bar/dosa).
#   - ARCHGYM_SCRATCH: a large, fast local scratch directory.
# NOTE: Gurobi is NOT needed for this eval path — it only runs Timeloop
# on a given (hw, mapping) pair. Gurobi is used only by DOSA's own
# mapping-search optimizer (the cached baseline). GRB_LICENSE_FILE is
# still exported below for that optional case, but is unused here.
DEFAULT_CONDA_PROFILE = os.environ.get(
    "CONDA_PROFILE", ""
)
DEFAULT_CONDA_ENV = os.environ.get(
    "CONDA_ENV", ""
)
DEFAULT_DOSA_ROOT = os.environ.get(
    "DOSA_ROOT",
    "",
)
DEFAULT_GUROBI_LIC = os.environ.get(
    "GUROBI_LIC", ""
)
DEFAULT_SCRATCH = os.environ.get(
    "ARCHGYM_SCRATCH", ""
)


# ═══════════════════════════════════════════════════════════════════════════
# ACTION
# ═══════════════════════════════════════════════════════════════════════════

class DSEAction(BaseAction):
    """DSE action emitted by one of the three roles (Historian, Planner,
    Executor).

    Three kinds:
      - "advice":    Historian's analysis text (no submission)
      - "direction": Planner's strategy text (no submission)
      - "submit":    Executor's package — text + a list of (hw, mapping) pairs
                     plus their Timeloop results.
    Episode termination is solely budget-driven (calls_used >= max_calls).
    """
    KIND_ADVICE = "advice"
    KIND_DIRECTION = "direction"
    KIND_SUBMIT = "submit"

    def __init__(
        self,
        text: str,
        author: str,
        role: str,
        kind: str,
        *,
        submissions: Optional[List[Dict[str, Any]]] = None,
        chose_historian: Optional[str] = None,
    ):
        self.text = text or ""
        self.author = author
        self.role = role
        self.kind = kind
        self.submissions: List[Dict[str, Any]] = submissions or []
        self.chose_historian = chose_historian
        self.is_final = False  # only budget exhaustion ends episodes here

    def __repr__(self) -> str:
        n = len(self.submissions)
        suffix = f", subs={n}" if n else ""
        return f"DSEAction({self.kind.upper()}, author={self.author!r}{suffix})"


# ═══════════════════════════════════════════════════════════════════════════
# TASK
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DSETask:
    id: str                              # layer name, e.g. "_outputs_input.83"
    workload: str                        # network name, e.g. "resnet50"
    workload_yaml: Path                  # absolute path to layer yaml
    max_evaluations: int = 10            # Timeloop calls budget
    chain_steps_per_episode: int = 20
    description: str = ""
    session_idx: int = 0                 # NEW: 0..N-1; distinguishes sessions of same layer

    @property
    def session_id(self) -> str:
        """Used to name per-task workspace dir uniquely across sessions."""
        return f"{self.id}_s{self.session_idx}"

    @classmethod
    def from_layer(cls, layer_name: str, *, workload: str = "resnet50",
                   budget: int = 10, chain_steps_per_episode: int = 20,
                   session_idx: int = 0) -> "DSETask":
        wl = WORKLOADS_DIR / workload / f"{layer_name}.yaml"
        if not wl.is_file():
            raise FileNotFoundError(f"layer yaml not found: {wl}")
        return cls(
            id=layer_name,
            workload=workload,
            workload_yaml=wl,
            max_evaluations=budget,
            chain_steps_per_episode=chain_steps_per_episode,
            description=f"Map {workload}/{layer_name} session {session_idx} → minimize EDP.",
            session_idx=session_idx,
        )


def load_dse_tasks_from_jsonl(paths: List[Path], limit: Optional[int] = None) -> List[DSETask]:
    tasks: List[DSETask] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                layer = row.get("layer") or row.get("id")
                if not layer:
                    continue
                tasks.append(
                    DSETask.from_layer(
                        layer,
                        workload=row.get("workload", "resnet50"),
                        budget=int(row.get("budget", 80)),
                        chain_steps_per_episode=int(row.get("chain_steps_per_episode", 20)),
                    )
                )
                if limit is not None and len(tasks) >= limit:
                    return tasks
    return tasks


def load_resnet50_unique_layers() -> List[str]:
    """Read the 24 unique ResNet-50 kernels from the DOSA paper manifest."""
    manifest = DEFAULT_RESNET50_DIR / "unique_layers.yaml"
    if not manifest.is_file():
        raise FileNotFoundError(f"unique_layers.yaml not found at {manifest}")
    import yaml as _yaml
    data = _yaml.safe_load(manifest.read_text())
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and "layers" in data:
        return [str(x) for x in data["layers"]]
    raise ValueError(f"unrecognized unique_layers.yaml format: {type(data)}")


# ═══════════════════════════════════════════════════════════════════════════
# TIMELOOP+ACCELERGY INVOCATION (host conda env, not container)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TimeloopResult:
    valid: bool
    edp: Optional[float] = None
    cycles: Optional[float] = None
    energy_uJ: Optional[float] = None
    area: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


def _stage_workspace(out_dir: Path, workload_yaml: Path) -> Path:
    """Build a host-side workspace that mirrors backends/dosa/workspace."""
    ws = (out_dir / "workspace").resolve()
    ws.mkdir(parents=True, exist_ok=True)
    for fname in (
        "eval.py", "action_space.md", "hardware.yaml",
        "submit.sh", "README.md",
        "precheck.py",  # local pre-check helper
    ):
        src = WORKSPACE_TEMPLATE / fname
        if src.is_file():
            shutil.copy(src, ws / fname)
    shutil.copy(workload_yaml, ws / "workload.yaml")
    (ws / "notes.md").write_text("")
    # Placeholder candidate.yaml (NEW format: hw + mapping string)
    (ws / "candidate.yaml").write_text(
        "hw:\n  pe_dim: 16\n  sp_size: 128\n  acc_size: 32\n"
        "mapping: \"L3[WIO] N1 - L2[WI] N1 - L1[O] N1 - L0[W] N1\"\n"
        "rationale: \"placeholder — agent will overwrite\"\n"
    )
    for exec_name in ("submit.sh", "eval.py", "precheck.py"):
        f = ws / exec_name
        if f.is_file():
            os.chmod(f, 0o755)
    return ws


def _build_eval_command(workspace: Path, *, args: str = "") -> List[str]:
    """Build a direct invocation of the dosa_dse conda env's python on eval.py.

    `conda activate` on shared lustre takes ~3 minutes per call. We skip it
    by calling the env's interpreter directly and exporting PATH/env-vars
    that downstream tools (Timeloop, Accelergy) need.

    The command runs OUTSIDE any container; isolation comes from the
    HayekMAS engine's own subprocess boundary, not podman.
    """
    conda_env = os.environ.get("DSE_CONDA_ENV", DEFAULT_CONDA_ENV)
    dosa_root = os.environ.get("DOSA_ROOT", DEFAULT_DOSA_ROOT)
    gurobi = os.environ.get("GRB_LICENSE_FILE", DEFAULT_GUROBI_LIC)
    scratch = os.environ.get("ARCHGYM_SCRATCH", DEFAULT_SCRATCH)
    missing = [
        name
        for name, value in {
            "DSE_CONDA_ENV or CONDA_ENV": conda_env,
            "DOSA_ROOT": dosa_root,
            "ARCHGYM_SCRATCH": scratch,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing DSE simulator environment variables: " + ", ".join(missing)
        )
    py = str(Path(conda_env) / "bin" / "python3")
    cmd = (
        f"export PATH={shlex.quote(str(Path(conda_env) / 'bin'))}:"
        f"{shlex.quote(str(Path(conda_env) / 'share' / 'cacti'))}:$PATH && "
        f"export LD_LIBRARY_PATH={shlex.quote(str(Path(conda_env) / 'lib'))}:"
        f"${{LD_LIBRARY_PATH:-}} && "
        f"export PYTHONNOUSERSITE=1 && "
        f"export DOSA_ROOT={shlex.quote(dosa_root)} && "
        f"export GRB_LICENSE_FILE={shlex.quote(gurobi)} && "
        f"export ARCHGYM_SCRATCH={shlex.quote(scratch)} && "
        f"cd {shlex.quote(str(workspace))} && {shlex.quote(py)} eval.py {args}".strip()
    )
    return ["bash", "-c", cmd]


def _extract_trailing_json(out: str) -> Optional[Dict[str, Any]]:
    """Find the trailing top-level JSON object in `out`, ignoring log noise.

    eval.py emits one pretty-printed JSON object as its final stdout. DOSA's
    pipeline can prepend log lines like `[2026-... INFO ...] ...`. We walk
    backward from the last `}` to find the matching `{` (depth=0).
    """
    end = out.rfind("}")
    if end < 0:
        return None
    depth = 0
    for i in range(end, -1, -1):
        c = out[i]
        if c == "}":
            depth += 1
        elif c == "{":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(out[i : end + 1])
                except Exception:
                    return None
    return None


def _reset_budget_in_workspace(workspace: Path, budget: int) -> None:
    """Wipe history.jsonl + best.json, reset budget.json. Use only when
    starting a brand-new layer (NOT between sessions of same layer)."""
    cmd = _build_eval_command(workspace, args=f"--reset --max-calls {budget}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(
            f"reset budget failed (rc={proc.returncode}): "
            f"stdout={proc.stdout[-400:]!r} stderr={proc.stderr[-400:]!r}"
        )


def _set_budget_only(workspace: Path, budget: int) -> None:
    """Reset budget counter to 0/<budget> WITHOUT wiping history.jsonl or
    best.json. Used between sessions of the same layer so the cumulative
    record-best persists and agents can see prior session submissions."""
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "budget.json").write_text(
        json.dumps({"max_calls": int(budget), "calls_used": 0}, indent=2)
    )


# ═══════════════════════════════════════════════════════════════════════════
# LOCAL PRE-CHECK + INTENT-TO-MAPPING (Plans A + C)
# ═══════════════════════════════════════════════════════════════════════════

# Parsing token grammar: <DIM><INT>[X|Y]
_MAPPING_TOKEN_RE = re.compile(r"^([RSPQCKN])([0-9]+)([XY]?)$")
_VALID_DIMS = ("R", "S", "P", "Q", "C", "K", "N")

# Per-level allow-lists (matches DOSA's flat_mapping_to_dict; see action_space.md):
#   L3 / L0 → temporal only; L2 → KX only; L1 → CX only.
_SPATIAL_ALLOWED_AT_LEVEL = {
    "L3[WIO]": set(),       # no spatial
    "L2[WI]":  {"K"},       # only KX
    "L1[O]":   {"C"},       # only CX
    "L0[W]":   set(),       # no spatial
}
# Per-level temporal allow-lists. L0[W] = registers — only P/Q permitted (>1).
_TEMPORAL_ALLOWED_AT_LEVEL = {
    "L3[WIO]": set(_VALID_DIMS),
    "L2[WI]":  set(_VALID_DIMS),
    "L1[O]":   set(_VALID_DIMS),
    "L0[W]":   {"P", "Q"},  # only P/Q non-trivial; everything else must be 1 (or omitted)
}
_LEVEL_ORDER = ("L3[WIO]", "L2[WI]", "L1[O]", "L0[W]")


def _parse_mapping_blocks(mapping_str: str) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """Split mapping into per-level token lists. Returns (blocks, error)."""
    if not mapping_str or not isinstance(mapping_str, str):
        return None, "empty mapping string"
    blocks_raw = mapping_str.split(" - ")
    if len(blocks_raw) != 4:
        return None, f"mapping must have 4 ' - '-joined level blocks, got {len(blocks_raw)}"
    out: List[Dict[str, Any]] = []
    for blk_idx, blk in enumerate(blocks_raw):
        parts = blk.split()
        if not parts:
            return None, f"block {blk_idx} is empty"
        lvl = parts[0]
        if lvl != _LEVEL_ORDER[blk_idx]:
            return None, (
                f"block {blk_idx} has level tag {lvl!r}, expected "
                f"{_LEVEL_ORDER[blk_idx]!r} (level order is fixed L3,L2,L1,L0)"
            )
        factors: List[Tuple[str, int, str]] = []
        seen: set = set()
        for tok in parts[1:]:
            m = _MAPPING_TOKEN_RE.match(tok)
            if not m:
                return None, f"malformed token {tok!r} at level {lvl}"
            dim, val_s, kind = m.group(1), m.group(2), m.group(3) or "T"
            try:
                val = int(val_s)
            except ValueError:
                return None, f"non-integer factor in token {tok!r} at level {lvl}"
            if val <= 0:
                return None, f"non-positive factor {val} in token {tok!r} at level {lvl}"
            key = (dim, kind)
            if key in seen:
                return None, (
                    f"duplicate {dim}{'(spatial)' if kind in ('X', 'Y') else '(temporal)'}"
                    f" at {lvl}"
                )
            seen.add(key)
            factors.append((dim, val, kind))
        out.append({"level": lvl, "factors": factors})
    return out, "ok"


def validate_mapping_locally(
    prob: Dict[str, int],
    hw: Dict[str, int],
    mapping_str: str,
) -> Tuple[bool, str]:
    """Local-only validity check. NO Timeloop call. Returns (ok, error_msg).

    Mirrors the rules eval.py enforces *post-Timeloop*:
      1. Token grammar (uses _MAPPING_TOKEN_RE; rejects duplicate dim/kind keys).
      2. Per-dim factor product across all levels == prob[D].
      3. Spatial slot: KX only at L2[WI]; CX only at L1[O]. L3/L0 no spatial.
      4. L0[W]: only P/Q temporal factors > 1 are permitted.
      5. Spatial product per level <= hw["pe_dim"].

    On success returns (True, "ok"). On failure returns (False, "<actionable error>")
    — the error string is inserted directly into the next LLM retry prompt.
    """
    pe_dim = int(hw.get("pe_dim", 16)) if hw else 16

    blocks, err = _parse_mapping_blocks(mapping_str)
    if blocks is None:
        return False, f"mapping parse error: {err}"

    # 2) per-dim factor product
    by_dim: Dict[str, int] = {d: 1 for d in _VALID_DIMS}
    for blk in blocks:
        for dim, val, _kind in blk["factors"]:
            by_dim[dim] = by_dim.get(dim, 1) * val
    for dim in _VALID_DIMS:
        expected = int(prob.get(dim, 1))
        got = by_dim.get(dim, 1)
        if got != expected:
            # Build a per-level breakdown so the LLM sees exactly what it set
            per_level = []
            for blk in blocks:
                for d2, v2, k2 in blk["factors"]:
                    if d2 == dim:
                        suffix = k2 if k2 in ("X", "Y") else "T"
                        per_level.append(f"{blk['level']}:{dim}{v2}{suffix}")
            breakdown = ", ".join(per_level) if per_level else f"(no {dim} factors mentioned)"
            return False, (
                f"factor product for {dim} = {got} but workload {dim} = {expected} "
                f"(off by factor {got}/{expected} = {got/expected:.4f}). "
                f"Your {dim} factors across levels: {breakdown}. "
                f"Fix: ensure {dim}_L3 * {dim}_L2 * {dim}_L1 * {dim}_L0 = {expected}."
            )

    # 3 + 4) per-level legality (spatial slot + L0 stationary rule)
    for blk in blocks:
        lvl = blk["level"]
        sp_allowed = _SPATIAL_ALLOWED_AT_LEVEL[lvl]
        tmp_allowed = _TEMPORAL_ALLOWED_AT_LEVEL[lvl]
        sp_product = 1
        for dim, val, kind in blk["factors"]:
            if kind in ("X", "Y"):
                if dim not in sp_allowed:
                    valid_lvl_for_spatial = "L2[WI]" if dim == "K" else (
                        "L1[O]" if dim == "C" else "(no level)"
                    )
                    return False, (
                        f"spatial factor {dim}{val}{kind} is not allowed at {lvl}. "
                        f"On Gemmini, KX is valid ONLY at L2[WI], CX is valid ONLY at L1[O]. "
                        f"Move {dim}{val}{kind} to {valid_lvl_for_spatial}."
                    )
                sp_product *= val
            else:
                # temporal
                if val > 1 and dim not in tmp_allowed:
                    return False, (
                        f"L0[W] (registers) accepts only P/Q temporal factors > 1; "
                        f"got {dim}{val} at L0. Set {dim}=1 at L0 (or omit) and place "
                        f"the {dim}={val} factor at L3/L2/L1 instead."
                    )

        # 5) spatial product cap
        if sp_product > pe_dim:
            return False, (
                f"spatial product at {lvl} = {sp_product} exceeds pe_dim = {pe_dim}. "
                f"Reduce the spatial factor at {lvl} so the product ≤ pe_dim."
            )

    return True, "ok"


def intent_to_mapping(
    hw: Dict[str, int],
    intent: Dict[str, Any],
    prob: Dict[str, int],
) -> Tuple[bool, Optional[str], str]:
    """Convert a structured intent dict into a guaranteed-valid Gemmini mapping string.

    Schema:
        {
          "stationary": "output" | "weight" | "input" | "mixed"  (advisory; affects
                          temporal placement defaults but does not change validity)
          "K_spatial_L2": int,        # K factor at L2 spatial (KX). 1 ≤ this ≤ min(pe_dim, K)
                                      # AND must divide K.
          "C_spatial_L1": int,        # C factor at L1 spatial (CX). 1 ≤ this ≤ min(pe_dim, C)
                                      # AND must divide C.
          "K_temporal_L3": int|"auto",  # remaining K (auto = K // K_spatial_L2 // K_temporal_L2)
          "K_temporal_L2": int|"auto",  # additional K temporal at L2 (auto = 1)
          "C_temporal_L2": int|"auto",  # auto = (C // C_spatial_L1) // C_temporal_L1
          "C_temporal_L1": int|"auto",  # auto = 1
          "P_temporal_L3": int|"auto",  # auto = P // (PQ_at_L0[0] * PQ_at_L1[0])
          "Q_temporal_L3": int|"auto",  # auto = Q // (PQ_at_L0[1] * PQ_at_L1[1])
          "PQ_at_L0":     [int, int]|"auto",  # [p,q] at L0[W] (registers).
                                              # auto = [1, 1].
          "PQ_at_L1":     [int, int]|"auto",  # [p,q] temporal at L1[O] (accumulator).
                                              # Output-stationary placement: P,Q tiles
                                              # at the accumulator. Required for layers
                                              # where small-PQ at L1 maximizes output
                                              # reuse (e.g. ResNet stride=2 layers).
                                              # Both must divide (P//p_l0, Q//q_l0)
                                              # respectively. auto = [1, 1].
          "R_temporal_L3": int|"auto",  # auto = R; must equal R total.
          "S_temporal_L3": int|"auto",  # auto = S.
          "N_at_L0":       int|"auto",  # auto = N (typically 1).
        }

    Algorithm:
      1. Validate K_spatial_L2 / C_spatial_L1 (range + divisibility + pe_dim cap).
      2. Resolve "auto" fields so per-dim products match prob[D].
      3. Format the symbolic L3 - L2 - L1 - L0 string.
      4. Run validate_mapping_locally on the result; if it fails, propagate
         the error (intent_to_mapping is then a strict superset of the validator).

    Returns (ok, mapping_str, err). On error, mapping_str is None.
    """
    if not isinstance(hw, dict) or not isinstance(intent, dict) or not isinstance(prob, dict):
        return False, None, "hw, intent, and prob must all be dicts"

    pe_dim = int(hw.get("pe_dim", 16))
    P = int(prob.get("P", 1))
    Q = int(prob.get("Q", 1))
    R = int(prob.get("R", 1))
    S = int(prob.get("S", 1))
    C = int(prob.get("C", 1))
    K = int(prob.get("K", 1))
    N = int(prob.get("N", 1))

    # ---- 1. Spatial factors ----
    try:
        K_sp = int(intent.get("K_spatial_L2", 1))
    except (TypeError, ValueError):
        return False, None, f"K_spatial_L2 must be an integer; got {intent.get('K_spatial_L2')!r}"
    try:
        C_sp = int(intent.get("C_spatial_L1", 1))
    except (TypeError, ValueError):
        return False, None, f"C_spatial_L1 must be an integer; got {intent.get('C_spatial_L1')!r}"

    if K_sp < 1:
        return False, None, f"K_spatial_L2={K_sp} must be ≥ 1"
    if K_sp > pe_dim:
        return False, None, f"K_spatial_L2={K_sp} exceeds pe_dim={pe_dim}"
    if K_sp > K:
        return False, None, f"K_spatial_L2={K_sp} exceeds workload K={K}"
    if K % K_sp != 0:
        return False, None, (
            f"K_spatial_L2={K_sp} does not divide K={K} "
            f"(K % K_spatial_L2 = {K % K_sp}). Pick a divisor of {K} that is "
            f"≤ min(pe_dim={pe_dim}, K={K})."
        )

    if C_sp < 1:
        return False, None, f"C_spatial_L1={C_sp} must be ≥ 1"
    if C_sp > pe_dim:
        return False, None, f"C_spatial_L1={C_sp} exceeds pe_dim={pe_dim}"
    if C_sp > C:
        return False, None, f"C_spatial_L1={C_sp} exceeds workload C={C}"
    if C % C_sp != 0:
        return False, None, (
            f"C_spatial_L1={C_sp} does not divide C={C} "
            f"(C % C_spatial_L1 = {C % C_sp}). Pick a divisor of {C} that is "
            f"≤ min(pe_dim={pe_dim}, C={C})."
        )

    # ---- 2. PQ at L0 ----
    pq_at_l0 = intent.get("PQ_at_L0", "auto")
    if pq_at_l0 == "auto" or pq_at_l0 is None:
        p_l0, q_l0 = 1, 1
    else:
        if not isinstance(pq_at_l0, (list, tuple)) or len(pq_at_l0) != 2:
            return False, None, (
                f"PQ_at_L0 must be 'auto' or a [p,q] pair; got {pq_at_l0!r}"
            )
        try:
            p_l0 = int(pq_at_l0[0])
            q_l0 = int(pq_at_l0[1])
        except (TypeError, ValueError):
            return False, None, f"PQ_at_L0 entries must be integers; got {pq_at_l0!r}"
        if p_l0 < 1 or q_l0 < 1:
            return False, None, f"PQ_at_L0=[{p_l0},{q_l0}] must both be ≥ 1"
        if p_l0 > P or P % p_l0 != 0:
            return False, None, (
                f"PQ_at_L0[0]={p_l0} does not divide P={P}; pick a divisor of P."
            )
        if q_l0 > Q or Q % q_l0 != 0:
            return False, None, (
                f"PQ_at_L0[1]={q_l0} does not divide Q={Q}; pick a divisor of Q."
            )

    # ---- 2b. PQ at L1 (temporal at accumulator → output-stationary placement) ----
    pq_at_l1 = intent.get("PQ_at_L1", "auto")
    if pq_at_l1 == "auto" or pq_at_l1 is None:
        p_l1, q_l1 = 1, 1
    else:
        if not isinstance(pq_at_l1, (list, tuple)) or len(pq_at_l1) != 2:
            return False, None, (
                f"PQ_at_L1 must be 'auto' or a [p,q] pair; got {pq_at_l1!r}"
            )
        try:
            p_l1 = int(pq_at_l1[0])
            q_l1 = int(pq_at_l1[1])
        except (TypeError, ValueError):
            return False, None, f"PQ_at_L1 entries must be integers; got {pq_at_l1!r}"
        if p_l1 < 1 or q_l1 < 1:
            return False, None, f"PQ_at_L1=[{p_l1},{q_l1}] must both be ≥ 1"
        # p_l0 * p_l1 must divide P (residual goes to L3 temporal)
        p_lo_total = p_l0 * p_l1
        if p_lo_total > P or P % p_lo_total != 0:
            return False, None, (
                f"PQ_at_L1[0]={p_l1} times PQ_at_L0[0]={p_l0} = {p_lo_total} "
                f"does not divide P={P}; pick a divisor of P/{p_l0}={P//p_l0}."
            )
        q_lo_total = q_l0 * q_l1
        if q_lo_total > Q or Q % q_lo_total != 0:
            return False, None, (
                f"PQ_at_L1[1]={q_l1} times PQ_at_L0[1]={q_l0} = {q_lo_total} "
                f"does not divide Q={Q}; pick a divisor of Q/{q_l0}={Q//q_l0}."
            )

    # ---- 3. Resolve K_temporal placements ----
    # Default: put residual K at L3 temporal (no extra K at L2 temporal, none at L0).
    K_t_L2_raw = intent.get("K_temporal_L2", "auto")
    K_t_L2 = 1 if (K_t_L2_raw == "auto" or K_t_L2_raw is None) else int(K_t_L2_raw)
    if K_t_L2 < 1:
        return False, None, f"K_temporal_L2={K_t_L2} must be ≥ 1"
    if (K_sp * K_t_L2) > K or K % (K_sp * K_t_L2) != 0:
        return False, None, (
            f"K_spatial_L2 * K_temporal_L2 = {K_sp}*{K_t_L2} = {K_sp*K_t_L2} "
            f"does not divide K={K}; pick K_temporal_L2 such that "
            f"K_spatial_L2 * K_temporal_L2 divides K."
        )
    K_t_L3_raw = intent.get("K_temporal_L3", "auto")
    if K_t_L3_raw == "auto" or K_t_L3_raw is None:
        K_t_L3 = K // (K_sp * K_t_L2)
    else:
        K_t_L3 = int(K_t_L3_raw)
    K_total = K_sp * K_t_L2 * K_t_L3
    if K_total != K:
        return False, None, (
            f"K factor product = K_spatial_L2 * K_temporal_L2 * K_temporal_L3 "
            f"= {K_sp} * {K_t_L2} * {K_t_L3} = {K_total} but workload K = {K}. "
            f"Make these multiply to {K} (e.g. K_temporal_L3 = K // (K_spatial_L2 * "
            f"K_temporal_L2) = {K // (K_sp * K_t_L2)})."
        )

    # ---- 4. Resolve C_temporal placements ----
    C_t_L1_raw = intent.get("C_temporal_L1", "auto")
    C_t_L1 = 1 if (C_t_L1_raw == "auto" or C_t_L1_raw is None) else int(C_t_L1_raw)
    if C_t_L1 < 1:
        return False, None, f"C_temporal_L1={C_t_L1} must be ≥ 1"
    if (C_sp * C_t_L1) > C or C % (C_sp * C_t_L1) != 0:
        return False, None, (
            f"C_spatial_L1 * C_temporal_L1 = {C_sp}*{C_t_L1} = {C_sp*C_t_L1} "
            f"does not divide C={C}; pick C_temporal_L1 such that "
            f"C_spatial_L1 * C_temporal_L1 divides C."
        )
    C_t_L2_raw = intent.get("C_temporal_L2", "auto")
    if C_t_L2_raw == "auto" or C_t_L2_raw is None:
        # default: residual goes entirely to L2 temporal
        C_t_L2 = C // (C_sp * C_t_L1)
    else:
        C_t_L2 = int(C_t_L2_raw)
    C_t_L3_raw = intent.get("C_temporal_L3", "auto")
    if C_t_L3_raw == "auto" or C_t_L3_raw is None:
        denom = C_sp * C_t_L1 * C_t_L2
        if denom == 0 or C % denom != 0:
            return False, None, (
                f"C residual at L3 cannot be auto-resolved: "
                f"C_spatial_L1 * C_temporal_L1 * C_temporal_L2 = {denom} does not "
                f"divide C={C}. Adjust C_temporal_L2."
            )
        C_t_L3 = C // denom
    else:
        C_t_L3 = int(C_t_L3_raw)
    C_total = C_sp * C_t_L1 * C_t_L2 * C_t_L3
    if C_total != C:
        return False, None, (
            f"C factor product = {C_t_L3}*{C_t_L2}*{C_t_L1}*{C_sp} = {C_total} "
            f"but workload C={C}; make C factors multiply to {C}."
        )

    # ---- 5. Resolve P / Q placements (L3 temporal + L1 temporal + L0 temporal) ----
    # P factor flow: P = P_t_L3 * p_l1 * p_l0
    if P % (p_l0 * p_l1) != 0:
        return False, None, (
            f"P_at_L0 * P_at_L1 = {p_l0} * {p_l1} = {p_l0*p_l1} does not divide P={P}"
        )
    if Q % (q_l0 * q_l1) != 0:
        return False, None, (
            f"Q_at_L0 * Q_at_L1 = {q_l0} * {q_l1} = {q_l0*q_l1} does not divide Q={Q}"
        )
    P_t_L3_raw = intent.get("P_temporal_L3", "auto")
    P_t_L3 = (P // (p_l0 * p_l1)) if (P_t_L3_raw == "auto" or P_t_L3_raw is None) else int(P_t_L3_raw)
    Q_t_L3_raw = intent.get("Q_temporal_L3", "auto")
    Q_t_L3 = (Q // (q_l0 * q_l1)) if (Q_t_L3_raw == "auto" or Q_t_L3_raw is None) else int(Q_t_L3_raw)
    if P_t_L3 * p_l1 * p_l0 != P:
        return False, None, (
            f"P factor product = P_t_L3 * P_at_L1 * P_at_L0 = "
            f"{P_t_L3} * {p_l1} * {p_l0} = {P_t_L3 * p_l1 * p_l0} but workload P={P}."
        )
    if Q_t_L3 * q_l1 * q_l0 != Q:
        return False, None, (
            f"Q factor product = Q_t_L3 * Q_at_L1 * Q_at_L0 = "
            f"{Q_t_L3} * {q_l1} * {q_l0} = {Q_t_L3 * q_l1 * q_l0} but workload Q={Q}."
        )

    # ---- 6. Resolve R, S, N — these always go to L3 temporal (or L0 for N=1) ----
    R_t_L3_raw = intent.get("R_temporal_L3", "auto")
    R_t_L3 = R if (R_t_L3_raw == "auto" or R_t_L3_raw is None) else int(R_t_L3_raw)
    if R_t_L3 != R:
        return False, None, (
            f"R factor product mismatch: R_temporal_L3={R_t_L3} but workload R={R}. "
            f"Set R_temporal_L3=R (or 'auto')."
        )
    S_t_L3_raw = intent.get("S_temporal_L3", "auto")
    S_t_L3 = S if (S_t_L3_raw == "auto" or S_t_L3_raw is None) else int(S_t_L3_raw)
    if S_t_L3 != S:
        return False, None, (
            f"S factor product mismatch: S_temporal_L3={S_t_L3} but workload S={S}. "
            f"Set S_temporal_L3=S (or 'auto')."
        )
    N_at_L0_raw = intent.get("N_at_L0", "auto")
    N_at_L0 = N if (N_at_L0_raw == "auto" or N_at_L0_raw is None) else int(N_at_L0_raw)
    if N_at_L0 != N:
        return False, None, (
            f"N at L0 = {N_at_L0} but workload N={N}. Set N_at_L0=N (or 'auto')."
        )

    # ---- 7. Format symbolic mapping ----
    def _fmt(dim: str, val: int, kind: str = "T") -> str:
        if val <= 1:
            return ""  # omit factor 1
        return f"{dim}{val}{'X' if kind == 'X' else ''}"

    # L3 temporal: K_t_L3, P_t_L3, Q_t_L3, R, S
    l3_toks = []
    for d, v in (("K", K_t_L3), ("P", P_t_L3), ("Q", Q_t_L3), ("R", R_t_L3), ("S", S_t_L3)):
        s = _fmt(d, v)
        if s:
            l3_toks.append(s)
    l3_block = "L3[WIO] " + " ".join(l3_toks) if l3_toks else "L3[WIO] N1"

    # L2: C_temporal_L2 (T), K_temporal_L2 (T), K_spatial_L2 (X)
    l2_toks = []
    s = _fmt("C", C_t_L2)
    if s:
        l2_toks.append(s)
    s = _fmt("K", K_t_L2)
    if s:
        l2_toks.append(s)
    if K_sp > 1:
        l2_toks.append(f"K{K_sp}X")
    l2_block = "L2[WI] " + " ".join(l2_toks) if l2_toks else "L2[WI] N1"

    # L1: P_temporal_L1, Q_temporal_L1 (output-stationary tile),
    #     C_temporal_L1 (T), C_spatial_L1 (X)
    l1_toks = []
    s = _fmt("P", p_l1)
    if s:
        l1_toks.append(s)
    s = _fmt("Q", q_l1)
    if s:
        l1_toks.append(s)
    s = _fmt("C", C_t_L1)
    if s:
        l1_toks.append(s)
    if C_sp > 1:
        l1_toks.append(f"C{C_sp}X")
    l1_block = "L1[O] " + " ".join(l1_toks) if l1_toks else "L1[O] N1"

    # L0: P, Q, N (only P/Q can be > 1)
    l0_toks = []
    s = _fmt("P", p_l0)
    if s:
        l0_toks.append(s)
    s = _fmt("Q", q_l0)
    if s:
        l0_toks.append(s)
    # N=1 (or whatever workload says, typically 1) — emit N1 to match action_space examples
    l0_toks.append(f"N{N_at_L0}")
    l0_block = "L0[W] " + " ".join(l0_toks)

    mapping_str = " - ".join((l3_block, l2_block, l1_block, l0_block))

    # ---- 8. Cross-check via validator (defense in depth) ----
    ok, err = validate_mapping_locally(prob, hw, mapping_str)
    if not ok:
        return False, None, f"intent produced invalid mapping ({err}); intent={intent!r}"

    return True, mapping_str, "ok"


def _load_workload_prob(workspace: Path) -> Dict[str, int]:
    """Read workspace/workload.yaml and extract {R,S,P,Q,C,K,N} → int."""
    wl_path = workspace / "workload.yaml"
    if not wl_path.is_file():
        return {}
    try:
        doc = _yaml_mod.safe_load(wl_path.read_text())
    except Exception:
        return {}
    p = doc.get("problem", doc) if isinstance(doc, dict) else {}
    out: Dict[str, int] = {}
    for d in _VALID_DIMS:
        v = p.get(d, 1)
        try:
            out[d] = int(v)
        except (TypeError, ValueError):
            out[d] = 1
    return out


def run_timeloop(
    workspace: Path,
    *,
    hw: Dict[str, int],
    mapping: str,
    rationale: str = "",
    timeout_seconds: int = 180,
) -> TimeloopResult:
    """Write candidate.yaml then call eval.py via the dosa_dse conda env.

    BEFORE invoking eval.py we run validate_mapping_locally() — if the
    mapping is locally rejected, we return TimeloopResult(valid=False,
    raw={"local_pre_check_failed": True, ...}) WITHOUT calling eval.py, so no
    budget is consumed. Callers (Executor agent) can retry the LLM with the
    error message and try again.
    """
    workspace = workspace.resolve()

    # ─── local pre-check (consumes ZERO budget on failure) ───────────
    prob = _load_workload_prob(workspace)
    if prob:
        ok_pre, err_pre = validate_mapping_locally(prob, hw, mapping)
        if not ok_pre:
            return TimeloopResult(
                valid=False,
                error=err_pre,
                raw={
                    "valid": False,
                    "local_pre_check_failed": True,
                    "reason": err_pre,
                    "mapping_str": mapping,
                    # Note: do NOT decrement budget; "calls_used"/"remaining"
                    # are absent so callers preserve env.calls_used unchanged.
                },
            )

    cand = workspace / "candidate.yaml"
    cand.write_text(
        "hw:\n"
        f"  pe_dim: {int(hw.get('pe_dim', 16))}\n"
        f"  sp_size: {int(hw.get('sp_size', 128))}\n"
        f"  acc_size: {int(hw.get('acc_size', 32))}\n"
        f"mapping: {json.dumps(mapping)}\n"
        f"rationale: {json.dumps(rationale)}\n"
    )
    try:
        cmd = _build_eval_command(workspace)
    except RuntimeError as exc:
        return TimeloopResult(valid=False, error=str(exc))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return TimeloopResult(valid=False, error=f"timeout after {timeout_seconds}s")

    out = proc.stdout.strip()
    if not out:
        return TimeloopResult(
            valid=False,
            error=f"empty stdout (rc={proc.returncode}); stderr={proc.stderr[-200:]!r}",
        )
    # eval.py prints exactly one indented JSON object on success/failure,
    # but DOSA's pipeline interleaves [INFO] log lines on stdout. Walk
    # backward from the end, counting braces, to extract the trailing JSON.
    data = _extract_trailing_json(out)
    if data is None:
        return TimeloopResult(valid=False, error=f"no JSON in stdout: {out[-300:]}")

    return TimeloopResult(
        valid=bool(data.get("valid")),
        edp=data.get("edp"),
        cycles=data.get("cycles"),
        energy_uJ=data.get("energy_uJ", data.get("energy_nJ")),
        area=data.get("area"),
        raw=data,
        error=str(data.get("reason", "")) if not data.get("valid") else "",
    )


# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════

class DSEEnv(BaseEnv):
    """A per-task DSE environment.

    Public state read by agents:
      - self.history: chronological list of {hw, mapping, edp, valid, ...}
      - self.advice_buffer: most recent H output, "consumed" when P reads it
      - self.direction_buffer: most recent P output, "consumed" when E reads it
      - self.best_edp / self.best_action / self.best_author
      - self.calls_used / self.max_calls

    Reward (returned by apply()):
      - +1.0 for the first record-break in a submit batch
      - +0.5 for each additional record-break in the same batch
      - 0 otherwise (advice / direction / failed submit / non-record submit)
    Episode ends when calls_used >= max_calls (budget-driven; there is no
    raise-hand / early-stop in the three-role design).
    """

    def __init__(
        self,
        task: DSETask,
        *,
        out_dir: Path,
        reward_config: Optional[RewardConfig] = None,
        max_steps: int = 200,
        preheat_until_session: int = 0,
        baseline_seed_dirs: Optional[List[Path]] = None,
    ):
        super().__init__()
        self.task = task
        self.out_dir = Path(out_dir).resolve()
        self.name = f"dse_{task.workload}_{task.id}"
        self.reward_config = (
            deepcopy(reward_config) if reward_config is not None
            else deepcopy(DEFAULT_HAYEK_CONFIG.reward)
        )
        self.max_steps = max_steps
        # Preheat: sessions with idx < preheat_until_session use only the
        # Executor (no Historian, no Planner) so the cold-start is a pure
        # single-agent search that builds an informed submit-history
        # before Historian/Planner start reasoning. Defaults to 0 = no
        # preheat.
        self.preheat_until_session = preheat_until_session
        # Baseline-hydrate: if supplied, the env will look for
        #   <root>/<task.id>/workspace/history.jsonl  +  best.json
        # in each root (first match wins) and copy them into this run's
        # workspace BEFORE _hydrate_from_disk runs. Effect: the
        # three-role population starts session 0 already knowing all the
        # baseline (single-agent ReAct) submits and their sim results —
        # costs 0 budget, bounds worst-case at baseline quality.
        # Resolve to absolute so a later cwd change cannot break lookup.
        self.baseline_seed_dirs: Optional[List[Path]] = (
            [Path(p).resolve() for p in baseline_seed_dirs] if baseline_seed_dirs else None
        )

        self.workspace = _stage_workspace(self.out_dir, task.workload_yaml)
        # Baseline-seed: if baseline_seed_dirs is configured AND this is
        # the first session AND the workspace is fresh (no history.jsonl
        # yet), seed the workspace from a cached baseline run for this
        # layer. The seed files become the initial state for
        # _hydrate_from_disk(). Sessions 1..N see the accumulated
        # post-seed state via the normal cross-session hydrate.
        if self.baseline_seed_dirs and task.session_idx == 0:
            self._seed_workspace_from_baseline()
        # Per-session: keep workspace's history.jsonl + best.json across
        # session boundaries (so running-best EDP is monotonically improving
        # across the 5 sessions of one layer); only reset budget counter.
        _set_budget_only(self.workspace, task.max_evaluations)

        self.history: List[Dict[str, Any]] = []
        self.advice_buffer: Optional[Dict[str, Any]] = None
        self.direction_buffer: Optional[Dict[str, Any]] = None
        self.advice_history: List[Dict[str, Any]] = []
        self.direction_history: List[Dict[str, Any]] = []
        self.calls_used: int = 0
        self.max_calls: int = task.max_evaluations

        self.best_edp: Optional[float] = None
        self.best_action: Optional[Dict[str, Any]] = None
        self.best_author: Optional[str] = None
        self.raised_hand_by: Optional[str] = None

        self.init_state: str = ""
        self.state: str = ""
        self._last_terminal_score: Optional[float] = None
        self._last_final_reward: float = 0.0
        self._last_extracted_answer: Optional[str] = None

        self.initialize()
        # NOTE: hydration moved INTO initialize() because HayekMAS.train_episode
        # calls env.initialize() again at episode start — calling it here would
        # be wiped. See initialize() end for _hydrate_from_disk().

    def _seed_workspace_from_baseline(self) -> None:
        """Copy a cached single-agent baseline's history.jsonl +
        best.json into this run's workspace, so the three-role
        population starts session 0 with full ReAct prior knowledge.
        First matching root wins. Idempotent: if the workspace already
        has a history.jsonl (e.g. resumed run), do nothing."""
        if not self.baseline_seed_dirs:
            return
        ws_hist = self.workspace / "history.jsonl"
        ws_best = self.workspace / "best.json"
        if ws_hist.is_file() and ws_hist.stat().st_size > 0:
            return  # already populated
        for root in self.baseline_seed_dirs:
            cand_hist = root / self.task.id / "workspace" / "history.jsonl"
            cand_best = root / self.task.id / "workspace" / "best.json"
            if cand_hist.is_file():
                shutil.copy(cand_hist, ws_hist)
                if cand_best.is_file():
                    shutil.copy(cand_best, ws_best)
                n_lines = sum(1 for _ in cand_hist.open())
                # Compact log so workers can see seeding actually fired
                print(
                    f"[baseline-seed] hydrated {n_lines} entries from {root.name}"
                    f"/{self.task.id}/workspace/ → {self.workspace}"
                )
                return
        print(
            f"[baseline-seed] WARN: no baseline found for layer {self.task.id} "
            f"in any of {[str(r) for r in self.baseline_seed_dirs]}"
        )

    def _hydrate_from_disk(self):
        """Pull existing history.jsonl + best.json into env memory so this
        episode (a follow-on session) starts knowing what was already tried.
        Safe no-op on first session (files don't exist yet)."""
        hist_file = self.workspace / "history.jsonl"
        best_file = self.workspace / "best.json"
        if hist_file.is_file():
            for line in hist_file.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                r = rec.get("result", {})
                self.history.append({
                    "step": 0,                       # historical, no step info
                    "author": "(prior_session)",
                    "hw": r.get("hw_config"),
                    "mapping": r.get("mapping_str"),
                    "edp": r.get("edp"),
                    "valid": r.get("valid"),
                    "raw": rec,
                })
        if best_file.is_file():
            try:
                b = json.loads(best_file.read_text())
                if b.get("valid") and b.get("edp") is not None:
                    self.best_edp = float(b["edp"])
                    self.best_action = {"hw": b.get("hw_config"),
                                        "mapping": b.get("mapping_str")}
                    self.best_author = "(prior_session)"
            except Exception:
                pass
        # Fallback for missing/malformed best.json: some cached
        # baselines have an older schema or store best.json as plaintext.
        # In that case, recompute best_edp from the just-hydrated
        # history. Without this, those layers would behave like
        # cold-start (best_edp=None → first valid submit gets a trivial
        # record-break) while other layers have a competitive bar.
        if self.best_edp is None and self.history:
            best_idx = -1
            best_edp = None
            for i, h in enumerate(self.history):
                if h.get("valid") and h.get("edp") is not None and h["edp"] > 0:
                    if best_edp is None or h["edp"] < best_edp:
                        best_edp = h["edp"]
                        best_idx = i
            if best_edp is not None:
                self.best_edp = float(best_edp)
                h = self.history[best_idx]
                self.best_action = {"hw": h.get("hw"),
                                    "mapping": h.get("mapping")}
                self.best_author = "(prior_session)"

    # ─── BaseEnv contract ───────────────────────────────────────────────────

    def initialize(self):
        self.step_count = 0
        self.terminated = False
        self.action_history = {}
        self.history = []
        self.advice_buffer = None
        self.direction_buffer = None
        self.advice_history = []
        self.direction_history = []
        self.calls_used = 0
        self.best_edp = None
        self.best_action = None
        self.best_author = None
        self.raised_hand_by = None
        self._last_terminal_score = None
        self._last_final_reward = 0.0
        self.init_state = self._build_init_state()
        self.state = self.init_state
        # CRITICAL: hydrate at end of initialize() (NOT just in __init__),
        # because HayekMAS.train_episode calls env.initialize() again at
        # episode start. Without re-hydrating here, prior-session history/best
        # are wiped before agents see them.
        self._hydrate_from_disk()

    def _build_init_state(self) -> str:
        wl = self.task.workload_yaml.read_text() if self.task.workload_yaml.is_file() else ""
        return "\n".join([
            f"## Task: {self.task.workload}/{self.task.id}",
            f"  budget: {self.max_calls} Timeloop submissions",
            "",
            "## Layer yaml",
            wl,
        ])

    def get_state_description(self) -> str:
        best = f"best_edp={self.best_edp:.3e}" if self.best_edp is not None else "no_best"
        return (
            f"DSEEnv({self.name}, calls={self.calls_used}/{self.max_calls}, "
            f"step={self.step_count}, {best}, "
            f"{'terminated' if self.terminated else 'running'})"
        )

    def get_terminal_score(self) -> Optional[float]:
        return self._last_terminal_score

    def get_task_description(self) -> str:
        return self.task.description or self.task.id

    def get_correct_answer(self) -> str:
        return f"Lower EDP is better. Best so far: {self.best_edp}"

    def get_final_answer_author(self) -> Optional[str]:
        return self.raised_hand_by or self.best_author

    def get_last_final_reward(self) -> float:
        return self._last_final_reward

    def is_successful(self) -> Optional[bool]:
        if self.best_edp is None:
            return None
        return True

    def build_episode_metrics(self) -> dict:
        return {
            "task_id": self.task.id,
            "workload": self.task.workload,
            "calls_used": self.calls_used,
            "max_calls": self.max_calls,
            "best_edp": self.best_edp,
            "best_action": self.best_action,
            "best_author": self.best_author,
            "raised_hand_by": self.raised_hand_by,
            "history_size": len(self.history),
            "advice_count": len(self.advice_history),
            "direction_count": len(self.direction_history),
        }

    # ─── apply ──────────────────────────────────────────────────────────────

    def apply(self, action: BaseAction) -> float:
        if self.terminated:
            return 0.0
        if not isinstance(action, DSEAction):
            raise TypeError("DSEEnv expects DSEAction")
        self.step_count += 1
        author = action.author or ""
        role = action.role or ""

        if action.kind == DSEAction.KIND_ADVICE:
            return self._handle_advice(action, author, role)
        if action.kind == DSEAction.KIND_DIRECTION:
            return self._handle_direction(action, author, role)
        if action.kind == DSEAction.KIND_SUBMIT:
            return self._handle_submit(action, author, role)
        self.action_history[self.step_count] = {
            "author": author, "role": role, "kind": action.kind,
            "text": action.text, "is_submission": False,
        }
        return 0.0

    # ─── kind-specific handlers ────────────────────────────────────────────

    def _handle_advice(self, action: DSEAction, author: str, role: str) -> float:
        record = {
            "step": self.step_count, "author": author,
            "calls_used_at_emit": self.calls_used,
            "history_size_at_emit": len(self.history),
            "text": action.text,
        }
        self.advice_buffer = record
        self.advice_history.append(record)
        self.action_history[self.step_count] = {
            "author": author, "role": role, "kind": "advice",
            "text": action.text, "is_submission": False,
        }
        self.state += f"\n\n[Step {self.step_count}] [Historian {author}]\n{action.text[:600]}"
        return 0.0

    def _handle_direction(self, action: DSEAction, author: str, role: str) -> float:
        record = {
            "step": self.step_count, "author": author,
            "chose_historian": action.chose_historian,
            "advice_consumed": deepcopy(self.advice_buffer),
            "calls_used_at_emit": self.calls_used,
            "text": action.text,
        }
        self.advice_buffer = None
        self.direction_buffer = record
        self.direction_history.append(record)
        self.action_history[self.step_count] = {
            "author": author, "role": role, "kind": "direction",
            "text": action.text, "is_submission": False,
            "chose_historian": action.chose_historian,
        }
        listened_to = action.chose_historian or "n/a"
        self.state += (
            f"\n\n[Step {self.step_count}] [Planner {author} → listened to "
            f"Historian {listened_to}]\n{action.text[:600]}"
        )
        return 0.0

    def _handle_submit(self, action: DSEAction, author: str, role: str) -> float:
        consumed_direction = self.direction_buffer
        self.direction_buffer = None

        any_record_break = False
        record_break_count = 0
        for sub in action.submissions:
            edp = sub.get("edp")
            valid = sub.get("valid", False)
            raw = sub.get("raw", {}) or {}
            # locally-rejected candidates DO NOT consume budget.
            # If the submission carries `calls_used_after`, trust it; otherwise
            # fall back to the legacy heuristic. A `local_pre_check_failed` raw
            # dict means we never called eval.py, so `self.calls_used` stays put.
            local_rejected = bool(raw.get("local_pre_check_failed"))
            if "calls_used_after" in sub:
                self.calls_used = int(sub["calls_used_after"])
            elif local_rejected:
                # No call was charged.
                pass
            else:
                self.calls_used += 1 if (valid or "reason" in raw) else 0
            self.history.append({
                "step": self.step_count,
                "author": author,
                "hw": sub.get("hw"),
                "mapping": sub.get("mapping"),
                "edp": edp,
                "cycles": sub.get("cycles"),
                "energy_uJ": sub.get("energy_uJ"),
                "valid": valid,
                "local_pre_check_failed": local_rejected,
                "raw": raw,
            })
            if valid and edp is not None:
                if self.best_edp is None or edp < self.best_edp:
                    self.best_edp = float(edp)
                    self.best_action = {
                        "hw": sub.get("hw"),
                        "mapping": sub.get("mapping"),
                    }
                    self.best_author = author
                    any_record_break = True
                    record_break_count += 1

        self.action_history[self.step_count] = {
            "author": author, "role": role, "kind": "submit",
            "text": action.text,
            "is_submission": True,
            "n_submits": len(action.submissions),
            "record_breaks": record_break_count,
            "best_edp_after": self.best_edp,
            "consumed_direction": (consumed_direction or {}).get("author"),
        }
        if action.submissions:
            edps = [s.get("edp") for s in action.submissions if s.get("edp") is not None]
            edp_summary = (
                f"min={min(edps):.3e}, max={max(edps):.3e}" if edps else "all-fail"
            )
            self.state += (
                f"\n\n[Step {self.step_count}] [Executor {author} submitted "
                f"{len(action.submissions)} action(s); {edp_summary}; "
                f"{record_break_count} new record(s); best now {self.best_edp})]"
            )

        if any_record_break:
            reward = 1.0 + 0.5 * (record_break_count - 1)
        else:
            reward = 0.0

        if self.calls_used >= self.max_calls:
            self._terminate(reason="budget_exhausted")
        return reward

    def _terminate(self, *, reason: str):
        self.terminated = True
        # Bounded terminal score: do NOT propagate raw EDP (huge ~1e11) into
        # path_reward — it makes every agent receive ~-1e10 reward and crashes
        # the wealth dynamics. Stepwise +1.0 record-break reward already
        # encodes per-step contribution; terminal score adds nothing useful.
        self._last_terminal_score = 0.0
        self._last_final_reward = 0.0
        self._log(
            f"\n🏁 DSEEnv terminated ({reason}); best_edp="
            f"{self.best_edp} by {self.best_author}, calls_used="
            f"{self.calls_used}/{self.max_calls}",
            indent=2,
        )

    # ─── prompt rendering helpers (used by agents) ──────────────────────────

    def render_history_for_agent(self, max_entries: int = 30) -> str:
        if not self.history:
            return "(no Timeloop submissions yet)"
        recent = self.history[-max_entries:]
        lines = []
        for h in recent:
            edp_s = f"{h['edp']:.3e}" if h.get("edp") is not None else "INVALID"
            best_marker = " ★" if (h.get("edp") == self.best_edp) else ""
            hw = h.get("hw") or {}
            mp = (h.get("mapping") or "")[:120]
            lines.append(
                f"- step{h['step']} [{h['author']}] hw={hw} mapping={mp!r} "
                f"→ EDP={edp_s}{best_marker}"
            )
        if len(self.history) > max_entries:
            lines.insert(0, f"(showing last {max_entries} of {len(self.history)} entries)")
        return "\n".join(lines)

    def render_advice_buffer(self) -> str:
        if not self.advice_buffer:
            return "(no fresh advice)"
        a = self.advice_buffer
        return f"[from Historian {a['author']} (step {a['step']})]\n{a['text']}"

    @property
    def in_preheat(self) -> bool:
        """True if the current session is a preheat (single-agent E only)
        session. Historians and Planners do NOT wake during preheat —
        the Executor builds the submit history alone."""
        return self.task.session_idx < self.preheat_until_session

    def render_direction_buffer(self) -> str:
        if self.in_preheat:
            return (
                "[PREHEAT MODE — session "
                f"{self.task.session_idx + 1}/{self.preheat_until_session}]\n"
                "No Historian or Planner this session — you are the lead.\n"
                "Propose mappings yourself based on the submission history;\n"
                "your goal is to discover good (hw, mapping) pairs that will\n"
                "anchor the Hayek H/P/E team in later sessions."
            )
        if not self.direction_buffer:
            return "(no fresh direction)"
        d = self.direction_buffer
        listened = d.get("chose_historian") or "n/a"
        return (
            f"[from Planner {d['author']} (step {d['step']}; listened to {listened})]\n"
            f"{d['text']}"
        )

    def render_task_block(self) -> str:
        return self.init_state

    def budget_remaining(self) -> int:
        return max(0, self.max_calls - self.calls_used)

    def submits_since_last_break(self) -> int:
        """Count valid submits since the most recent record-break (best_edp
        update). If no record yet, returns count of all valid submits."""
        if self.best_edp is None:
            return sum(1 for h in self.history if h.get("valid"))
        cnt = 0
        for h in reversed(self.history):
            if not h.get("valid"):
                continue
            if h.get("edp") == self.best_edp:
                break
            cnt += 1
        return cnt
