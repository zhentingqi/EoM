#!/usr/bin/env python3.11
"""Build priors_per_layer.json from per-layer evidence + a small archetype set.

This is a ONE-SHOT generator — its output is the FIXED table committed as
priors_per_layer.json. The runtime never re-runs this script; it just
reads the JSON. The mapping (which archetype goes to which layer slot)
is encoded explicitly in LAYER_ASSIGNMENT below — there is NO online
computation.

Glossary for the archetype names below (Gemmini / Timeloop terminology):
  - K, C, P, Q       : output-channel, input-channel, output height, output width
  - pe_dim           : square root of the PE-array dimension (so pe_dim=128
                       means a 128×128 systolic array)
  - L0[W], L1[O], L2 : memory-hierarchy levels. L0 is closest to the PEs and
                       typically holds the weight matrix; L1 holds the
                       output tile; L2 is the global buffer.
  - K-spatial / C-spatial : whether the K (or C) dimension is mapped
                       spatially (across the PE array) at a given level.
  - stride=2         : standard conv-stride-2 layers (downsampling).

The six archetypes:
  A_K128       — pe_dim=128, K mapped spatially at L2 (dominates layers with many output channels)
  A_L1_PQ128   — output-stationary at L1[O] + pe_dim=128 (canonical for stride=2 / small P,Q)
  A_L0_PQ32    — weight-stationary at L0[W] + pe_dim=32 (small-P,Q, low-PE alternative)
  A_C_spatial  — pe_dim=64-128, C mapped spatially at L1 (mid-size feature maps)
  A_pe16_sweep — pe_dim=16, sweep small (for tiny-C layers like the first conv)
  A_refine     — refine around the current best (always relevant late in a session)

Per-layer LAYER_ASSIGNMENT picks 3 strategies per role based on layer dims:
  - large K (K ≥ 256, P,Q ≥ 14): K128, C_spatial, refine
  - small P,Q (P,Q ≤ 14, large K): L1_PQ128, K128, refine
  - small P,Q + small K: L1_PQ128, L0_PQ32, refine
  - stride=2 mid P,Q: L1_PQ128, K128, refine
  - tiny C (first conv): pe16_sweep, K128, refine
  - special P=Q=1: pe16_sweep, K128, refine

Run:  python3.11 scripts/arch_dse_world/build_priors.py
Output:  hayekmas/adapters/arch_dse_world/configs/priors_per_layer.json
"""
import json
from pathlib import Path

# ---------------------------------------------------------------- ARCHETYPES
ARCH = {
    "A_K128": {
        "historian": (
            "EXPERIENCE PRIOR — large-K bias. This layer has many output "
            "channels (K ≥ 256). The dominant axis is K-spatial at L2. ALWAYS "
            "advocate pe_dim=128 with K_spatial_L2=128 (or =pe_dim) to maximize "
            "PE utilization. If recent submits use pe_dim ≤ 32, push the Planner "
            "to scale up."
        ),
        "planner": (
            "EXPERIENCE PRIOR — K-spatial direction. Direct the Executor to try "
            "pe_dim ∈ {64, 128} with K_spatial_L2 = pe_dim and small "
            "C_spatial_L1 (1-8). Keep R/S at L3 temporal. This is the canonical "
            "weight-stationary-at-L2 layout for large-K convs."
        ),
        "executor": (
            "EXPERIENCE PRIOR — K-spatial executor. Default to pe_dim=128 "
            "and submit candidates spanning K_spatial_L2 ∈ {64, 128} with "
            "C_spatial_L1 ∈ {1, 8, 16}. Use intent schema. Batch 3-4 candidates "
            "per turn so the next turn has comparison data."
        ),
    },
    "A_L1_PQ128": {
        "historian": (
            "EXPERIENCE PRIOR — output-stationary at L1[O]. This layer's spatial "
            "dimensions (P, Q) are small enough to fit a full tile in the L1 "
            "accumulator. Push the Planner toward `L1[O] P<P> Q<Q> C<n>X` — "
            "the canonical output-stationary placement. Validated empirically "
            "to give 10-100× EDP improvement over K-tiling on stride=2 / "
            "small-PQ layers."
        ),
        "planner": (
            "EXPERIENCE PRIOR — full-PQ-at-L1 direction. Direct the Executor "
            "to try pe_dim=128 with PQ_at_L1 = [P, Q] (entire tile at "
            "accumulator), K_spatial_L2 = 1, C_spatial_L1 = pe_dim. Both this "
            "and the K-spatial alternative should be tested in the same session."
        ),
        "executor": (
            "EXPERIENCE PRIOR — L1[O] PQ executor. Use intent schema with "
            "PQ_at_L1=[P, Q] (or use Schema B raw mapping `L1[O] P<n> Q<m> "
            "C<k>X`). Default pe_dim=128. Batch 3-4 candidates varying "
            "C_spatial_L1 ∈ {64, 128} and K_temporal_L3."
        ),
    },
    "A_L0_PQ32": {
        "historian": (
            "EXPERIENCE PRIOR — weight-stationary at L0[W] (PQ at registers). "
            "Small-PQ layers benefit from placing the (P, Q) tile at L0[W] so "
            "weights stream once and stay register-stationary while outputs "
            "accumulate. This placement is ORTHOGONAL to pe_dim choice — try "
            "the L0 placement at pe_dim ∈ {32, 64, 128}, picking pe_dim "
            "to match the layer's K and C (large K → pe=128, tiny K → pe=32). "
            "Empirically wins on layers where P×Q ≤ ~128 and stride=2."
        ),
        "planner": (
            "EXPERIENCE PRIOR — register-stationary direction. Direct Executor "
            "to PQ_at_L0=[P, Q] with pe_dim spanning {32, 64, 128} so the "
            "auction can compare them. Pair with K_spatial_L2 = pe_dim and "
            "small C_spatial_L1. Compare directly against the L1[O] regime."
        ),
        "executor": (
            "EXPERIENCE PRIOR — L0[W] PQ executor. Use intent schema with "
            "PQ_at_L0=[P, Q]. Submit 3-4 candidates per turn varying pe_dim "
            "∈ {32, 64, 128} and K_spatial_L2 to find the right balance for "
            "this layer's K/C ratio. Validate locally."
        ),
    },
    "A_C_spatial": {
        "historian": (
            "EXPERIENCE PRIOR — C-spatial bias. Layers with large input "
            "channels (C ≥ 128) benefit from C-spatial at L1[O] = pe_dim. "
            "Push the Planner toward pe_dim=64-128 with C_spatial_L1=pe_dim "
            "and small-to-medium K_spatial_L2."
        ),
        "planner": (
            "EXPERIENCE PRIOR — C-spatial direction. Direct Executor to "
            "pe_dim ∈ {64, 128}, C_spatial_L1 = pe_dim, K_spatial_L2 ∈ {1, 8, 32}. "
            "Optionally add PQ_at_L1 for output-stationary at the accumulator."
        ),
        "executor": (
            "EXPERIENCE PRIOR — C-spatial executor. Default pe_dim=64-128. "
            "Sweep K_spatial_L2 ∈ {1, 8, 32, 64} with fixed C_spatial_L1=128. "
            "Submit 4 candidates per turn."
        ),
    },
    "A_pe16_sweep": {
        "historian": (
            "EXPERIENCE PRIOR — small-pe regime. This layer has tiny C or "
            "tiny K; large pe_dim wastes PEs. Push the Planner toward "
            "pe_dim=16 with full sweep over K and C tiling."
        ),
        "planner": (
            "EXPERIENCE PRIOR — small-pe direction. pe_dim=16. Sweep "
            "K_spatial_L2 ∈ {1, 4, 8, 16} and C_spatial_L1 ∈ {1, 3} (since "
            "C is tiny). Vary PQ_at_L0/L1 across submits."
        ),
        "executor": (
            "EXPERIENCE PRIOR — small-pe executor. pe_dim=16. Submit 4-5 "
            "candidates per turn over K_spatial × PQ placements."
        ),
    },
    "A_refine": {
        "historian": (
            "EXPERIENCE PRIOR — refine-around-best. After a few submits, focus "
            "on small perturbations of the current best mapping. Vary one HW "
            "knob (sp_size, acc_size) at a time. Don't restart exploration."
        ),
        "planner": (
            "EXPERIENCE PRIOR — refinement direction. Take the best mapping "
            "the Executor found and propose 2-3 perturbations: ±1 step in "
            "K_temporal_L2, ±1 step in C_temporal_L2, swap PQ_at_L0 vs L1."
        ),
        "executor": (
            "EXPERIENCE PRIOR — refinement executor. Take the best mapping "
            "from history and submit 2-3 small variants. Validate each locally "
            "before consuming budget."
        ),
    },
}

# ---------------------------------------------------------------- LAYER MAP
# Per-layer assignment: 3 archetypes per role for the 3 starter agents in role.
# Order matters — agent index 0 gets the first, etc. The auction will let the
# best-fit archetype survive. Other 17 layers (no aligned baseline data) use
# dimensional rules; the 7 with baseline evidence have refinements.
LAYER_ASSIGNMENT = {
    # ----- 7 layers we have aligned-baseline best mappings for -----
    "_outputs_input.8":    ["A_C_spatial", "A_K128", "A_refine"],     # b: pe=64, K64X+C64X — C_spatial captures
    "_outputs_input.11":   ["A_K128",      "A_C_spatial", "A_refine"], # b: pe=128, K128X — direct match
    "_outputs_input.36":   ["A_L1_PQ128",  "A_K128", "A_refine"],     # b: pe=128, L1[O] P28 Q28 — direct match
    "_outputs_input.73":   ["A_L1_PQ128",  "A_L0_PQ32", "A_refine"],  # b: pe=32, L0[W] P7 Q7 — A_L0_PQ32 critical
    # FIX: baseline placed PQ at L0[W] not L1[O]; replace L1_PQ128 with L0_PQ32
    "_outputs_input.76":   ["A_K128",      "A_L0_PQ32", "A_refine"],  # b: pe=128, K128X + L0[W] P14 Q2
    "_outputs_input.80":   ["A_L1_PQ128",  "A_K128", "A_refine"],     # b: L1[O] P14 Q14 C128X — direct match
    "_outputs_input.131":  ["A_L1_PQ128",  "A_K128", "A_refine"],     # b: pe=128, L1[O] P7 Q7 C128X — direct match

    # ----- other 17 layers: dim-rule priors (P,Q,C,K,stride from layer_evidence.json) -----
    # The L0_PQ32 prior says "PQ_at_L0=[P,Q]" — for P=Q=112 that's
    # impossible (L0 holds ~32 cells max). Empirically the Executor got
    # rejected mappings, gave up on L0 placement, and regressed to "all PQ
    # at L3" (1.89e11) vs a 3.31e8 winner with L0[W] P4 Q112 partial-PQ.
    # So replace L0_PQ32 with C_spatial — C=3 makes C-spatial=3 a natural
    # choice.
    "_outputs_input.2":    ["A_pe16_sweep","A_C_spatial", "A_refine"], # P=Q=112, C=3 (tiny), K=64 — first conv
    # FIX: K=64 not large; L1_PQ128 is more honest competitor
    "_outputs_input.5":    ["A_L1_PQ128",  "A_C_spatial", "A_refine"], # P=Q=56, C=K=64
    # FIX: K=64; substitute L1_PQ128 for K128
    "_outputs_input.15":   ["A_L1_PQ128",  "A_C_spatial", "A_refine"], # P=Q=56, C=256, K=64
    "_outputs_input.33":   ["A_K128",      "A_C_spatial", "A_refine"], # P=Q=56, C=256, K=128 — PASS
    "_outputs_input.39":   ["A_K128",      "A_C_spatial", "A_refine"], # P=Q=28, C=128, K=512 — PASS
    # FIX: P=Q=28 too large for full PQ-at-L1; C=256 dominant → C_spatial primary
    "_outputs_input.40":   ["A_C_spatial", "A_K128", "A_refine"],     # P=Q=28, stride=2
    "_outputs_input.43":   ["A_K128",      "A_C_spatial", "A_refine"], # P=Q=28, C=512, K=128 — PASS
    # FIX: 3×3 + moderate PQ — add L1_PQ128 competitor (parallels .36 winner)
    "_outputs_input.46":   ["A_C_spatial", "A_L1_PQ128", "A_refine"],  # P=Q=28, C=K=128
    "_outputs_input.70":   ["A_K128",      "A_C_spatial", "A_refine"], # P=Q=28, C=512, K=256 — PASS
    "_outputs_input.77":   ["A_L1_PQ128",  "A_K128", "A_refine"],     # P=Q=14, stride=2 — PASS
    # FIX: K=256 large; drop refine, add K128 competitor
    "_outputs_input.83":   ["A_L1_PQ128",  "A_L0_PQ32", "A_K128"],    # P=Q=14, C=K=256 — 3×3
    # FIX: C=1024 (huge); A_C_spatial deserves slot
    "_outputs_input.125":  ["A_L1_PQ128",  "A_C_spatial", "A_refine"], # P=Q=14, C=1024, K=512
    # FIX: tiny PQ=7 — register-stationary L0 is natural alternative to L1
    "_outputs_input.128":  ["A_L1_PQ128",  "A_L0_PQ32", "A_refine"],  # P=Q=7, stride=2, C=K=512 — 3×3
    "_outputs_input.132":  ["A_L1_PQ128",  "A_K128", "A_refine"],     # P=Q=7, stride=2, C=1024, K=2048 — PASS
    # FIX: C=2048 (max in network); C_spatial more relevant than another K128
    "_outputs_input.135":  ["A_L1_PQ128",  "A_C_spatial", "A_refine"], # P=Q=7, C=2048, K=512
    # FIX: tiny PQ=7 + 3×3 — register-stationary L0 deserves slot
    "_outputs_input.138":  ["A_L1_PQ128",  "A_L0_PQ32", "A_refine"],  # P=Q=7, C=K=512
    # CRITICAL FIX: FC-as-conv has C=2048, K=1000 (huge); pe16_sweep wrong (was for tiny C/K).
    # Right strategy: pe=128 with K-spatial + C-spatial.
    "_outputs_3356":       ["A_K128",      "A_C_spatial", "A_refine"], # P=Q=1, C=2048, K=1000
}


def build():
    """Materialize per-layer priors from archetypes."""
    out = {}
    for layer, arch_names in LAYER_ASSIGNMENT.items():
        layer_priors = {"historian": [], "planner": [], "executor": []}
        for arch_name in arch_names:
            arch = ARCH[arch_name]
            for role in ("historian", "planner", "executor"):
                layer_priors[role].append(arch[role])
        out[layer] = layer_priors

    # Sanity: every layer has 3 priors per role
    for L, p in out.items():
        for role in ("historian", "planner", "executor"):
            assert len(p[role]) == 3, f"{L}/{role} has {len(p[role])} priors"

    return out


if __name__ == "__main__":
    data = build()
    dst = Path("hayekmas/adapters/arch_dse_world/configs/priors_per_layer.json")
    dst.write_text(json.dumps({
        "_meta": {
            "generator": "scripts/arch_dse_world/build_priors.py",
            "archetypes": list(ARCH.keys()),
            "n_layers": len(data),
            "fixed_table": True,
            "doc": "DO NOT edit by hand. Re-run build_priors.py to regenerate.",
        },
        "layers": data,
    }, indent=2))
    print(f"wrote {dst}: {len(data)} layers, each with 3 priors per role")
