# Hardware + Mapping Design Task (Gemmini / Timeloop)

You are a hardware architect. Your task is to **jointly** design:
1. The Gemmini HW configuration — scratchpad and accumulator sizes
2. The dataflow mapping — how the computation is tiled across the memory hierarchy

for a specific neural network layer. You minimize EDP (energy × cycles).

## Architecture

Gemmini has a 4-level memory hierarchy:

| Level | Name | Holds | Abbrev |
|-------|------|-------|--------|
| L3 | DRAM | weights + inputs + outputs | WIO |
| L2 | Scratchpad (SRAM, shared) | weights + inputs | WI |
| L1 | Accumulator (SRAM, per-column) | outputs | O |
| L0 | Registers (per-PE) | weights | W |

The PE array is `pe_dim × pe_dim` (you choose pe_dim ∈ [2, 128] — total PEs = pe_dim²,
e.g., pe_dim=16 → 256 PEs). Only these spatial parallelisms are valid in the
Gemmini Timeloop pipeline (any other will be silently zeroed):
- **L2 (Scratchpad)**: K can be parallelized across PEs → `KX` (≤ pe_dim)
- **L1 (Accumulator)**: C can be parallelized across PEs → `CX` (≤ pe_dim)

**Critical**: the spatial dimension and the level it lives at are coupled. Putting `CX` at
L2 or `KX` at L1 will be rejected as a malformed mapping (the eval cross-checks Timeloop's
total Computes count against the workload size).

## HW search space (you choose each submission)

- `pe_dim`: PE array side-length, range **[2, 128]** (powers of 2: 2,4,8,16,32,64,128)
- `sp_size`: scratchpad size in KB, range **[1, 2048]** (powers of 2 recommended)
- `acc_size`: accumulator size in KB, range **[1, 2048]** (powers of 2 recommended)

Total PEs = `pe_dim²`. e.g., `pe_dim=16` gives 16×16=256 PE; `pe_dim=32` gives 1024 PE.
Larger pe_dim → more parallelism (lower cycles) but more area + per-PE energy.

⚠️ **DO NOT default to pe_dim=16**. Larger pe_dim often dramatically reduces cycles at
modest energy increase — for many ResNet50 layers, pe_dim=32 (1024 PEs) is 4× lower EDP
than pe_dim=16. **You should sweep at least {8, 16, 32, 64} early in your search.**
Don't declare convergence until you've tested multiple pe_dim values.

Larger SRAM → lower DRAM traffic (lower energy) but larger area. Smaller → spills to
DRAM (high energy). The sweet spot depends on the layer size.

Hardware spec is in `hardware.yaml`. Layer dimensions are in `workload.yaml`.

## Your design format

You express the mapping as a symbolic string, one space-separated block per memory level, joined by ` - `:

```
L3[WIO] <factors> - L2[WI] <factors> - L1[O] <factors> - L0[W] <factors>
```

Where each factor is `<DIM><VALUE>` (temporal) or `<DIM><VALUE>X` (spatial along X).

### Dimensions (CNN layer with 7 dims)
| Dim | Meaning |
|-----|---------|
| R | filter height |
| S | filter width |
| P | output height |
| Q | output width |
| C | input channels |
| K | output channels |
| N | batch |

### Examples

Output-stationary K-parallel mapping for a 1×1 conv (R=S=1) with C=1024, K=1024, P=1024:
```
L3[WIO] C4 K16 P8 - L2[WI] C4X - L1[O] K4 P16 K4X - L0[W] C64
```
Reading: DRAM tiles outer (C=4, K=16, P=8), scratchpad tiles 4 in C and spatial-parallelizes
C across X PEs, accumulator tiles K=4, P=16, spatial-parallelizes K across X PEs, registers
hold C=64 reduction.

### Factorization rule

For every dimension D with size `prob[D]`, the product of **all factors in D across all
4 levels** (temporal + spatial) must equal `prob[D]`. Otherwise Timeloop rejects the mapping.

Dimensions with size 1 (e.g., R=S=1 for 1×1 conv) should just have `R1` or be omitted at every
level — most cleanly: omit them.

## Objective

**Minimize Energy-Delay Product (EDP = cycles × μJ).** Lower is better.

## Design workflow

1. Read `workload.yaml` — get dimensions K, C, P, Q, R, S, N
2. Read `hardware.yaml` — get PE dim, scratchpad size, accumulator size
3. Factor every dimension into a product across L3/L2/L1/L0, respecting the factorization rule
4. Write your design to `candidate.yaml`:
   ```yaml
   hw:
     pe_dim: 16        # PE array side, [2..128]
     sp_size: 128      # KB, [1..2048]
     acc_size: 32      # KB, [1..2048]
   mapping: "L3[WIO] ... - L2[WI] ... - L1[O] ... - L0[W] ..."
   rationale: "explain both your HW choice and your mapping reasoning"
   ```
5. Submit: `./submit.sh`
   Response:
   ```json
   {"valid": true, "cycles": ..., "energy_uJ": ..., "edp": ...,
    "hw_config": {"pe_dim": 16, "sp_size": ..., "acc_size": ...},
    "calls_used": N, "remaining": ...}
   ```
6. Analyze, update `notes.md`, iterate.

## Constraints

- You have a **limited number of evaluation runs** — see `budget.json`. Each `./submit.sh` costs
  one eval call, even if the design is invalid. Plan carefully.
- Factors must multiply to the dimension. `prob[C] = 1024` → `C_L3 × C_L2 × C_L1 × C_L0 = 1024`.
- Spatial factors on CX (L2) and KX (L1) are limited by your chosen `pe_dim`
  (the spatial product at each level must be ≤ `pe_dim`).
- **Buffer capacity matters**: your mapping's tile sizes must fit in the HW you pick.
  Larger SP/ACC gives you more mapping flexibility but costs area. Too-small SP/ACC with
  a large-tile mapping causes DRAM spilling (bad energy).

## Design notebook (`notes.md`)

After **every** submission, record:
- What you tried and why
- What the result tells you
- What you plan to try next

This notebook is your primary engineering record.

## Files

| File | Purpose |
|---|---|
| `README.md` | This brief |
| `action_space.md` | Full spec of the mapping string format |
| `workload.yaml` | The layer dimensions |
| `hardware.yaml` | Gemmini configuration |
| `candidate.yaml` | Your current design — **edit this** |
| `submit.sh` | Submit for evaluation — **run this** |
| `notes.md` | Your notebook — **write here after each eval** |
| `history.jsonl` | Auto-logged evaluation history |
| `best.json` | Auto-updated best design |
| `budget.json` | Remaining budget |
| `refine.sh` | *(if present)* Local-search refinement tool — see below |

## Optional tool: `refine.sh`

If a `refine.sh` is present in this directory, you have access to a local-search
refinement subroutine. It takes your current best mapping, generates ~50 perturbations
(swap loop orders, nudge tile sizes to adjacent divisors), evaluates each via Timeloop,
and returns the best found.

**Usage**:
```bash
./refine.sh --start-mapping "L3[WIO] ... - L2[WI] ... - L1[O] ... - L0[W] ..." --n-samples 50
```
Returns JSON:
```json
{"best_mapping": "...", "best_edp": ..., "start_edp": ...,
 "improvement_ratio": ..., "n_evaluated": ..., "n_valid": ..., "duration_s": ...}
```

**Budget**: As of 2026-04-25, `refine.sh` **DOES** consume your evaluation budget —
each internal Timeloop evaluation it runs (typically up to 50) charges 1 unit
against `budget.json`, the same as `./submit.sh`. This was changed after a fairness
audit found that the previous "free" model gave the agent unfair compute advantage.

Use refine.sh sparingly. Each call may consume up to N+1 budget (N samples + 1
seed eval). Monitor remaining budget via `python3 eval.py --check-budget`.

**When to use**: after you've found a good mapping with `./submit.sh`, call
`refine.sh` with that mapping to see if small local tweaks help. If it returns
`improvement_ratio > 1`, submit the new best mapping via the normal
`candidate.yaml` + `./submit.sh` flow to officially log it in `best.json`.

## Getting started

1. Read `action_space.md` for the mapping string grammar.
2. Read `workload.yaml` for the layer dimensions.
3. Read `hardware.yaml` for the chip.
4. Pick a dataflow strategy (weight-stationary, output-stationary, row-stationary, etc.)
   that suits this layer on this chip.
5. Factor each dim and write your mapping.
6. Submit, observe, iterate.
