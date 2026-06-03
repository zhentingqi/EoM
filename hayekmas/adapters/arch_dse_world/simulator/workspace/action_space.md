# Action Space: Gemmini Mapping String

## Format

```
L3[WIO] <factors> - L2[WI] <factors> - L1[O] <factors> - L0[W] <factors>
```

Four blocks joined by ` - `. Each block has:
- **Level tag** `L<n>[tensors]` — fixed per level, do not change.
- **Factors** — space-separated, each of the form `<DIM><INT>` (temporal) or `<DIM><INT>X`
  (spatial along X).

## Level tags (fixed)

| Tag | Memory | Tensors held |
|-----|--------|--------------|
| `L3[WIO]` | DRAM | Weights, Inputs, Outputs |
| `L2[WI]`  | Scratchpad (shared SRAM) | Weights, Inputs |
| `L1[O]`   | Accumulator (per-column SRAM) | Outputs |
| `L0[W]`   | Registers (per-PE) | Weights |

## Dimensions (CNN layer, 7 dims)

| Symbol | Meaning | Typical range |
|--------|---------|---------------|
| `R` | filter height | 1, 3, 5, 7 |
| `S` | filter width | 1, 3, 5, 7 |
| `P` | output height | 7, 14, 28, 56, 112, 224 |
| `Q` | output width | same as P |
| `C` | input channels | 3..2048 |
| `K` | output channels | 64..2048 |
| `N` | batch size | 1 |

## Factor rules

1. **Factor product must equal dimension**: for each dim D,
   `factor_L3[D] × factor_L2[D] × factor_L1[D] × factor_L0[D] = prob[D]`.
   Factors not mentioned default to 1.

2. **Spatial slots are restricted on Gemmini** (matches DOSA's `flat_mapping_to_dict` 113-116):
   - **Only** `K` can be spatial at L2 (Scratchpad) → `K<N>X` allowed, others: temporal only
   - **Only** `C` can be spatial at L1 (Accumulator) → `C<N>X` allowed, others: temporal only
   - L3 and L0 have no spatial factors
   - The product of spatial factors at a level must be ≤ `pe_dim`
   - **Putting CX at L2 or KX at L1 will be rejected** — eval cross-checks Timeloop's
     total Computes against the workload size (catches silent-spatial-zeroing bug).

3. **L0 (registers) — VERY STRICT**:
   - **Only P and Q** temporal factors > 1 are allowed at L0.
   - All other dims (R, S, C, K, N) at L0 **MUST be 1** (i.e., either `D1` or omit).
   - Register capacity is literally 1 value per PE — any factor > 1 on a non-PQ dim will
     be rejected by Timeloop with "mapped tile size N exceeds buffer capacity 1".
   - Typical L0: `L0[W] N1` (nothing), or `L0[W] P2` (small P unroll) if P permits it.

4. **Omit factor 1**: `C1 K1 P1` is noise; just leave them out. Dimension with no factor at a
   level = factor 1 implicitly.

5. **Permutation**: factors within a block are written in **outer-to-inner** order. Outer loops
   are listed first. Example: `L3[WIO] C4 K16 P8` means C is outermost, K middle, P innermost.

## Worked example

Workload: ResNet50 Conv3_1 `1×1, C=256, K=512, P=28, Q=28, R=1, S=1, N=1`
Hardware: Gemmini default, `pe_dim=16, sp_size=128 KB, acc_size=32 KB`

Design (output-stationary, **K spatial at L2, C spatial at L1** — the only valid Gemmini convention):
```
L3[WIO] K32 P28 Q28 - L2[WI] C16 K16X - L1[O] K1 C16X - L0[W] N1
```
- C: 1 × 16 × 16 × 1 = 256 ✓ (16 temporal at L2, 16 spatial at L1)
- K: 32 × 16 × 1 × 1 = 512 ✓ (32 temporal at L3, 16 spatial at L2)
- P: 28 × 1 × 1 × 1 = 28 ✓
- Q: 28 × 1 × 1 × 1 = 28 ✓
- R, S, N: all 1 (implicit) ✓
- Spatial product at L2: 16 ≤ pe_dim=16 ✓ (KX only)
- Spatial product at L1: 16 ≤ pe_dim=16 ✓ (CX only)

This parallelizes K across PEs at the Scratchpad level, then C across PEs at the Accumulator
level. The full 256-PE array is utilized.

⚠️ **DO NOT** write `L2[WI] C16X` (CX at L2) or `L1[O] K16X` (KX at L1). Those positions
are silently zeroed by `flat_mapping_to_dict`, which causes Timeloop to compute only 1/256
of the workload using a single PE. The eval cross-checks `Computes (total) == K×C×P×Q×R×S×N`
and rejects such mappings.

## Common mistakes

- **Non-PQ factor at L0**: most common rejection. `L0[W] C4` fails because registers only
  hold 1 weight per PE; only P/Q unrolls are allowed at L0. Rule: if a dim D other than P/Q
  is not fully factored by L3+L2+L1, put the remaining factor at L3.
- **Factor product mismatch**: Timeloop will reject. Always sanity-check `prod(factors)==prob[D]`
  for each D.
- **Too-large spatial factor**: spatial product at a level can't exceed `pe_dim`.
- **Large tiles exceeding buffer capacity**: Timeloop will evaluate but energy will be terrible
  because of DRAM spilling.
- **Missing tensor dims**: if `R=3, S=3` but you write nothing for R/S, factor 1 is assumed.
  For 3×3 convs you must place R3 and S3 somewhere.

## Dataflow patterns to try

- **Weight-stationary**: put large C or K temporal factor in L0[W] (register) and L2[WI]
  (scratchpad). Keep weight tile in the lowest level across output iterations.
- **Output-stationary**: make C spatial at L1 (accumulator) — `C16X` at L1 — to keep outputs
  resident there during C-reductions; K spatial at L2 (`K16X`) for output-channel parallelism.
- **Row-stationary (Eyeriss-like)**: P spatial parallelism, but not supported on Gemmini — skip.

## Submitting

A submission has two mandatory parts — a hardware config and a mapping
string (plus a short rationale):
```yaml
hw:
  sp_size: 128      # scratchpad KB, range [1, 2048]
  acc_size: 32      # accumulator KB, range [1, 2048]
mapping: "L3[WIO] K32 P28 Q28 - L2[WI] C16 K16X - L1[O] C16X - L0[W] N1"
rationale: "Output-stationary: K spatial at L2 (16-way), C spatial at L1 (16-way) →
           full 256-PE utilization. SP=128KB holds W tile + I tile. ACC=32KB for O tile."
```

The evaluation returns, per submission: `valid`, `cycles`, `energy_uJ`
(microjoules), `edp`, `hw_config` (echoed back), and budget counters.

> Note: the LLM agents in this adapter submit through the JSON
> `candidates` array described in the Executor instructions (each entry
> carries the same `hw` + `mapping` fields shown above); the
> `candidate.yaml` + `./submit.sh` form is the equivalent low-level
> workspace interface and is not what the agent emits directly.

## Choosing sp_size and acc_size

- **L2 scratchpad** holds Weights + Inputs for the inner computation. Size ≥ tile_W + tile_I.
- **L1 accumulator** holds Outputs being summed. Size ≥ tile_O.
- Rough formulas (1 byte per value, adjust for batch/reduction):
  - `tile_W ≈ (C_at_or_above_L2) × (K_at_or_above_L2) × R × S`
  - `tile_I ≈ (C_at_or_above_L2) × tile_PQ`
  - `tile_O ≈ (K_at_or_above_L1) × tile_PQ`
- If you pick SP=1KB but your weight tile is 64KB, Timeloop will spill to DRAM → bad energy.
- If you pick SP=2048KB but only need 64KB, you pay area for unused SRAM (though EDP is the
  minimize target so this doesn't directly hurt EDP; the area field is reported separately).
- Powers of 2 are conventional (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048).
