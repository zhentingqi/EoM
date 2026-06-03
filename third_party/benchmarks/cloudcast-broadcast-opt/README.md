# CloudCast — Multi-Cloud Broadcast Optimization (HayekMAS wrapper)

This directory packages the [CORAL ADRS CloudCast](../../../CORAL/examples/ADRS/cloudcast)
task into the cloudcast layout that
`hayekmas/adapters/cloudcast/` already understands:

```
cloudcast-broadcast-opt/
├── task.toml                     # cloudcast metadata (resource hints)
├── instruction.md                # problem statement shown to the agents
├── environment/                  # workspace template, copied per-episode
│   ├── initial_program.py        # the EVOLVE-BLOCK code the agents edit
│   └── profiles/                 # cost.csv + throughput.csv (used by helpers)
└── tests/
    ├── compute_reward.py         # verifier shim (cloudcast protocol)
    └── cloudcast_eval/           # bundled CORAL evaluator + dataset
        ├── broadcast.py
        ├── evaluator.py          # patched: removed top-level seed import
        ├── simulator.py
        ├── utils.py
        ├── profiles/{cost,throughput}.csv
        └── examples/config/{intra_aws,intra_azure,intra_gcp,inter_agz,inter_gaz2}.json
```

## Reward shaping

The agent's `score` reported to HayekMAS is the **fractional cost reduction
relative to the seed**:

    score = max(0, 1 − total_cost / 1035.0)

so seed → 0, halved cost → 0.5, near-free transfer → ~1.  The raw
CORAL `combined_score = 1/(1+total_cost)` is preserved in the
`reward.json` `raw_evaluator` block and the `subscores` list.

The cloudcast env applies the standard pipeline on top:

- `request_eval()` mid-episode → reward = `env_reward_scale × Δscore`
- `final_answer(...)` end-of-episode → terminal verifier run → reward
  = `terminal_output_bonus_scale × score`

so the agents are paid every time they make the cost go down (and pay
nothing — but also lose nothing — for regressions).

## Running

From the hayekmas repo root:

```bash
# verifier sanity check (no LLM calls)
scripts/run_cloudcast.sh verifier
# → score=0.0000  total_cost=1035.0  configs=5/5

# 1-episode smoke through the multi-agent loop (10 steps)
scripts/run_cloudcast.sh smoke

# 5-episode evolutionary training (25 steps / episode)
scripts/run_cloudcast.sh train

# eval-only: no births, no rent
scripts/run_cloudcast.sh eval
```

Or invoke directly:

```bash
python main.py global_configs/smoke_cloudcast_cloudcast.json
python main.py global_configs/train_cloudcast_cloudcast.json
python main.py global_configs/eval_cloudcast_cloudcast.json
```

Run-level config lives in
`hayekmas/adapters/cloudcast/configs/{smoke,train,eval}_cloudcast.json`;
the global JSON wraps it with the model + profile name.

## Standalone evaluator

You can also bypass hayekmas entirely and reuse the bundled CORAL evaluator
for ad-hoc experiments on a single program file:

```bash
python tasks/cloudcast-broadcast-opt/tests/compute_reward.py \
  --app-dir <workspace_with_initial_program.py> \
  --output-dir <where_to_write_reward.json>
```

The workspace must contain `initial_program.py`; `profiles/` is optional
(only used by the seed's standalone helpers — the verifier reads from the
bundled `tests/cloudcast_eval/profiles/`).

## Source

CORAL upstream: <https://github.com/sky-proj/coral> (path
`examples/ADRS/cloudcast`).  Dataset comes from skydiscover's
`benchmarks/ADRS/cloudcast/evaluator/download_dataset.sh`.
