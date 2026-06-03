#!/bin/bash
# Launcher for arch_dse_world ResNet-50 per-layer DSE.
#
# Spawns one SLURM worker per kernel (default 24 kernels = ResNet-50
# unique conv layers) and runs each worker through main.py against the
# arch_dse_world adapter. Each worker writes its checkpoint under
# outputs/arch_dse_world_perlayer_<TIMESTAMP>/workers/worker_<i>/.
#
# Reward configuration applied at launch (encodes our published setup):
#   mas.reward.distribution_mode      = "per_unique_agent"
#       Path reward is divided by the number of unique agents that
#       contributed to the episode, not by total action count. Without
#       this, the Executor (which acts most often) eats most of the
#       reward and the Historian / Planner starve.
#   mas.reward.step_reward_split_chain = true
#   mas.reward.step_reward_chain_window = 3
#       When an Executor submit earns env reward, the payout is split
#       across the last 3 unique auction winners — i.e. the
#       Historian → Planner → Executor chain that produced the submit —
#       instead of going only to the Executor.
#
# USAGE:
#   scripts/arch_dse_world/launch_24jobs_perlayer.sh [WORKER_COUNT]
#
# REQUIRED ENV (override the placeholders below for your cluster):
#   REPO_ROOT             absolute path to this repo on the submit host
#   GLOBAL_CONFIG_SRC     path to global_configs/train_arch_dse_world.json
#   ADAPTER_CONFIG_SRC    path to the per-workload adapter config
#   PRIORS_SRC            path to priors_per_layer.json
#   GEMMA_ENDPOINT_JSON   JSON file containing {"base_url": "<gemma-server>"}
#                         (the Gemma vLLM server is the default LLM)

set -euo pipefail

WORKER_COUNT=${1:-24}
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
SCRIPT_DIR="$REPO_ROOT/scripts/arch_dse_world"
GLOBAL_CONFIG_SRC="${GLOBAL_CONFIG_SRC:-$REPO_ROOT/global_configs/train_arch_dse_world.json}"
ADAPTER_CONFIG_SRC="${ADAPTER_CONFIG_SRC:-$REPO_ROOT/hayekmas/adapters/arch_dse_world/configs/train_arch_dse_world-resnet50.json}"
PRIORS_SRC="${PRIORS_SRC:-$REPO_ROOT/hayekmas/adapters/arch_dse_world/configs/priors_per_layer.json}"
GEMMA_ENDPOINT_JSON="${GEMMA_ENDPOINT_JSON:-}"

if [ ! -f "$PRIORS_SRC" ]; then
  echo "ERROR: priors not found at $PRIORS_SRC" >&2
  echo "       Generate it first: python3.11 scripts/arch_dse_world/build_priors.py" >&2
  exit 1
fi

cd "$REPO_ROOT"
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$REPO_ROOT/outputs/arch_dse_world_perlayer_$TS"
for i in $(seq 0 $((WORKER_COUNT-1))); do
  mkdir -p "$RUN_DIR/workers/worker_$i"
done
echo "RUN_DIR=$RUN_DIR" | tee "$RUN_DIR/run_dir.txt"

# Snapshot the exact code + configs that produced this run, so the
# result is reproducible from RUN_DIR alone.
cp "$0" "$RUN_DIR/launcher.snapshot.sh"
cp "$SCRIPT_DIR/worker.sbatch" "$RUN_DIR/worker.sbatch.snapshot"
cp "$SCRIPT_DIR/build_priors.py" "$RUN_DIR/build_priors.snapshot.py"
cp "$GLOBAL_CONFIG_SRC" "$RUN_DIR/global_config.snapshot.json"
cp "$ADAPTER_CONFIG_SRC" "$RUN_DIR/adapter_config.snapshot.json"
cp "$PRIORS_SRC" "$RUN_DIR/priors_per_layer.snapshot.json"
[ -n "$GEMMA_ENDPOINT_JSON" ] && cp "$GEMMA_ENDPOINT_JSON" "$RUN_DIR/gemma_endpoint_at_launch.json" || true
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/git_sha.txt" 2>/dev/null || echo "no-git" > "$RUN_DIR/git_sha.txt"
mkdir -p "$RUN_DIR/code_snapshot"
cp -r "$REPO_ROOT/hayekmas/adapters/arch_dse_world" "$RUN_DIR/code_snapshot/arch_dse_world"
cp "$REPO_ROOT/hayekmas/base/mas.py" "$RUN_DIR/code_snapshot/mas.py"
cp "$REPO_ROOT/hayekmas/base/config.py" "$RUN_DIR/code_snapshot/config.py"

# Resolve the Gemma server URL (optional — only if user provided an endpoint file).
if [ -n "$GEMMA_ENDPOINT_JSON" ] && [ -f "$GEMMA_ENDPOINT_JSON" ]; then
  GEMMA_URL=$(python3.11 -c "import json; print(json.load(open('$GEMMA_ENDPOINT_JSON'))['base_url'].rstrip('/'))")
else
  GEMMA_URL=""
fi

# Build the derived global config: inject api_base, priors path, and the
# v7 reward-shaping knobs. The user-visible config on disk stays clean.
DERIVED_GLOBAL="$RUN_DIR/global_config_derived.json"
python3.11 - "$GLOBAL_CONFIG_SRC" "$DERIVED_GLOBAL" "$GEMMA_URL" "$PRIORS_SRC" <<'PYEOF'
import json, sys
from pathlib import Path
src, dst, url, priors = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
cfg = json.loads(Path(src).read_text())
if url:
    cfg.setdefault("model", {})["api_base"] = url
cfg.setdefault("run", {})["experience_priors_path"] = priors
cfg.setdefault("mas", {}).setdefault("reward", {})["distribution_mode"] = "per_unique_agent"
cfg["mas"]["reward"]["step_reward_split_chain"] = True
cfg["mas"]["reward"]["step_reward_chain_window"] = 3
Path(dst).write_text(json.dumps(cfg, indent=2))
print(f"derived global → {dst}")
print(f"  api_base={url or '(unset — uses config default)'}")
print(f"  priors={priors}")
print(f"  reward.distribution_mode=per_unique_agent")
print(f"  reward.step_reward_split_chain=True")
print(f"  reward.step_reward_chain_window=3")
PYEOF

if [ -n "$GEMMA_URL" ] && ! curl -fsS --max-time 5 "$GEMMA_URL/models" > /dev/null 2>&1; then
  echo "WARN: Gemma endpoint not reachable: $GEMMA_URL" >&2
fi

# Materialize the SLURM script for this exact run dir + worker count.
RUN_SBATCH="$RUN_DIR/worker.sbatch"
sed "s|outputs/arch_dse_world_perlayer_TS_PLACEHOLDER|$RUN_DIR|g; s|--array=0-23|--array=0-$((WORKER_COUNT-1))|g" \
    "$SCRIPT_DIR/worker.sbatch" > "$RUN_SBATCH"
chmod +x "$RUN_SBATCH"

JOB_OUT="$RUN_DIR/submitted_jobs.txt"
JOBID=$(RUN_DIR="$RUN_DIR" GLOBAL_CONFIG="$DERIVED_GLOBAL" WORKER_COUNT="$WORKER_COUNT" \
        sbatch --parsable "$RUN_SBATCH")
echo "Submitted arch_dse_world array job: $JOBID (workers 0..$((WORKER_COUNT-1)))" | tee "$JOB_OUT"

cat <<EOF

=== arch_dse_world per-layer run — submitted ===
Run dir:        $RUN_DIR
Workers:        $WORKER_COUNT
Array job:      $JOBID
Reward mode:    per_unique_agent (path) + step_reward_split_chain=True
Gemma endpoint: ${GEMMA_URL:-(not set)}

ETA on our 24-kernel ResNet-50 setup: ~75 min wallclock.

  squeue -j $JOBID
EOF
