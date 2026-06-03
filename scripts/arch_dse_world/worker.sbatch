#!/bin/bash
#SBATCH --job-name=arch_dse_worker
#SBATCH --account=your_account
#SBATCH --partition=shared,sapphire,seas_compute
#SBATCH --time=2:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --output=outputs/arch_dse_world_perlayer_TS_PLACEHOLDER/workers/worker_%a/slurm-%j.out
#SBATCH --array=0-23
#
# Worker for one kernel of the arch_dse_world per-layer ResNet-50 run.
# Launched as a SLURM array job by launch_24jobs_perlayer.sh — one task
# per kernel (24 kernels = ResNet-50 unique conv layers). Each task
# carves out its own worker subdir, materializes a per-worker config
# (with worker_id / worker_count baked in), and invokes main.py.
#
# Required environment (set by launcher.sh, not by sbatch CLI):
#   RUN_DIR        run-level output directory
#   GLOBAL_CONFIG  derived global config JSON (already has priors / reward
#                  knobs / Gemma endpoint baked in)
#   WORKER_COUNT   total number of workers in the array (defaults to 24)
#
# Cluster-specific paths below assume the FASRC environment that
# produced the paper figures. Override the four export lines for your
# own cluster.

set -euo pipefail

: "${RUN_DIR:?RUN_DIR required}" "${GLOBAL_CONFIG:?GLOBAL_CONFIG required}"
WORKER_ID=${SLURM_ARRAY_TASK_ID:-0}
WORKER_COUNT=${WORKER_COUNT:-24}

REPO_ROOT="${REPO_ROOT:-/home/user/projects/hayekmas}"
# Exported so the `python3.11 main.py` child inherits it — env.py resolves the
# conda env from DSE_CONDA_ENV, then CONDA_ENV, then a hardcoded default.
export CONDA_ENV="${CONDA_ENV:-/home/user/conda_envs/dosa_dse}"
DOSA_ROOT="${DOSA_ROOT:-/home/user/projects/dosa-src/dosa}"
GUROBI_LIC="${GUROBI_LIC:-/n/sw/gurobi/gurobi-13.0.1/linux64/gurobi.lic}"
ARCHGYM_SCRATCH="${ARCHGYM_SCRATCH:-/home/user/scratch/dosa_tmp}"

cd "$REPO_ROOT"
WORKER_DIR="$RUN_DIR/workers/worker_$WORKER_ID"
mkdir -p "$WORKER_DIR" "$ARCHGYM_SCRATCH"

DERIVED="$WORKER_DIR/derived_config.json"
python3.11 - "$GLOBAL_CONFIG" "$WORKER_DIR" "$WORKER_ID" "$WORKER_COUNT" "$DERIVED" <<'PYEOF'
import json, sys
from pathlib import Path
src, wdir, wid, wcount, dst = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
cfg = json.loads(Path(src).read_text())
cfg["run"]["output"] = str(wdir / "checkpoint_final.json")
cfg["run"]["checkpoint_steps"] = str(wdir / "checkpoint_step.json")
cfg["run"]["worker_id"] = wid
cfg["run"]["worker_count"] = wcount
cfg["profile"] = f"arch_dse_world_w{wid}_of_{wcount}"
Path(dst).write_text(json.dumps(cfg, indent=2))
print(f"derived → {dst}: worker {wid}/{wcount}")
PYEOF

export PATH="$CONDA_ENV/bin:$CONDA_ENV/share/cacti:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
export DOSA_ROOT="$DOSA_ROOT"
export GRB_LICENSE_FILE="$GUROBI_LIC"
export ARCHGYM_SCRATCH="$ARCHGYM_SCRATCH"

echo "[$(date)] arch_dse_world worker $WORKER_ID/$WORKER_COUNT starting; output → $WORKER_DIR"
python3.11 main.py "$DERIVED" 2>&1 | tee "$WORKER_DIR/run.log"
echo "[$(date)] arch_dse_world worker $WORKER_ID/$WORKER_COUNT done; rc=$?"
