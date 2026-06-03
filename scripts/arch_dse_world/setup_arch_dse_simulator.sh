#!/usr/bin/env bash
# =============================================================================
# setup_arch_dse_simulator.sh — one-command install of the arch_dse_world
# simulator backend (Timeloop + Accelergy + DOSA) into a self-contained conda env.
#
# WHY THIS EXISTS: the arch_dse_world Executor's reward comes from running
# Timeloop+Accelergy on a (hardware, mapping) pair via the DOSA Gemmini
# pipeline. Timeloop is a C++ tool that must be compiled — it is not a pip
# package — so a pure `pip install` cannot provide it. This script automates
# DOSA's documented build (https://github.com/ucb-bar/dosa) using conda-forge
# for the C++ dependencies, so NO `sudo apt` is needed.
#
# Gurobi is NOT installed/needed: the arch_dse_world eval path only runs
# Timeloop on a given mapping. Gurobi is used only by DOSA's own mapping-
# search optimizer (the baseline, already cached in dosa_bounded_edps.json).
#
# WHAT YOU GET: a conda env laid out exactly how this repo's adapter expects
# (timeloop-* binaries in $ENV/bin, cacti in $ENV/share/cacti), plus a DOSA
# checkout. The script prints the exact env vars + run command at the end.
#
# REQUIREMENTS: conda/mamba, git, a C++ toolchain (the script installs gxx
# from conda-forge), ~20 min for the Timeloop compile, ~5 GB disk.
#
# NOTE: this mirrors the environment that produced the paper and follows
# DOSA's upstream build steps verbatim; the heavy Timeloop compile is not
# re-run in CI, so treat build failures as upstream-DOSA/Timeloop issues and
# consult https://github.com/ucb-bar/dosa.
#
# USAGE:
#   bash scripts/arch_dse_world/setup_arch_dse_simulator.sh [ENV_NAME] [DOSA_DIR]
#     ENV_NAME  conda env name to create   (default: arch_dse_sim)
#     DOSA_DIR  where to clone DOSA         (default: ./third_party/dosa)
# =============================================================================
set -euo pipefail

ENV_NAME="${1:-arch_dse_sim}"
DOSA_DIR="${2:-$(pwd)/third_party/dosa}"
JOBS="${JOBS:-4}"

command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found on PATH." >&2; exit 1; }
command -v git   >/dev/null 2>&1 || { echo "ERROR: git not found on PATH." >&2; exit 1; }

echo "==> [1/5] Creating conda env '$ENV_NAME' with C++ build deps (conda-forge, no sudo)"
# Versions mirror the env that produced the paper (boost 1.82, yaml-cpp 0.8,
# libconfig 1.7, scons 4.x, python 3.10) — the conda-forge equivalents of
# DOSA's apt dependency list.
conda create -y -n "$ENV_NAME" -c conda-forge \
    python=3.10 scons \
    boost=1.82 libboost-devel=1.82 yaml-cpp=0.8 libconfig=1.7 \
    ncurses gxx_linux-64 gcc_linux-64 make cmake \
    numpy=1.26 scipy pyyaml

# Resolve the env prefix so we can install binaries into the layout env.py expects.
ENV_PREFIX="$(conda env list | awk -v n="$ENV_NAME" '$1==n {print $NF}')"
[ -n "$ENV_PREFIX" ] || { echo "ERROR: could not resolve prefix for env '$ENV_NAME'." >&2; exit 1; }
echo "    env prefix: $ENV_PREFIX"
run() { conda run -p "$ENV_PREFIX" --no-capture-output bash -c "$*"; }

echo "==> [2/5] Cloning DOSA + submodules into $DOSA_DIR"
if [ ! -d "$DOSA_DIR/.git" ]; then
    git clone https://github.com/ucb-bar/dosa "$DOSA_DIR"
fi
git -C "$DOSA_DIR" submodule update --init --recursive

INFRA="$DOSA_DIR/accelergy-timeloop-infrastructure/src"

echo "==> [3/5] Building Accelergy + CACTI + plug-ins (per DOSA's README)"
run "cd '$INFRA/accelergy'              && pip install ."
run "cd '$INFRA/cacti'                  && make -j$JOBS"
# env.py expects cacti at \$ENV/share/cacti and adds it to PATH.
mkdir -p "$ENV_PREFIX/share"
rm -rf "$ENV_PREFIX/share/cacti"
cp -r "$INFRA/cacti" "$ENV_PREFIX/share/cacti"
run "cd '$INFRA/accelergy-cacti-plug-in'         && pip install ."
run "cd '$INFRA/accelergy-aladdin-plug-in'       && pip install ."
run "cd '$INFRA/accelergy-table-based-plug-ins'  && pip install ."

echo "==> [4/5] Building Timeloop (scons --accelergy --static)"
run "cd '$INFRA/timeloop/src' && ln -sf ../pat-public/src/pat ."
run "cd '$INFRA/timeloop' && PATH=\"$ENV_PREFIX/share/cacti:\$PATH\" scons --accelergy --static -j$JOBS"
# env.py finds timeloop-* on PATH via \$ENV/bin — install the built binaries there.
cp "$INFRA/timeloop/build/"timeloop-* "$ENV_PREFIX/bin/"

echo "==> [5/5] Installing DOSA (pip install -e .)"
run "cd '$DOSA_DIR' && pip install -e ."

cat <<EOF

============================================================================
✅ Simulator env ready: $ENV_PREFIX

Sanity check (should print a JSON with an 'edp' field):
  conda run -p "$ENV_PREFIX" timeloop-model --help >/dev/null && echo "timeloop OK"

To RUN arch_dse_world end-to-end (LLM + simulator):
  export TOGETHER_API_KEY=...          # your Together AI key (or use another backend)
  export DSE_CONDA_ENV="$ENV_PREFIX"
  export DOSA_ROOT="$DOSA_DIR"
  export ARCHGYM_SCRATCH="\$(mktemp -d)"
  python main.py global_configs/train_arch_dse_world.json
============================================================================
EOF
