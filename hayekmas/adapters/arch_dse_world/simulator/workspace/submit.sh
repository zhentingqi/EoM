#!/bin/bash
# BEFORE invoking eval.py (which costs 1 budget call valid-or-invalid),
# we run precheck.py to catch obviously-malformed mappings locally. If
# precheck rejects, we print its JSON and return 0 — no Timeloop call,
# no budget consumed. Otherwise eval.py runs normally and prints its own JSON.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONNOUSERSITE=1

# Local pre-check. precheck.py has a built-in YAML fallback parser, so it
# works with any python3 ≥ 3.6 (no PyYAML required). We pick whichever
# python3 is first on PATH — typically the conda env's python3, which the
# adapter's launcher puts first via PATH=$CONDA_ENV/bin:...
PRECHECK_PY="$(command -v python3 || command -v python3.11 || echo python)"
PRECHECK_OUT="$($PRECHECK_PY precheck.py 2>&1 || true)"
if echo "$PRECHECK_OUT" | grep -q '"local_pre_check_failed":[[:space:]]*true'; then
  # Locally rejected — emit the precheck JSON and exit. Budget unchanged.
  echo "$PRECHECK_OUT"
  exit 0
fi

python3 eval.py "$@"
