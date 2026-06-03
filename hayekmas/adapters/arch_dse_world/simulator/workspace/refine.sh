#!/bin/bash
# Local-search refinement tool wrapper. CONSUMES evaluation budget: each
# internal Timeloop call it runs charges 1 unit against budget.json (same
# accounting as submit.sh). Use it after you've found a promising mapping to
# locally polish it.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
python3 refine_mapping.py "$@"
