#!/usr/bin/env bash
#
# Execute notebooks and embed their outputs (plots, images, tables) into the
# .ipynb, so anyone viewing the notebook on GitHub/Colab sees the rendered
# results. The paired .py holds inputs only, so the .ipynb is what carries
# outputs — run this when you finalize a notebook or before pushing.
#
# Usage:
#   scripts/execute_notebooks.sh                       # all notebooks
#   scripts/execute_notebooks.sh notebooks/Notebook_1/Notebook_1a_Overview_and_Data_Representation.ipynb
#   scripts/execute_notebooks.sh path/to/Notebook.py   # .py is accepted too
#
# Requires the runtime deps to be installed:  uv sync --extra dev  (plus the
# notebook deps in pyproject.toml). Needs a python3 Jupyter kernel.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/venv/bin"
TIMEOUT="${CELL_TIMEOUT:-120}"   # per-cell timeout in seconds; override via env
# Use the matplotlib inline backend during execution: it both EMBEDS each figure
# as a PNG in the .ipynb (which Agg does NOT — Agg renders off-screen and the plot
# is silently dropped) and avoids plt.show() hanging on a missing display.
export MPLBACKEND=module://matplotlib_inline.backend_inline
# Silence tqdm progress bars during batch execution: their carriage-return
# updates get captured to stderr and embedded as ugly static text. Bars are only
# useful when running live (e.g. a student in Colab), not in the rendered output.
export TQDM_DISABLE=1

# Collect targets: args (normalized to .ipynb) or every notebook under notebooks/.
targets=()
if [ "$#" -gt 0 ]; then
  for f in "$@"; do
    targets+=("${f%.py}")          # strip .py if given
    targets[-1]="${targets[-1]%.ipynb}.ipynb"
  done
else
  while IFS= read -r nb; do targets+=("$nb"); done \
    < <(find "$ROOT/notebooks" -name '*.ipynb')
fi

failed=()
for nb in "${targets[@]}"; do
  [ -f "$nb" ] || { echo "skip (missing): $nb"; continue; }
  echo ">> executing: $nb"
  # 1) propagate any .py edits into the .ipynb first
  "$BIN/jupytext" --sync "$nb" >/dev/null 2>&1
  # jupytext may regenerate the .ipynb with nbformat_minor=0, but cell ids
  # require 4.5 — bump it so nbconvert validation passes
  python3 -c "
import json, sys
nb = json.load(open(sys.argv[1]))
if nb.get('nbformat_minor', 0) < 5:
    nb['nbformat_minor'] = 5
    json.dump(nb, open(sys.argv[1], 'w'), indent=1)
" "$nb"
  # 2) run all cells and write outputs back in place; --allow-errors keeps going
  #    so one failing cell (e.g. a homework blank or missing dep) still lets the
  #    rest of the notebook render its images/plots.
  if ! "$BIN/jupyter" nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.kernel_name=python3 \
        --ExecutePreprocessor.timeout="$TIMEOUT" \
        "$nb"; then
    failed+=("$nb")
  fi
  # 3) keep the paired .py in step (inputs only; outputs stay in the .ipynb)
  "$BIN/jupytext" --sync "$nb" >/dev/null 2>&1
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo ""
  echo "WARNING: nbconvert reported problems for:"
  printf '  - %s\n' "${failed[@]}"
  echo "(outputs were still embedded where cells succeeded.)"
fi
echo "done."
