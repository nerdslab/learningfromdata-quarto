#!/usr/bin/env bash
#
# Render notebooks to reveal.js slides for in-class presentation.
#
# Why this is not just "quarto render": each notebook is a Jupytext pair
# (Notebook_X.ipynb + Notebook_X.py living side by side). Quarto detects the
# pair, picks the .py as the canonical source, then chokes on the Colab cell
# metadata embedded in the percent comments — so a normal project render finds
# 0 inputs and produces nothing. To avoid that, we copy ONLY the .ipynb files
# (which carry the executed outputs/plots) into a clean build dir with the
# Quarto config, render there, and copy the slides back to _slides/.
#
# Usage:
#   scripts/render_slides.sh                       # all notebooks
#   scripts/render_slides.sh notebooks/Notebook_1/Notebook_1_Learning_from_Data.ipynb
#   scripts/render_slides.sh path/to/Notebook.py   # .py is accepted too
#
# Then view (e.g. over an SSH tunnel):
#   python3 -m http.server 8080 --directory _slides
#
# Note: slides use the OUTPUTS already embedded in the .ipynb (eval: false), so
# run scripts/execute_notebooks.sh first if a notebook's plots are stale.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUARTO="$(command -v quarto)"
OUT="$ROOT/_slides"
[ -n "$QUARTO" ] || { echo "quarto not found on PATH; install from quarto.org"; exit 1; }
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

# Collect targets: args (normalized to .ipynb) or every notebook under notebooks/.
targets=()
if [ "$#" -gt 0 ]; then
  for f in "$@"; do
    nb="${f%.py}"; nb="${nb%.ipynb}.ipynb"
    targets+=("$nb")
  done
else
  while IFS= read -r nb; do targets+=("$nb"); done \
    < <(find "$ROOT/notebooks" -name '*.ipynb')
fi

# Stage the build dir: config + .ipynb files (no .py pairs to confuse Quarto).
cp "$ROOT/_quarto.yml" "$ROOT/custom.scss" "$BUILD/"
for nb in "${targets[@]}"; do
  [ -f "$nb" ] || { echo "skip (missing): $nb"; continue; }
  rel="${nb#"$ROOT"/}"                 # path relative to repo root
  mkdir -p "$BUILD/$(dirname "$rel")"
  cp "$nb" "$BUILD/$rel"
done

# Render as a project (applies _quarto.yml: theme, fonts, chalkboard, etc.).
( cd "$BUILD" && "$QUARTO" render ) || { echo "render failed"; exit 1; }

# Publish the slides to _slides/ at the repo root.
rm -rf "$OUT"
mkdir -p "$OUT"
cp -r "$BUILD/_slides/." "$OUT/"

echo ""
echo "slides written to: $OUT"
echo "view with:  python3 -m http.server 8080 --directory _slides"
