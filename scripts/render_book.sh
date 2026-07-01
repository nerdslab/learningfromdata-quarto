#!/usr/bin/env bash
#
# Render all notebooks into a textbook-style HTML site (Quarto book): scrollable
# pages, a sidebar with every chapter, full-text search, and per-page tables of
# contents.
#
# Jupytext caveat: a normal "quarto render" finds 0 inputs because the .py pair
# shadows each .ipynb, so we render from a clean build dir containing only the
# .ipynb files plus the book config.
#
# Usage:
#   scripts/render_book.sh
#
# Then view (e.g. over an SSH tunnel):
#   python3 -m http.server 8080 --directory _book
#
# Note: pages use the OUTPUTS already embedded in the .ipynb (eval: false), so
# run scripts/execute_notebooks.sh first if any plots are stale.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUARTO="$(command -v quarto)"
OUT="$ROOT/_book"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
[ -n "$QUARTO" ] || { echo "quarto not found on PATH; install from quarto.org"; exit 1; }

# Stage the build dir: book config (as _quarto.yml), landing page, and every
# .ipynb (no .py pairs to confuse Quarto's project scan).
cp "$ROOT/_quarto-book.yml" "$BUILD/_quarto.yml"
cp "$ROOT/index.qmd" "$ROOT/book.css" "$BUILD/"
while IFS= read -r nb; do
  rel="${nb#"$ROOT"/}"
  mkdir -p "$BUILD/$(dirname "$rel")"
  cp "$nb" "$BUILD/$rel"
done < <(find "$ROOT/notebooks" -name '*.ipynb')

# Render the book.
( cd "$BUILD" && "$QUARTO" render ) || { echo "render failed"; exit 1; }

# Publish to _book/ at the repo root.
rm -rf "$OUT"
mkdir -p "$OUT"
cp -r "$BUILD/_book/." "$OUT/"

echo ""
echo "book written to: $OUT"
echo "view with:  python3 -m http.server 8080 --directory _book"
