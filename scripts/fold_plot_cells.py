#!/usr/bin/env python3
"""Add `#| code-fold: true` to plotting-dominant code cells, uniformly.

Design: cells whose job is a visualization should show the figure but tuck the
plotting boilerplate behind a "Code" expander; cells that are computation/logic
stay fully shown for students to study. A cell qualifies as "plotting-dominant"
when it renders a figure AND the majority of its code lines are plotting calls.

Cells that already carry a `#|` directive are left untouched (so manual choices
like the kNN split survive). Notebook 1 is skipped by default.

Usage:
    python3 scripts/fold_plot_cells.py --dry-run     # show what would change
    python3 scripts/fold_plot_cells.py               # apply
    python3 scripts/fold_plot_cells.py --include-nb1 # also touch Notebook 1
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELL = re.compile(r'^# %%')

# A call that actually renders a figure (so the cell has a visible viz).
RENDERS = re.compile(r'plt\.show\(|\.plot\(|sns\.\w+plot|sns\.(heatmap|scatter|hist|box|violin|kde|joint|pair|count|bar|line|dis|reg|displot|catplot)|imshow|\.imshow|plot_tree|plot_confusion|\.bar\(|\.hist\(|\.scatter\(')
# Lines that are "plotting" in nature.
PLOTTY = re.compile(r'(plt\.|sns\.|\bax\b|\baxs\b|\baxes\b|\bfig\b|\.plot\(|\.bar\(|\.hist\(|\.scatter\(|imshow|colorbar|set_(title|xlabel|ylabel|xlim|ylim|xticks|yticks|xticklabels|yticklabels|aspect)|legend\(|tight_layout|subplots|suptitle|axhline|axvline|fill_between|annotate|despine|set_theme|gca\(|gcf\()')
# Model / training logic worth keeping visible — never fold a cell with these.
LOGIC = re.compile(r'\.fit\(|\.fit_transform\(|\.fit_predict\(|\.predict\(|\.predict_proba\(|\.transform\(|\.score\(|\.backward\(|optimizer\.|loss\.backward|\.train\(\)|\blinkage\(')


def is_code_line(s):
    s = s.strip()
    return bool(s) and not s.startswith('#')


def classify(body):
    code = [l for l in body if is_code_line(l[2:] if l.startswith('# ') else l)]
    # body lines are raw code in percent cells (not '# '-prefixed)
    code = [l for l in body if l.strip() and not l.strip().startswith('#')]
    if not code:
        return False, 0, 0
    renders = any(RENDERS.search(l) for l in body)
    plot_n = sum(1 for l in code if PLOTTY.search(l))
    return (renders and plot_n >= 0.5 * len(code)), plot_n, len(code)


def process(path, apply):
    lines = open(path).read().split('\n')
    starts = [i for i, l in enumerate(lines) if CELL.match(l)] + [len(lines)]
    fold_at = []
    for s, e in zip(starts, starts[1:]):
        header = lines[s]
        if '[markdown]' in header or '[raw]' in header:
            continue
        body = lines[s + 1:e]
        first = next((l for l in body if l.strip()), '')
        if first.strip().startswith('#|'):        # already has a directive
            continue
        if any(LOGIC.search(l) for l in body):     # keep model/training logic visible
            continue
        ok, pn, cn = classify(body)
        if ok:
            fold_at.append((s, first.strip()[:46], pn, cn))
    if apply:
        for s, *_ in sorted(fold_at, reverse=True):
            lines.insert(s + 1, '#| code-fold: true')
        open(path, 'w').write('\n'.join(lines))
    return fold_at


def main():
    apply = '--dry-run' not in sys.argv
    inc_nb1 = '--include-nb1' in sys.argv
    total = 0
    for path in sorted(glob.glob(os.path.join(ROOT, 'notebooks/**/*.py'), recursive=True)):
        if not inc_nb1 and 'Notebook_1_Learning' in path:
            continue
        hits = process(path, apply)
        if hits:
            print(f'\n{os.path.basename(path)}  ({len(hits)} cells)')
            for _, first, pn, cn in hits:
                print(f'    [{pn}/{cn} plot lines]  {first}')
            total += len(hits)
    verb = 'folded' if apply else 'would fold'
    print(f'\n{verb} {total} plotting cells')


if __name__ == '__main__':
    main()
