#!/usr/bin/env python3
"""Inject "Open in Colab" badges that deep-link to exercise cells.

Convention: mark an exercise by giving its markdown cell an id that starts with
"ex-", e.g.

    # %% [markdown] id="ex-matrix-diagnose"
    # ### Exercise: Diagnose a Matrix Transformation
    # ...

This script then ensures each such cell carries a Colab badge whose URL is
`...<notebook>.ipynb#scrollTo=<that-id>`, so clicking it opens the notebook in
Colab scrolled straight to the exercise. The badge is generated (never hand-
written) and the script is idempotent — re-running updates URLs in place rather
than duplicating badges.

Usage:
    python3 scripts/add_exercise_badges.py            # all notebooks
    python3 scripts/add_exercise_badges.py notebooks/Notebook_1/...py

Run `jupytext --sync` afterwards (or scripts/execute_notebooks.sh) to propagate
into the .ipynb. The deep-link only does anything in Colab; elsewhere it is a
harmless link to the notebook.
"""
import glob
import os
import re
import sys

REPO = "nerdslab/learningfromdata-course"
BRANCH = "main"
BADGE_IMG = "https://colab.research.google.com/assets/colab-badge.svg"
PREFIX = "ex-"                      # cells whose id starts with this are exercises
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CELL = re.compile(r'^# %%')
IDRE = re.compile(r'id="([^"]+)"')
HEADING = re.compile(r'^# #{1,6} ')


def badge_line(rel_ipynb, cell_id):
    url = (f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/"
           f"{rel_ipynb}#scrollTo={cell_id}")
    return f"# [![Open in Colab]({BADGE_IMG})]({url})"


def collapse_blank_comments(body):
    """Collapse runs of bare '#' lines into a single one (keeps cells tidy
    across re-runs)."""
    out = []
    for ln in body:
        if ln.strip() == '#' and out and out[-1].strip() == '#':
            continue
        out.append(ln)
    return out


def process(path):
    rel_ipynb = os.path.relpath(path, ROOT)[:-3] + ".ipynb"
    lines = open(path).read().split('\n')

    starts = [i for i, l in enumerate(lines) if CELL.match(l)] + [len(lines)]
    new = lines[:starts[0]]                       # preamble (jupytext header)
    n_changed = 0

    for s, e in zip(starts, starts[1:]):
        header, body = lines[s], lines[s + 1:e]
        m = IDRE.search(header)
        cid = m.group(1) if m else None

        if '[markdown]' in header and cid and cid.startswith(PREFIX):
            want = badge_line(rel_ipynb, cid)
            # drop any existing deep-link badge, then tidy blank lines
            body = [ln for ln in body if 'scrollTo=' not in ln]
            body = collapse_blank_comments(body)
            # insert right after the first heading line (or at the top)
            ins = next((j + 1 for j, ln in enumerate(body) if HEADING.match(ln)), 0)
            body = body[:ins] + ['#', want, '#'] + body[ins:]
            body = collapse_blank_comments(body)
            n_changed += 1

        new.append(header)
        new.extend(body)

    open(path, 'w').write('\n'.join(new))
    return n_changed


def main():
    targets = sys.argv[1:] or sorted(
        glob.glob(os.path.join(ROOT, 'notebooks/**/*.py'), recursive=True))
    total = 0
    for path in targets:
        c = process(path)
        total += c
        if c:
            print(f"{os.path.relpath(path, ROOT)}: {c} exercise badge(s)")
    if total == 0:
        print(f'no exercise cells found (tag one with id="{PREFIX}<slug>")')


if __name__ == '__main__':
    main()
