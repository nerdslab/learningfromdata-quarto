#!/usr/bin/env python3
"""Remove tqdm progress-bar text from notebook outputs.

tqdm writes carriage-return bar updates to stderr; when a notebook is executed
by nbconvert those get captured and embedded as ugly static text like
`0%|          | 0/200 [00:00<?, ?it/s]`. This strips those lines from stderr
stream outputs while keeping any genuine stderr (e.g. real warnings).

New executions avoid the problem entirely (execute_notebooks.sh sets
TQDM_DISABLE=1); this is for cleaning notebooks that were run before that.

Usage:
    python3 scripts/strip_tqdm_output.py            # all notebooks
    python3 scripts/strip_tqdm_output.py path/to/Notebook.ipynb
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TQDM = re.compile(r'\d+%\|| it/s|it/s\]')


def is_bar(line):
    return bool(TQDM.search(line))


def clean(path):
    nb = json.load(open(path))
    removed = 0
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        new_outputs = []
        for o in cell.get('outputs', []):
            if o.get('output_type') == 'stream' and o.get('name') == 'stderr':
                text = o.get('text', '')
                if isinstance(text, list):
                    text = ''.join(text)
                # tqdm uses \r to overwrite; split on both and drop bar fragments
                kept = [ln for ln in re.split(r'[\r\n]', text)
                        if ln.strip() and not is_bar(ln)]
                if not kept:
                    removed += 1
                    continue                       # drop the whole stderr output
                o['text'] = '\n'.join(kept) + '\n'
            new_outputs.append(o)
        cell['outputs'] = new_outputs
    if removed:
        json.dump(nb, open(path, 'w'), indent=1)
    return removed


def main():
    targets = sys.argv[1:] or sorted(
        glob.glob(os.path.join(ROOT, 'notebooks/**/*.ipynb'), recursive=True))
    total = 0
    for path in targets:
        n = clean(path)
        if n:
            print(f'{os.path.relpath(path, ROOT)}: removed {n} tqdm output(s)')
            total += n
    print(f'TOTAL: {total} tqdm output(s) removed' if total else 'no tqdm outputs found')


if __name__ == '__main__':
    main()
