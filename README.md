# ENM 3800: Learning from Data

Course materials for ENM 3800 at the University of Pennsylvania, Fall 2026.

## Repository Structure

```
docs/        All course documents: syllabus, schedule, project guide, homework specs, and instructor guides
notebooks/   Module Jupyter notebooks (Notebooks 1–5)
examples/    Supplemental Colab notebooks and narrative guides
```

## Documents (`docs/`)

| File | Contents |
| --- | --- |
| `Syllabus.md` | Student-facing syllabus: pod days, lab notebook, project tracks, homework, grading, and schedule |
| `ENM3800_Revised_Schedule.md` | Revised course schedule |
| `ENM3800_Project_Guide.md` | Project guide: tracks, pod format, lab notebook template, deliverables, and grading |

## Notebooks (`notebooks/`)

| Notebook | Topic |
| --- | --- |
| `Notebook_1/Notebook_1a_Overview_and_Data_Representation.ipynb` | Overview & Data Representation (Lecture 2) |
| `Notebook_1/Notebook_1b_Noise_and_Modeling_Data.ipynb` | Noise, Variability & Modeling Data (Lecture 3) |
| `Notebook_2/Notebook_2a_Building_a_Model.ipynb` | Model Building |
| `Notebook_2/Notebook_2b_Model_Selection.ipynb` | Model Selection |
| `Notebook_3/Notebook_3a_Probability_and_Covariance.ipynb` | Probability, Random Variables, and Covariance (Lecture 8) |
| `Notebook_3/Notebook_3b_Hypothesis_Testing.ipynb` | Hypothesis Testing and Confidence Intervals (Lecture 9) |
| `Notebook_3/Notebook_3c_Bootstrapping_and_Permutation_Tests.ipynb` | Bootstrapping, Permutation Tests, and Noisy Measurements (Lecture 10) |
| `Notebook_4/Notebook_4a_Regression.ipynb` | Regression |
| `Notebook_4/Notebook_4b_Classification.ipynb` | Classification |
| `Notebook_4/Notebook_4c_PyTorch_Model.ipynb` | Neural Networks |
| `Notebook_5/Notebook_5a_DimReduction.ipynb` | Dimensionality Reduction |
| `Notebook_5/Notebook_5b_Clustering.ipynb` | Clustering |

## Examples (`examples/`)

Supplemental notebooks for background concepts. These are standalone Colab notebooks students can open independently.

| File | Contents |
| --- | --- |
| `linear-algebra-primer-colab.ipynb` | Linear Algebra Primer (interactive Colab version) |
| `linear-algebra-primer-colab.md` | Linear Algebra Primer (narrative reference) |
| `eigenvectors-covariance-visualization-colab.ipynb` | Covariance and Eigenvectors (interactive Colab version) |
| `eigenvectors-covariance-visualization-colab.md` | Covariance and Eigenvectors (narrative reference) |

## Working with Notebooks (Jupytext)

Notebooks are paired with `py:percent` text files via [Jupytext](https://jupytext.readthedocs.io).
The `.py` file holds inputs only (no outputs) and is the clean thing to review and commit;
the `.ipynb` can be regenerated from it. Edit either file, then sync.

Dependencies are declared in `pyproject.toml` (notebook runtime deps under `dependencies`,
the Jupytext/exec tooling under the `dev` extra).

```bash
# one-time setup (per environment) — activate the venv, then install deps
source venv/bin/activate
pip install -e ".[dev]"

# pair a notebook with a percent script (creates the .py and stamps metadata)
jupytext --set-formats ipynb,py:percent "path/to/Notebook.ipynb"

# sync the pair after editing either file (propagates inputs, no execution)
jupytext --sync "path/to/Notebook.py"

# sync AND run every cell, writing fresh outputs into the .ipynb
jupytext --sync --execute "path/to/Notebook.py"
```

Notes:
- Activate the venv once per terminal session (`source venv/bin/activate`) and all commands
  work without any prefix. Run `deactivate` when done.
- `--sync` matches the pair by base filename in the same folder, so the `.ipynb` and `.py`
  must share a name (e.g. `Notebook_1_Learning_from_Data.{ipynb,py}`).
- `--execute` runs the notebook through the `python3` kernel and **stops on the first error**.
  Cells must therefore have all their dependencies installed in the venv first.

### Embedding outputs (plots & images)

The `.py` file stores inputs only, so the **`.ipynb` is what carries outputs** that viewers see
on GitHub/Colab. Re-render a notebook's outputs when you finalize it:

```bash
scripts/execute_notebooks.sh                                   # all notebooks
scripts/execute_notebooks.sh notebooks/Notebook_1/Notebook_1_Learning_from_Data.ipynb
```

This syncs, runs every cell, and writes the outputs back into the `.ipynb` (a `.py` path works
too). It uses `--allow-errors`, so one failing cell still lets the rest render. **Install the
full runtime deps first** (`pip install -e ".[dev]"`) — a missing package turns that cell's image
into an error traceback instead.

### Keeping pairs in sync (and outputs fresh) automatically

A [pre-commit](https://pre-commit.com) config (`.pre-commit-config.yaml`) provides two hooks:

```bash
source venv/bin/activate
pre-commit install                      # commit-stage: jupytext --sync
pre-commit install --hook-type pre-push # push-stage: execute changed notebooks
```

- **On commit** — `jupytext --sync` keeps each `.ipynb` and `.py` consistent. If a commit
  touches only one half of a pair, the hook reconciles the other and aborts — `git add` the
  updated files and commit again.
- **On push** (opt-in via the second command) — any changed notebook is executed so its outputs
  are embedded. If that changes the rendered outputs, the push aborts so you commit the
  re-rendered notebook, then push again. Requires the full runtime deps installed.

A GitHub Actions workflow (`.github/workflows/notebooks.yml`) enforces the **sync** check on
push/PR: it syncs all notebooks and fails if anything was out of date. It does **not** execute
the notebooks (their deps and dataset downloads make full execution slow and flaky in CI) —
execution stays a local step via the script / pre-push hook above.

### Troubleshooting: `'X.ipynb' and 'X.py' are inconsistent` (exit code 10)

The commit hook reports this when the two halves of a pair disagree *and* have the same
modification time (the usual case right after `git checkout`/`git add`), so jupytext can't
tell which one is newer and refuses to guess. It happens most often when a `.py` was created
with `--set-formats` (which keeps the original cell-metadata key order) while jupytext's
canonical form sorts those keys — the contents match, only the metadata ordering differs.

Fix by regenerating the `.py` from the `.ipynb` (the side with the outputs), then re-staging:

```bash
source venv/bin/activate

# for one notebook:
touch "notebooks/Notebook_X/<file>.ipynb"   # mark the .ipynb as the newer half
jupytext --sync "notebooks/Notebook_X/<file>.ipynb"

# or for every notebook at once:
find notebooks -name '*.ipynb' -exec sh -c \
  'touch "$1" && jupytext --sync "$1"' _ {} \;

git add -A   # then commit again
```

## Publishing: Slides & Textbook

The notebooks can be rendered two ways for the classroom, both driven by
[Quarto](https://quarto.org) (install the CLI once from quarto.org):

| View | Script | Output | Use |
| --- | --- | --- | --- |
| **Slides** | `scripts/render_slides.sh` | `_slides/` | reveal.js deck for presenting in lecture |
| **Textbook** | `scripts/render_book.sh` | `_book/` | scrollable site with sidebar nav + search, for student reference |

```bash
scripts/render_slides.sh                       # slides for every notebook
scripts/render_slides.sh notebooks/Notebook_1/Notebook_1_Learning_from_Data.ipynb   # one deck
scripts/render_book.sh                         # the whole textbook

# then view (works over an SSH tunnel: ssh -L 8080:localhost:8080 <host>):
python3 -m http.server 8080 --directory _book      # or _slides
```

Both views use the **outputs already embedded in the `.ipynb`** (Quarto runs with
`eval: false`), so run `scripts/execute_notebooks.sh` first if any plots are stale.
The `_slides/` and `_book/` directories are git-ignored — they are build artifacts,
rebuilt on demand.

### Why the scripts (and not plain `quarto render`)

Each notebook is a Jupytext pair (`.ipynb` + `.py` side by side). A normal
`quarto render` detects the pair, treats the `.py` as the source, then chokes on
the Colab cell metadata — so it silently finds **0 inputs**. The scripts work
around this by copying only the `.ipynb` files into a temporary build dir with the
Quarto config and rendering there. Don't run `quarto render` directly in this repo.

### Configuration

| File | Controls |
| --- | --- |
| `_quarto.yml` | slides: reveal.js theme, chalkboard, incremental bullets |
| `custom.scss` | slide font sizes — edit `$presentation-font-size-root` (default `26px`) to scale everything |
| `_quarto-book.yml` | textbook: theme, code visibility, section numbering (`number-depth`, `number-sections`) |
| `book.css` | textbook: hides Quarto's redundant chapter-number prefix |
| `index.qmd` | textbook landing page |

Notebook headings carry **no hard-coded section numbers** — Quarto numbers
sections automatically and continuously across the book, so renumbering is never
manual. Chapter titles are the topic only (e.g. "Regression and Regularization");
the module grouping lives in the book's Part headings in `_quarto-book.yml`.

### Deep-linking exercises to Colab

To give an exercise an "Open in Colab" badge that jumps **straight to that cell**
in Colab, give the exercise's markdown cell an id starting with `ex-`:

```python
# %% [markdown] id="ex-matrix-diagnose"
# ### Exercise: Diagnose a Matrix Transformation
# ...
```

then generate the badges (idempotent — safe to re-run; updates URLs in place):

```bash
python3 scripts/add_exercise_badges.py        # all notebooks (or pass one .py)
jupytext --sync notebooks/Notebook_X/<file>.ipynb
```

The script injects a badge linking to `…<notebook>.ipynb#scrollTo=<id>`. The id is
stored as the cell's `metadata.id`, which is what Colab's `scrollTo` matches. The
deep-link only does anything in Colab; in the book/GitHub it's a normal link to
the notebook.

## Weekly Rhythm

| Day | Activity |
| --- | --- |
| Monday | Homework due |
| Tuesday | Lecture |
| Thursday | Pod Day (75 min) |
| Friday | Lab Notebook Log due |
