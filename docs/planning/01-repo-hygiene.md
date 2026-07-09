# Phase 01 — Repo hygiene and configuration dedup

Goal: remove dead/duplicated config and untracked junk so later phases work
against a clean tree. Everything here is low-risk and mechanical. One commit
(or a few small ones) is fine.

## Tasks

### 1. Fix or delete `sanitize.sh` (broken since creation)
`sanitize.sh:6` sets `SAGING_DPATH` (typo, missing `T`), then lines 8 and 10
use the never-set `STAGING_DPATH`, so `mkdir -p` gets an empty arg and rsync
gets an empty destination. Fix: rename the variable to `STAGING_DPATH` on
line 6. If the script has no remaining purpose, delete it instead.

### 2. Single-source the pytest config
`pytest.ini` and `pyproject.toml [tool.pytest.ini_options]` are duplicated
and have diverged (pyproject has `--ignore-glob=setup.py`, a `_skbuild`
norecursedirs entry, and a third filterwarnings line that pytest.ini lacks).
pytest.ini silently wins. **Delete `pytest.ini`** and reconcile any useful
divergence into the pyproject block. Verify afterwards that
`python3 -m pytest tests/ -q` still picks up `--xdoctest` addopts (run with
`-q --collect-only | tail -3` and confirm doctests collect when
`--xdoctest kwarray/` is passed).

### 3. Delete stale `.coveragerc`
CI and `run_tests.py` pass `--cov-config pyproject.toml`; the tracked
`.coveragerc` is unused there and contains relics (`if six.PY2:` exclude,
omit entries `kwarray/tests/*` and `kwarray/_win32_links.py` — neither path
exists). Delete it, keep the `[tool.coverage.*]` blocks in pyproject.toml.

### 4. Delete stale `docs/requirements.txt`
`.readthedocs.yml` uses `requirements/docs.txt`. The tracked
`docs/requirements.txt` is an unversioned, incomplete duplicate (missing
myst_parser, sphinx-reredirects, xdoctest, kwplot[headless]). Delete it, or
replace its contents with the single line `-r ../requirements/docs.txt` if
some external tooling references it. Also add a trailing newline to
`requirements/docs.txt`.

### 5. Resolve the `kwarray/_backend/` stub
Untracked directory containing only an empty `__init__.py` plus a stale
`__pycache__` with compiled `pebaystats` bytecode (its source now lives at
`dev/poc/pebaystats.py`, whose doctests still import
`kwarray._backend.pebaystats`). Because `setup.py` uses `find_packages(".")`,
any wheel built from this tree ships an empty `kwarray._backend` package.

Action now: **delete `kwarray/_backend/` entirely.** The decision about
whether to create a real `_backend` package (for pebaystats / Welford stats)
belongs to phase 09 — record the pointer there, don't keep the empty stub.
If `dev/poc/pebaystats.py` is kept, fix its doctest imports to reference its
actual location.

### 6. Untracked junk and `.gitignore`
- Add `git-of-theseus/` to `.gitignore` (generated analysis artifacts) or
  delete the directory.
- `dev/benchmarks/benchmark_assignment_algos.py` and
  `dev/poc/{discrete_histograms.py,pebaystats.py}` are real code sitting
  untracked. Commit them under `dev/` (preferred — they're referenced by
  TODOs) or delete them. If committing pebaystats.py, fix its
  `kwarray._backend.pebaystats` doctest imports first.
- Optional cleanup: remove local `htmlcov/`, `.coverage`, `kwarray.egg-info/`
  (already gitignored).

### 7. MANIFEST.in
Add `include CHANGELOG.md` so the sdist ships the changelog.

### 8. Retire `appveyor.yml` and its README badge
The matrix is Python 3.8-only and its `torch==1.7.1+cpu` pin is immediately
undone by the following `pip install -r requirements.txt -U` line. GitLab CI
covers the full matrix. Delete `appveyor.yml` and remove the Appveyor badge
from `README.rst` (~line 195). (If the maintainer wants Windows CI, that is
a separate work item — note it in the commit message.)

### 9. Remove leftover debug prints in `algo_setcover.py`
`kwarray/algo_setcover.py:257-258, 268-269, 298-299, 301` — bare debug
`print()` calls in `_setcover_greedy_new` dump `avail_covers = ...` /
`CHOOSE COVER SET = ...` to stdout on every call. Delete them. (Functional
bugs in this module are handled in phase 04; this is pure noise removal.)

### 10. Remove import-time "Warning, no params" prints in `distributions.py`
`kwarray/distributions.py:551` prints "Warning, no params" on every
construction of `Mixture`, `PDF`, and `NonlinearUniform`. Convert to
`warnings.warn(..., stacklevel=2)` or remove; the real fix (declaring
`__params__` for those classes) is phase 03/09 scope.

## Acceptance criteria
- `git status` shows no untracked files that aren't intentionally local.
- Exactly one pytest config and one coverage config exist (in pyproject.toml).
- `python3 -m pytest tests/ --xdoctest kwarray/ -q` green.
- `python3 -c "import kwarray, io, contextlib"` prints nothing to stdout.
- Fresh `python3 -m build` (if available) produces a wheel WITHOUT
  `kwarray/_backend` and an sdist WITH `CHANGELOG.md`.
