# Phase 07 — Packaging and Python-support matrix

Context: the project claims Python 3.8–3.14 (`setup.py` classifiers,
`python_requires>=3.8`), CI is GitLab (`.gitlab-ci.yml` is the source of
truth), and packaging metadata is xcookie-generated. Where a change belongs
in xcookie config (`pyproject.toml [tool.xcookie]`), prefer changing it
there and regenerating; otherwise hand-edit and note it.

## 1. Decide Python 3.8's fate **[DECISION]** — recommended: DROP

Evidence of drift: commit 609f527 changed `requirements/tests.txt` so no
pytest entry applies on 3.8 at all (the 4 cp38 GitLab jobs only work because
pytest-cov transitively drags in an unpinned pytest). Python 3.8 has been
EOL since Oct 2024, and keeping it alive costs compat shims in arrayapi
(functools.cache fallback etc.) and a scipy fallback in util_random.

If dropping (one commit):
- `pyproject.toml [tool.xcookie]` min_python → 3.9 (or higher), regen.
- `setup.py` classifiers + `python_requires`.
- Remove the four cp38 jobs from `.gitlab-ci.yml` and the cp38 entry in the
  `gpgsign` needs list (~line 588).
- Prune dead `python_version < '3.9'` marker rows across `requirements/`.
- Delete the scipy `comb` fallback in `util_random.py` and the
  `functools.cache` fallback in `arrayapi.py:68-71`.

If keeping: restore a 3.8 pytest marker in `requirements/tests.txt`
(`pytest>=6.2.5 ; python_version < '3.9' and python_version >= '3.8'`).

## 2. Replace nptyping with numpy.typing

`requirements/optional.txt:36-41` pins nptyping, which is unmaintained
(since 2023) and whose metadata pins `numpy<2.0` — so
`pip install kwarray[optional]` on 3.11/3.12 downgrades numpy to 1.x, and
the GitLab `full-loose` cp311/cp312 jobs never actually test numpy 2.x.
There is no nptyping row for 3.13/3.14 at all, yet every `.pyi` stub imports
`from nptyping import ...` — the stubs are unresolvable on 3.13+.

Work:
- Rewrite all `.pyi` stubs against `numpy.typing.NDArray` (grep:
  `grep -l nptyping kwarray/*.pyi`).
- Delete the nptyping rows from `requirements/optional.txt`.
- If docstring types reference nptyping names, update those too (the Sphinx
  type-parsing config in `docs/source/conf.py` may map them — check).
- Verify: `pip install -e .[optional]` on py3.12 keeps numpy ≥2;
  `python3 -m mypy --ignore-missing-imports kwarray/` (or pyright) parses
  the stubs without unresolved-import errors.

## 3. Requirements file hygiene

- Prune dead marker rows for Python 2.7/3.4/3.5/3.6/3.7 across
  `requirements/*.txt` (they can never fire with python_requires>=3.8).
- Fix the wrong comment in `requirements/tests.txt:2` ("Python 3.9-11"
  actually covers 3.9–3.10).
- `requirements/optional.txt:14` `torch>=2.9.0 ; python_version == '3.14'`:
  verify the cp314 full jobs are actually green (torch/scipy/pandas cp314
  wheels must all exist on the runner's index) before the 0.7.2 release.

## 4. Adopt PEP 621 metadata

`pyproject.toml:5-6` still says `# TODO: [project]`; all metadata lives in
generated `setup.py`. Regenerate with a modern xcookie that emits a
`[project]` table (check `xcookie --help` / current xcookie docs). This is
the right moment because phases 01/07 already touch the same files. If
xcookie can't do it yet, defer — do NOT hand-roll a `[project]` table that
xcookie will clobber.

## 5. CI matrix consistency check (after items 1–4)

Write a tiny checker (or do it manually) confirming these agree:
- `setup.py` classifiers / python_requires,
- `requirements/*.txt` markers,
- `.gitlab-ci.yml` job matrix,
- README badges.
Remove the Appveyor badge/config if not already done in phase 01.

## Acceptance criteria
- `pip install -e .[all]` succeeds on the oldest and newest supported
  Python with numpy ≥2 retained.
- Stubs import-clean under a type checker on 3.13+.
- CI matrix, classifiers, and requirements markers name the same version
  set.
- sdist/wheel build clean (`python3 -m build`) and contain no `_backend`
  stub, no junk, plus CHANGELOG.md (from phase 01).
