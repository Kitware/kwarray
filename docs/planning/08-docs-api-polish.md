# Phase 08 — Documentation and API-surface polish

Prereq: phases 02–05 landed (docs must describe fixed behavior).

## 1. Regenerate `kwarray/__init__.py` with mkinit

`sparse_assignment` is public in `algo_assignment.py` but missing from the
mkinit-generated export block (verified: everything else in `__all__`
resolves). Run `mkinit kwarray --relative --nomods -w`, diff carefully, and
confirm no other new publics were missed. Update the `.pyi` for
`__init__` if one exists.

## 2. Docstring corrections (all verified against signatures)

- `kwarray/util_slices.py:24` — `padded_slice` documents a phantom `ndim`
  parameter that does not exist in the signature. Remove it.
- `apply_embedded_slice` — `padkw` undocumented; add a doctest (it has
  none).
- `kwarray/util_numpy.py` `unique_rows` — `return_index` undocumented, no
  Returns section despite the return type changing to a tuple.
- `iter_reduce_ufunc` — `out` undocumented. `uniform32` — `rng`
  undocumented.
- `dataframe_light.py:23` `LocLight` — exported publicly with NO docstring;
  write one. `NoSupportError` — expand the one-line stub.
- `seed_global` — add a doctest showing it seeds the global numpy state
  (and torch when present); it currently has none.
- `util_slider.py:157-160` — `_compute_stride` docstring typo
  ("hasoverlap").
- `kwarray/__init__.py` module docstring says "extensions to numpy and
  torch" — add one sentence noting torch is optional.

## 3. README.rst refresh

- The "top-level API" block (lines ~31–54) is stale: `normalize` is listed
  under util_numpy (it lives in util_robust), and 0.7.x additions are
  missing (`RunningHistogram`, `nanquantile`, `robust_normalize`,
  `find_robust_normalizers`, `equal_with_nan`, `generalized_logistic`,
  `unique_rows`, `Slice`, `apply_embedded_slice`,
  `random_unique_integers`, `sparse_assignment`, `NoSupportError`).
  Regenerate the block from the fresh `__init__.py` (the
  `dev/gen_api_for_docs.py` script may help — check it first).
- Remove the stale `# TODO Get CI services running on gitlab` comment at
  the top (GitLab CI is fully operational).
- Remove the Appveyor badge if phase 01 deleted appveyor.yml.
- Add a short "Development" section: how to run tests
  (`python3 -m pytest tests/ --xdoctest kwarray/`), where requirements
  live, that `__init__.py` is mkinit-generated, that `.gitlab-ci.yml` is CI
  truth.

## 4. Add CLAUDE.md (agent/contributor onboarding)

Contents (keep under ~60 lines):
- Test invocation and the addopts that make xdoctest google-style work.
- The optional-dependency doctest-gating convention
  (`# xdoctest: +REQUIRES(module:X)`) and the rule that `import kwarray`
  must never require torch/scipy/pandas.
- `__init__.py` is mkinit-generated; regenerate, don't hand-edit.
- `setup.py`/CI scaffolding is xcookie-generated from
  `pyproject.toml [tool.xcookie]`.
- `.pyi` stub convention (update stubs when signatures change).
- CHANGELOG.md discipline (every user-visible change).
- Pointer to docs/planning/ for the improvement roadmap.

## 5. Sphinx docs sanity

- `docs/source/auto/*.rst` covers every module (verified at audit) — after
  any module add/remove, regenerate.
- Confirm the RTD build works after phase 07's requirements changes
  (`.readthedocs.yml` uses `requirements/docs.txt`).
- `docs/source/function_usefulness.rst` is a nice asset — refresh the
  usage-frequency table if `dev/count_usage_freq.py` still runs.

## 6. CHANGELOG.md

Consolidate all phase 02–05 entries under `0.7.2` (or bump to 0.8.0 given
the behavior changes — **[DECISION]**: semver would argue for 0.8.0 since
`equal_with_nan`, sigmoid `beta`, `to_dict('dict')`, and `Uniform` param
order all change observable behavior).

## Acceptance criteria
- `python3 -c "import kwarray; assert 'sparse_assignment' in kwarray.__all__"`.
- No public export without a docstring
  (`python3 -c "import kwarray; missing=[n for n in kwarray.__all__ if not getattr(kwarray, n).__doc__]; assert not missing, missing"`).
- README API block matches `kwarray.__all__`.
- Docs build green on RTD config.
