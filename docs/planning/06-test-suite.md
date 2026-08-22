# Phase 06 — Test suite overhaul

Prereq: phases 02–05 (tests must encode fixed behavior). Baseline at audit
time: `python3 -m pytest tests/ --xdoctest kwarray/ -q` → 125 passed, 61
skipped. 10 of 15 modules have NO dedicated file in `tests/`; they rely
entirely on doctests, many of which are smoke-only or gated behind optional
deps that CI may not install for every job.

## 1. Fix the two dead tests (do this first — they are silent CI holes)

- `tests/test_arrayapi.py::test_numpy_torch_compat` **cannot fail**: it
  accumulates `errors.append(row)` but never asserts. Add
  `assert not errors, f'inconsistent ArrayAPI results: {errors}'` at the
  end. Also: the `basis` grid (`shape`, `dtype`) is iterated but `item` is
  never used — every iteration re-tests the same unseeded float64
  `rng.rand(3, 5)`; actually use the grid params and seed the rng
  (`kwarray.ensure_rng(0)`). This test is the acceptance harness for phase
  05 — extend its op/dtype coverage per that doc.
- `tests/test_averages.py::combine_mean_std_zero_nums` is **never
  collected** (missing `test_` prefix). It is a real regression test for a
  Bessel-correction NaN bug and passes when run manually. Rename to
  `test_combine_mean_std_zero_nums`.

## 2. Strengthen weak tests

- `tests/test_averages.py::test_with_nan_case`: zero assertions (four
  updates + prints). Compute the expected means/maxes by hand and assert.
- `tests/test_averages.py::test_fuzzed_random_running`: unseeded
  `ensure_rng()` — failures would be unreproducible. Seed it (or derive
  from an env var and print the seed on start).
- `tests/test_normalize.py::test_normalize_grid`: asserts only
  shape/dtype/out-identity; builds an unused `results` list. Add value
  assertions for at least the simple linear cases, and pin down the defined
  behavior for empty/all-NaN → int-dtype `out=` (see phase 02 item 19).
- `tests/test_slider.py::test_stitcher_grid`: collects results, asserts
  nothing about stitched values. Assert the stitched output equals the
  input for identity stitching, and correct averaging on overlaps.

## 3. Remove stale doctest gates

- `kwarray/util_random.py:176,194`: drop `+REQUIRES(module:scipy)` on the
  `random_combinations` doctests (function uses `math.comb`); replace
  `+IGNORE_WANT` with assertions (phase 02 item 25).
- `kwarray/algo_setcover.py:60,315,322`: convert old-style
  `# xdoc: +REQUIRES(module:pulp)` prefixes to `# xdoctest:` for
  consistency.
- `kwarray/util_torch.py:180`: the `module:onnx, module:onnx_tf` gate is
  permanently dead (onnx_tf is unmaintained) — delete the example or the
  feature (flag to maintainer if the onnx export path is still wanted).

## 4. New dedicated test files (priority order)

For each, cover the phase 02–04 regression cases plus the listed gaps:

1. `tests/test_algo_assignment.py` — currently **5% coverage, zero
   locally-executed verification** (all 8 doctests are scipy-gated).
   - scipy-free tests for `_sparse_assignment_networkx` (needs only
     networkx; the function currently has NO doctest and is never exercised
     anywhere).
   - scipy-gated: property test `sparse_assignment` vs dense reference on
     random small instances, negative values, maximize on/off, empty
     inputs (phase 04 items 1, 2, 4).
2. `tests/test_dataframe_light.py` — pandas-parity canary (pandas 3.x!).
   Cover `__eq__`, `union` with empty frames, `to_dict` orients vs real
   pandas, `.values` on empty, `DataFrameArray.append/clear/extend`,
   `rename` column order, `groupby`, `drop`, `from_dict` round-trips.
3. `tests/test_util_random.py` — `random_combinations` (num=None fast
   path), `random_product` exact cardinality (`999**3` case), `seed_global`
   (numpy side), `random_unique_integers` strategies.
4. `tests/test_fast_rand.py` — seeded distributional tests for
   `standard_normal{,32,64}` incl. odd sizes and size-1 (phase 02 items
   20–21), dtype handling.
5. `tests/test_util_numpy.py` — `equal_with_nan` vs
   `np.isclose(equal_nan=True)`, `argmaxima/argminima` symmetry over a
   param grid (num vs size, ordered on/off, axis), `boolmask` empty,
   `isect_flags` vs `np.isin`.
6. `tests/test_util_groups.py` — `group_indices` for int arrays, string
   arrays, tuple keys (int tuples — phase 02 item 26), 2D ndarray input;
   `group_consecutive`, `apply_grouping` edge cases.
7. `tests/test_algo_setcover.py` — max_weight budget respected (phase 04
   item 5), greedy vs ILP parity gated on pulp, `items` argument semantics.
8. `tests/test_util_slices.py` — `padded_slice` deep-pad branches,
   `apply_embedded_slice` with trailing channel dims (phase 02 item 27).
9. `tests/test_util_torch.py` (torch-gated) — `one_hot_embedding` negative
   dims vs positive equivalents on both backends, `one_hot_lookup`.
10. `tests/test_util_misc.py` — `FlatIndexer.unravel` bounds/negatives,
    scalar-vs-array path equivalence.

Also extend `tests/test_distributions.py` per phase 03's verification
section (scipy-agreement, seeding reproducibility, no-scipy import test) and
`tests/test_slider.py` per phase 04 (memmap str path, stride validation,
nan_policy spellings).

## 5. Coverage targets and CI

Audit-time coverage (torch/scipy absent — CI with full deps will be higher
for arrayapi/util_torch): algo_assignment 5%, util_torch 10%, arrayapi 42%,
algo_setcover 53%, util_robust 55%, dataframe_light 57%, distributions 59%,
util_random 66%, util_averages 73%.

- Target: no module below 75% line coverage in the full-deps CI job; add
  `--cov-fail-under` per-run once achieved (start global at 80%).
- Add pulp to the full-deps CI extras so `_setcover_ilp` runs at least
  somewhere (phase 04 item 6).
- Add ~21 smoke-only doctests real assertions opportunistically as their
  modules are touched (list in the audit: sparse_assignment Examples 2–3,
  util_robust.py:316,530,548, distributions.py:310,789,914,930,1040,1193,
  util_averages.py:118,130,621,977, util_numpy.py:686, arrayapi.py:56,410,
  dataframe_light.py:50, util_random.py:533, util_slices.py:59).

## Acceptance criteria
- `python3 -m pytest tests/ --xdoctest kwarray/ -q` green in both minimal
  (numpy-only) and full-deps environments.
- Zero tests that cannot fail (grep for test functions without asserts).
- All fuzz/statistical tests seeded and reproducible.
- Coverage floor enforced in CI.
