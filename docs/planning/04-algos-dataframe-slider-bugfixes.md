# Phase 04 — Confirmed bugs in algo_assignment, algo_setcover, dataframe_light, util_slider

All items empirically verified unless marked otherwise. scipy is needed to
exercise the assignment doctests; pandas for the dataframe comparisons.

## algo_assignment.py

### 1. `sparse_assignment` wrong for negative costs
`algo_assignment.py:490, 539`: `approx_inf = 2 * (n + values.max())` can be
small/negative when costs are negative, and the feasibility filter
`if cost < approx_inf:` then rejects legitimate assignments. Verified:
- `sparse_assignment([0], [0], [-10.0])` → `([], 0.0)`; correct
  `([(0, 0)], -10.0)`.
- `sparse_assignment([0,1], [0,1], [-100.0, -50.0])` → one pair instead of
  two (dense `mincost_assignment` returns both, total -150).

Fix: do not infer feasibility from magnitude. Record the actual set of real
`(i, j)` pairs and filter the solver output on membership. (If keeping the
sentinel approach as a fallback, at minimum
`approx_inf = 2 * (n + np.abs(values).max())` with an empty guard — but
membership filtering is strictly better.)

### 2. Dense-fallback unpack crashes on empty assignment
`algo_assignment.py:529`: `row_ind, col_ind = list(zip(*_assigns))` with
empty `_assigns` → ValueError (verified with a square instance that has no
perfect matching under `maximize=True`). Fix: early-return `([], 0.0)` when
`_assigns` is empty.

### 3. `mincost_assignment` overflow guards
Costs near float max make `feasible_extent` overflow to inf with
RuntimeWarnings (observed). Clamp with `np.isfinite` guards. Low priority,
same file visit.

### 4. maximize semantics inconsistency **[DECISION]**
`sparse_assignment(..., maximize=True)` keeps negative-value assignments
when the sparse solver succeeds, but its dense fallback
(`maxvalue_assignment`) documents dropping non-positive pairs — the same
inputs can produce different results depending on the internal path. Pick
one semantic (recommend: keep all assignments returned by the solver, i.e.
the sparse behavior, since callers can filter) and enforce it as a
post-filter on both paths; document it.

## algo_setcover.py

### 5. `setcover(..., max_weight=...)` violates its documented constraint
`algo_setcover.py:127-137`: the loop checks the budget BEFORE adding a set,
so the last chosen set can exceed `max_weight` (verified: budget 3, returned
cover weight 5). Fix: skip candidates whose weight would exceed the
remaining budget and continue to the next-best candidate; only stop when no
candidate fits.

### 6. pulp 3.x compat in `_setcover_ilp` (unverified — pulp not installed)
`algo_setcover.py:357, 389-393`: `pulp.LpVariable.dicts(..., indexs=...)`
— the `indexs` kwarg was removed in pulp 3.x (renamed `indices`). Pass
positionally or try/except the kwarg name. Also rename LpProblem titles to
avoid spaces ("Set Cover" → "SetCover") to silence pulp warnings. Add pulp
to `requirements/tests.txt` or at least to a CI job so this path is ever
exercised (coordinate with phase 06).

### 7. `items` argument ignored by the greedy path
`algo_setcover.py:78-144`: `setcover(..., items=...)` only affects the ILP
path; the greedy algorithm ignores it (verified no effect). Either honor it
(restrict the universe) or document that it is ILP-only.

### 8. `_setcover_greedy_new` is unused
After phase 01 removes its debug prints, decide: wire it in as the `approx`
implementation (it is reportedly the better implementation) with tests
proving parity on the existing doctests, or delete it. Don't leave both
greedy implementations rotting.

## dataframe_light.py

Context: pandas 3.0.2 is current; DataFrameLight claims loose pandas parity.
Each fix should include a doctest comparing against real pandas gated with
`# xdoctest: +REQUIRES(module:pandas)`.

### 9. `__eq__` broken for DataFrameLight vs DataFrameLight
`dataframe_light.py:197-202`: missing `elif` — the pandas-instance check
overwrites `other_vals` computed for the DataFrameLight case, so
light == light raises "truth value is ambiguous" (verified). Fix: `elif`.

### 10. `union()` silently drops all data when self is empty
`dataframe_light.py:576-580`: `if not both.keys:` tests a bound method
(always truthy) → the adopt-keys branch is dead, and empty.union(full)
returns 0 rows (verified; pandas concat keeps them). Fix: test
`len(both._data) == 0`, and copy the `_data` keys from the first non-empty
frame (do not assign over `.keys`).

### 11. `to_dict(orient='dict')` implements pandas' 'index' orient
`dataframe_light.py:232-233` (implementation around 514–522): light returns
`{row: {col: val}}` for `'dict'`; pandas `'dict'` is `{col: {row: val}}`
(verified side-by-side). Fix: implement `'dict'` correctly and route
`'index'` to the current behavior. **[DECISION]** This changes existing
output for `'dict'` callers — CHANGELOG "Changed" entry.

### 12. `.values` crashes on empty frames
`dataframe_light.py:396-400`: `DataFrameLight({}).values` and
all-empty-column frames raise ValueError (verified; pandas returns an empty
(0, n) array). Fix: early-return `np.empty((len(self), len(self.columns)))`
for the empty cases; use `reshape(num, -1)` only when non-empty.

### 13. `DataFrameArray.append`/`clear` crash (list-column assumptions)
`dataframe_light.py:408-414, 524-526`: `DataFrameArray` overrides
extend/compress/take but inherits list-based `append`/`clear` →
AttributeError on ndarray columns (verified). Fix: override both
(`np.concatenate` for append; `arr[:0]` for clear) or raise
NotImplementedError with a message pointing at `extend`.

### 14. Smaller dataframe_light items
- `rename` moves renamed columns to the end (verified; pandas preserves
  order) — rebuild the dict preserving order (`dataframe_light.py:655-685`).
- `concat([])` / bare `union()` raise a confusing TypeError — return an
  empty frame.
- Line ~639 non-executed doctest uses stale `np.int`.
- `LocLight` has no docstring (also phase 08).

## util_slider.py

### 15. `Stitcher(shape, memmap='<dir>')` crashes
`util_slider.py:412-433`: when memmap is a str/PathLike, the fpath locals
are only assigned in the tempdir branch → UnboundLocalError (verified).
Fix: construct `memmap_sums_fpath`/`memmap_weights_fpath` (and
`self.paths`) for both branches.

### 16. `nan_policy='propagate'` (correct spelling) rejected
`util_slider.py:381, 449-457`: only the misspelling `'propogate'` is
accepted; the correct spelling raises ValueError (verified). Fix: accept
`'propagate'` as canonical, keep `'propogate'` as a deprecated alias with a
DeprecationWarning — mirror the treatment in `RunningStats` (phase 02 item
7). Fix the doctest comment at 360–361 too.

### 17. `SlidingWindow._iter_basis_frac` references nonexistent attribute
`util_slider.py:211-214`: `self.source` does not exist; any call raises
(verified). Fix to `self.input_shape` or delete the method (it appears
unused — grep first; deleting dead code is preferred).

### 18. Negative or non-integer stride: infinite hang / garbage slices
`util_slider.py:203-205, 596-611`: `if not all(stride)` only rejects 0.
`stride=-1` hangs forever in `_slices1d` (verified, killed at 5s);
`stride=2.5` yields `slice(2.5, 7.5)` which crashes on real indexing.
Fix: validate `all(isinstance(s, numbers.Integral) and s > 0 for s in
stride)` and raise ValueError (the message at 205 already says "Step must be
positive everywhere").

### 19. `_slices1d` doctest passes keepbound into start
`util_slider.py:561-564`: `_slices1d(margin, stop, step, keepbound,
check=False)` binds True to `start`. The doctest exercises nothing it
claims (verified: yields `slice(True, 361)`). Fix the call to use
`keepbound=keepbound`, and strengthen the assertion to check right-boundary
coverage (the 1801–2000 region).

### 20. Torch paths in `Stitcher` (unverified — torch not installed)
- `util_slider.py:445-446`: torch branch ignores the `dtype` argument.
- `util_slider.py:481-484`: `nan_policy='omit'` on torch tensors uses numpy
  APIs (`.astype`, `.copy`) → will crash on device tensors.
Fix both when a torch env is available (coordinate with phase 05
verification environment).

### 21. `stride > window` silently leaves coverage gaps
`util_slider.py:199`: recomputed `overlap` can go negative (verified
`overlap=[-1.0]`) although the docstring promises `0 <= overlap < 1`.
Decide: allow gaps with an explicit warning, or raise. Document either way.

## Verification
- New unit tests per bug (phase 06 specifies target files).
- Property test for `sparse_assignment`: on random small dense instances
  (including negative values and `maximize` both ways), its result must
  match `mincost_assignment`/`maxvalue_assignment` totals.
- `python3 -m pytest tests/ --xdoctest kwarray/ -q` green with pandas and
  scipy installed.
