# Phase 02 — Confirmed bugs in core numeric utilities

Scope: `util_numpy.py`, `util_averages.py`, `util_robust.py`, `fast_rand.py`,
`util_groups.py`, `util_slices.py`, `util_misc.py`, `util_random.py`.

Every item below was empirically reproduced (numpy 2.4.4, Python 3.12).
For each fix: add a regression test (doctest or `tests/`), update any doctest
that encodes the buggy output, and add a CHANGELOG entry. Items marked
**[DECISION]** change numeric output for existing callers — implement the fix
but call it out prominently in the CHANGELOG.

## util_numpy.py

### 1. `equal_with_nan` returns inverted results for non-NaN comparisons **[DECISION — behavior change]**
`kwarray/util_numpy.py:768`: `flags = value_sameness ^ nan_sameness` is wrong.
Two equal non-NaN values → `value_sameness=True`, `nan_sameness=True` → XOR
→ `False`. Unequal values → `True`. Verified:
`equal_with_nan([1.],[1.]) -> [False]`, `equal_with_nan([1.],[2.]) -> [True]`.
Only NaN-vs-NaN (True) and NaN-vs-number (False) are correct today.

Fix: `flags = value_sameness | (a1nan & a2nan)`. The doctest at lines
746–761 encodes the broken behavior (expects `False` where both values are
0) — update its expected values. Add a direct unit test comparing against
`np.isclose(a, b, equal_nan=True)` on float data.

### 2. `argminima(arr, num, ordered=False)` returns ALL indices for 2D input
`kwarray/util_numpy.py:454`: the `axis=None` branch uses
`elif num < len(arr):` where `len(arr)` is `shape[0]`, not `arr.size`
(`argmaxima` correctly uses `arr.size` at line 274). For 2D input with
`shape[0] <= num < size`, `ordered=False` returns `np.arange(arr.size)` —
every index. Fix: `elif num < arr.size:`. Add a test:
`len(argminima(np.random.rand(3,10), num=5, ordered=False)) == 5`.

### 3. `boolmask([])` without `shape` crashes
`kwarray/util_numpy.py:47`: `indices.max()` on empty input →
`ValueError: zero-size array to reduction`. Guard:
`shape = 0 if indices.size == 0 else indices.max() + 1` (mind dtype of the
empty asarray — cast indices with `astype(int)` first if needed).

### 4. `isect_flags` is O(len(other) x arr.size) via a Python generator
`kwarray/util_numpy.py:144`: replace the generator-reduce with
`np.isin(arr, other)` (NOT `np.in1d`, removed in numpy 2.x). Confirm the
doctests still pass; behavior should be identical.

### 5. Dead code cleanup
`util_numpy.py:536-542` — `arr_view_unique` computed then unused in
`unique_rows`'s ordered branch; commented-out `unique_axis` /
`argsort_threshold` blocks at 472–500 and 612–638. Delete (move to `dev/`
only if there is real intent to finish them).

## util_averages.py

### 6. `stats_dict(..., quantile=True)` crashes
`kwarray/util_averages.py:212`: `quantile == 'auto'` is a comparison used as
a statement (should be assignment `quantile = 'auto'`), so `quantile=True`
later iterates over `True` → `TypeError`. Verified. Fix the assignment,
collapse the duplicated `if quantile == 'auto': quantile = [...]` at 214–215
and 224–225 into one place, and additionally accept a scalar float
(`quantile=0.5` currently also crashes; wrap scalars in a list — the
docstring types the param `str | bool | List[float]`).

### 7. `RunningStats.__init__` nan_policy validation broken
`kwarray/util_averages.py:494-501`:
- valid set is `{'omit', 'propogate'}` — misspelled, and missing
  `'raise'` even though `update()`/`update_many()` implement a `'raise'`
  path (lines 582, 657) that is unreachable from the constructor.
- `RunningStats(nan_policy='raise')` and `nan_policy='propagate'` (correct
  spelling) both raise `KeyError`; the misspelling `'propogate'` is accepted.
- the inner `check_weights` guard at 495–500 sits inside the invalid-value
  branch so it never fires.

Fix: validate against `{'omit', 'propagate', 'raise'}`; accept `'propogate'`
as a deprecated alias that is normalized to `'propagate'` at the door (emit
`DeprecationWarning`); update the string comparisons at lines 580 and 655 to
the correct spelling; move the check_weights guard out of the error branch so
it actually runs. **[DECISION]** — keep the alias for at least one minor
release; coordinate with the identical misspelling in `util_slider.py`
(phase 04, item 15).

### 8. `RunningStats.update_many` crashes with scalar weights under propagate policy
`kwarray/util_averages.py:615`: `run.n += weights.sum(axis=0)` assumes
ndarray; with default `weights=1` and non-omit policy, weights stays `int`
→ `AttributeError`. Fix: normalize `weights = np.asarray(weights)` up front
(mirroring `update()`'s handling).

### 9. `RunningStats.update` with integer data + zero weight → OverflowError
`kwarray/util_averages.py:678`: `data[ignore_flags] = -np.inf` fails on int
arrays. `update_many` already solves this with `_numpy_dtype_info(data.dtype)`
(lines 593–608). Port the same sentinel approach into `update`. Better:
extract the shared nan-policy/weights/ignore handling from `update` and
`update_many` into one helper so the two paths cannot diverge again (they are
near-duplicates today).

### 10. `RunningHistogram._update_other` references nonexistent `.counts`
`kwarray/util_averages.py:1265`: `self.accum += other.counts` — the class
stores counts in `.accum`. `hist1 += hist2` raises AttributeError. Fix:
`self.accum += other.accum`. Add a merge test.

### 11. `RunningDiscreteHistogram.__iadd__` references nonexistent `.values`
`kwarray/util_averages.py:1379`: `if other.values.size:` → AttributeError;
should be `other.bins`. Also note the merge at line 1381 does
`np.repeat(other.bins, other.counts)` — O(total observations) memory for what
should be an O(n_bins) merge; rewrite to merge (bins, counts) pairs directly.

### 12. `_gmean` broken for `axis != 0`
`kwarray/util_averages.py:405`: `mem = log_a[axis]` indexes position `axis`
along axis 0 and is then used as an aliasing `out=` buffer for the reduction
→ broadcast ValueError for axis=1 on 3D input (verified). The in-place
`out=` trick also writes into a view of the array being reduced (unspecified
numpy behavior) even for axis 0. Fix: drop the aliased-out trick; compute
`log_a.mean(axis=axis)` normally. Verify against `scipy.stats.gmean` for
axis 0 and 1 in a test gated on scipy.

### 13. `stats_dict` mixed-precision output
`kwarray/util_averages.py:218-222`: mean/std/min/max are forced through
`np.float32(...)` regardless of input dtype, while median/quantiles stay
float64. Fix: preserve input float precision (use the array's dtype if
floating, else float64). Also fix the shape inconsistency: `median`,
quantiles, and `sum` ignore `keepdims` while mean/std/min/max honor it
(lines 227–254) — thread keepdims through all of them.

## util_robust.py

### 14. Tukey fence upper bound uses q1 instead of q3
`kwarray/util_robust.py:197`: `fence_upper = q1 + 1.5 * iqr` — standard
Tukey upper fence is `q3 + 1.5 * iqr`. Verified: for data 0..100 step 10 the
returned upper fence is 100.0; correct is 150.0. Fix the formula, add a test
with a textbook example.

### 15. `_custom_quantile_extreme_estimator` never clips high outliers
`kwarray/util_robust.py:261`: `max_val = max(quant_high_abs, quant_high_val - high_pad_val)`.
Since `high_pad_val >= 0`, the max always resolves to `quant_high_abs`, the
raw maximum — one extreme outlier dominates the default
`'adaptive-quantile'` normalization (verified: 10k uniforms + one 1e6
outlier → max_val = 1e6). The code carries a "todo: verify... not 100%
sure" comment. By symmetry with the min side, fix:
`max_val = min(quant_high_abs, quant_high_val + high_pad_val)`.
Add a test asserting the outlier is clipped.

### 16. `robust_normalize(..., axis=..., return_info=False)` silently ignores axis
`kwarray/util_robust.py:404-407`: the per-axis branch computes `final` but
only returns it when `return_info=True`; otherwise control falls through to
a joint whole-array normalization (wrong result AND wasted compute).
Verified. Fix: return `final` in both cases from the per-axis branch.

### 17. `find_robust_normalizers(data, params='std')` crashes
`kwarray/util_robust.py:135-136`: the `'std'` branch is `pass`, leaving
params a string → `ub.dict_union` AttributeError. Either implement a
std-based mode (mean ± k*std extrema) or delete the branch so the input hits
the explicit `raise KeyError(params)`. Deleting is fine; docstring must
match.

### 18. `normalize` sigmoid default `beta` **[DECISION — behavior change]**
`kwarray/util_robust.py:663`: `beta = old_max - old_min` (the span). The
docstring (line 501) says the default is `(max - min) / 2`, and the evident
intent (center the sigmoid on the data) implies `(old_max + old_min) / 2`.
Verified: input `linspace(100, 200)` produces output `[0.5 ... 0.998]` —
sigmoid centered at the minimum. Fix to `(old_max + old_min) / 2` and
correct the docstring; CHANGELOG must flag the numeric change.

### 19. Investigate RuntimeWarnings on int-dtype normalize with out=
`kwarray/util_robust.py:690,696` emit
`RuntimeWarning: invalid value encountered in cast/multiply` when
normalizing empty or all-NaN arrays into integer `out=` buffers (surfaced by
`tests/test_normalize.py::test_normalize_grid`). Decide the defined behavior
(probably: fill with 0 for empty/all-NaN) and make the cast explicit and
warning-free.

## fast_rand.py

### 20. `standard_normal` crashes with its own default arguments
`kwarray/fast_rand.py:64`: default is `dtype=float`, but the body only
accepts `np.float32` / `np.float64` identity — builtin `float` matches
neither → `ValueError` (verified: `standard_normal((3,3))` crashes).
Fix: normalize `dtype = np.dtype(dtype).type` at the top so builtin
`float` maps to float64, and keep the docstring's claim ("float32 default")
consistent with whatever default you choose — docstring currently says
float32 but the signature says float. Pick float32 (matching the docstring
and the function's raison d'être) and note it.

### 21. `standard_normal32` mis-distributes one element for odd sizes
`kwarray/fast_rand.py:206-212`: for odd `total`, `out1 = flat[0:n]` and
`out2 = flat[-n:]` overlap at index `n-1`; the write sequence multiplies the
overlapping element by R twice, producing `R_0 * R_{n-1} * sin(theta)`
(verified: std of 200k draws of size-1 output is 1.9995 instead of 1.0).
Fix: make the halves non-overlapping — write cos results into `flat[:n]`
and sin results into `flat[n:]` using only the first `total - n` entries of
Theta/R (apply the R multiplication before any overlapping write). Add a
statistical test: mean/std of `standard_normal32(1)` over many seeded draws
within tolerance, and same for size 3 middle element.

### 22. Dead `np.core.getlimits` fallbacks
`kwarray/fast_rand.py:174-177, 317-320`: `np.iinfo` never raises
AttributeError, and `np.core` is deprecated in numpy 2.x. Delete the
try/excepts. Also fix the `uniform` Benchmark docstring (lines 41–54) which
benchmarks `standard_normal` — copy-paste artifact.

## util_random.py

### 23. `random_combinations(num=None)` degenerates to rejection sampling
`kwarray/util_random.py:228`: the "generate all + shuffle" fast path requires
`num is not None`, so the documented `num=None` ("generate them all") case
collects every combination by coupon-collector rejection (verified 1.5s for
C(18,9)). Fix: `if num is None or num_ > n_max // 2:` for the exhaustive
path.

### 24. `random_product` computes max_num in float32
`kwarray/util_random.py:298`:
`np.multiply.reduce(idx_cards.astype(np.float32))` is inexact past 2**24
(verified `999**3` → 997003008.0 vs exact 997002999) and the result is used
for `min(num, max_num)` and loop bounds. Fix:
`max_num = math.prod(int(c) for c in idx_cards)`.

### 25. Stale scipy doctest gates on `random_combinations`
`kwarray/util_random.py:176,194`: both doctests are gated
`+REQUIRES(module:scipy)` but the function uses `math.comb` (scipy is only a
py<3.8 fallback inside the code). Remove the gates so minimal installs test
the function, and replace `+IGNORE_WANT` with real assertions. If phase 07
drops Python 3.8, also delete the scipy fallback inside the function body.

## util_groups.py

### 26. `group_indices` crashes on a list of integer tuples
`kwarray/util_groups.py:199-206`: acknowledged FIXME.
`np.asarray([(1,2),(3,4),(1,2)])` makes a 2D int array, skipping the
object-array fallback, and the flattened-size bookkeeping breaks →
`ValueError` (verified). Fix: after `asarray`, if `idx_to_groupid.ndim > 1`
and it wasn't produced from a 2D ndarray input intentionally, route to the
`ub.group_items`-style fallback treating rows as keys (string tuples already
take that path because they coerce to kind 'U'). Be careful to preserve the
existing documented behavior for genuine 2D ndarray inputs — check the
existing doctests for that case first.

## util_slices.py

### 27. `_apply_padding` crashes when pad_width is a tuple with trailing dims
`kwarray/util_slices.py:171`: `pad_width + ([(0,0)] * trailing_dims)` —
tuple + list TypeError. Triggered by following `apply_embedded_slice`'s own
documented types with channel dims (verified). Fix:
`pad_width = list(pad_width) + [(0, 0)] * trailing_dims`.

## util_misc.py

### 28. `FlatIndexer.unravel` edge cases and O(n) scalar path
`kwarray/util_misc.py:83-94`: negative indices return nonsense
(`unravel(-1)` → `(0, -1)`), OOB raises a confusing IndexError, and the
scalar path uses `np.where(self.cums > index)[0][0]` (O(n)) while the array
path uses searchsorted. Fix: use `np.searchsorted(self.cums, index,
side='right')` for both paths and add explicit bounds validation with a
clear error message.

## Verification
- Add regression tests per item (phase 06 lists where they should live).
- `python3 -m pytest tests/ --xdoctest kwarray/ -q` green.
- For the statistical fixes (items 20, 21): seeded distributional tests with
  generous tolerances, not exact-value assertions.
- CHANGELOG entries for every item; **[DECISION]** items 1, 7, 18 get their
  own "Changed" bullets, not buried under "Fixed".
