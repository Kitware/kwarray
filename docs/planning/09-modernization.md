# Phase 09 — Modernization (design work; do last)

These items need maintainer buy-in on direction. Each is independent; each
should be its own branch/MR. Prereqs: phases 02, 05, 06 (fixed and tested
baselines to refactor against).

## 1. Numerically stable RunningStats (Welford/Chan)

`RunningStats` uses the naive sum-of-squares formula
(`util_averages.py:690-701`), which suffers catastrophic cancellation — this
is SILENT DATA CORRUPTION for large-mean/small-variance data. Verified:
10k samples with mean 1e8, std 1e-3 → reported std 1.159 (>1000x error, no
warning). The class docstring already carries a TODO (line ~427) citing
Pébay's parallel-moments paper, and `dev/poc/pebaystats.py` contains a
started implementation (its doctests import `kwarray._backend.pebaystats` —
the intended landing spot before the `_backend` stub was abandoned).

Plan:
- Replace total/squares accumulators with per-element (n, mean, M2)
  Welford state; use Chan's parallel combination formula for
  `update_many`, `_update_from_other`, and `+=` merging.
- Keep the public `summarize()` output schema identical; add a regression
  test with the mean-1e8 case asserting relative std error < 1e-6.
- Decide whether `dev/poc/pebaystats.py` (higher moments: skew/kurtosis)
  graduates into the package (e.g. `kwarray/_backend/pebaystats.py` or just
  `kwarray/_pebaystats.py`) or whether plain Welford inline in
  util_averages suffices. If graduating, this is when a real `_backend`
  package gets created (phase 01 deleted the empty stub).

## 2. Interoperate with the Python array-API standard

`kwarray.ArrayAPI` predates the official standard; numpy ≥2.0 implements it
natively and `array-api-compat` wraps torch/cupy/dask/jax.
`ArrayAPI.coerce(data)` is morally `array_api_compat.array_namespace(x)`.

Plan (incremental, non-breaking):
- Add an optional backend that returns the standard namespace when
  `array_api_compat` is importable, keeping kwarray's method names as thin
  aliases: `cat→xp.concat`, `T/transpose→matrix_transpose/permute_dims`,
  `astype→xp.astype`, `dtype_kind→xp.isdtype`, `dtype_info→xp.finfo/iinfo`,
  `numel→math.prod(x.shape)`, `argsort(descending=, stable=)` (exact
  signature match), `take`, `clip`, `maximum`, `minimum`, `tile`, `repeat`.
  This eliminates most torch-side divergence classes fixed in phase 05 by
  construction.
- Keep custom: `pad`, `softmax`, `nan_to_num`, `max_argmax/min_argmin`,
  `ifloor/iceil/iround`, `compress`, `contiguous`, `tensor/numpy`
  conversions.
- Gate everything on the phase-06 parity test grid; add cupy/jax to the
  grid opportunistically (optional CI job).
- Longer term **[DECISION]**: document `ArrayAPI` as a compatibility shim
  and point new users at the standard, or keep it as kwarray's primary
  public interface. Recommend the former.

## 3. ArrayAPI dispatch robustness

Carryover design items from phase 05's audit:
- `_apimethod` binds impls at class-creation (`arrayapi.py:197-199`) —
  subclass/monkeypatch of impl classes never takes effect. Either resolve
  lazily or document.
- The `'data_func'` func_type taxonomy mislabels binary ops (`maximum`,
  `isclose`, `arctan2`) — works only because dispatch keys off the first
  arg. Introduce a `'binary'` func_type or document.
- Run `_ensure_datamethods_names_are_registered` in CI (done in phase 05
  item 3) and extend it to verify facade completeness both directions.

## 4. Deprecation sweep

Formalize with `ub.schedule_deprecation` (ubelt) where available:
- `'propogate'` nan-policy alias (phases 02/04) — schedule removal.
- `DataFrameLight` **[DECISION]**: pandas 3.x is fast and ubiquitous now;
  decide whether DataFrameLight remains a supported public API or gets
  deprecated. If kept, wire the phase-06 pandas-parity canary tests into
  CI against latest pandas.
- `dataframe_light` doctest/typing mentions of old pandas idioms; stale
  numpy aliases in non-executed docs.
- `util_torch` onnx/onnx_tf example (dead upstream — remove or replace
  with torch.onnx.export).
- `distributions._process_docstrings()` import-time mutation → make static
  (deferred from phase 03 item 18).

## 5. Performance opportunities (benchmark before/after; keep dev/ scripts)

- `argmaxima/argminima` ordered axis path: replace the Python row loop
  (`util_numpy.py:319-323, 423-427`) with
  `take_along_axis`-based vectorization; also unify the two functions
  (`argminima` = `argmaxima(-arr)` or a shared `_arg_extrema`).
- `RunningDiscreteHistogram` merge: O(n_bins) merge instead of
  `np.repeat` materialization (phase 02 item 11 note).
- `Mixture.sample`: vectorize the per-component dict_hist loop with
  searchsorted on sorted choice indices.
- `algo_assignment` padded-fill loops (`:497-502`): build via coo arrays.
- `isect_flags` → `np.isin` (done in phase 02 item 4; benchmark to confirm).
- Add a `dev/benchmarks/` harness commit (phase 01 decided whether the
  existing untracked file was kept) so perf claims are reproducible.

## 6. Type-stub completeness (after phase 07's nptyping removal)

- Autogenerate or hand-verify `.pyi` coverage for every public export;
  add a CI step that runs a type checker over the stubs
  (`mypy --strict kwarray/*.pyi`-level is aspirational; start with
  import-clean).
