# Phase 05 — Confirmed bugs in arrayapi.py and util_torch.py

Scope: `kwarray/arrayapi.py`, `kwarray/util_torch.py`.

**Environment requirement:** the numpy-side bugs below were verified without
torch; the torch-side twins were verified by code inspection against the
torch API contract. Before starting this phase, create a venv WITH torch
(CPU wheel is fine: `pip install torch --index-url
https://download.pytorch.org/whl/cpu`) so every fix can be exercised on both
backends. The parity test in phase 06 (`test_numpy_torch_compat`, currently
a dead test) is the acceptance harness for this phase — fix it first (see
phase 06 item 1) and extend its op coverage as you go.

## Confirmed bugs (empirically verified on the numpy side)

### 1. `empty` broken on BOTH backends
`arrayapi.py:1333-1335` (numpy) and `:610-614` (torch): both call
`full(shape, dtype=dtype)` — `full` requires a positional `fill_value`, so
`impl.empty(shape)` always raises TypeError (verified numpy-side; torch has
the identical typo). Fix: `np.empty(shape, dtype=dtype)` /
`torch.empty(shape, dtype=dtype)`. Note `empty/full/ones/zeros` are
registered but commented out of the `ArrayAPI` facade at 1749–1752 — expose
them (they're implemented and registered; the omission looks accidental).

### 2. `ArrayAPI.ensure` does not exist (double assignment)
`arrayapi.py:1738-1739`:
```python
asarray = _apimethod('asarray')
asarray = _apimethod('ensure')   # overwrites the line above
```
Verified: `hasattr(ArrayAPI, 'ensure')` is False and
`ArrayAPI.asarray.__name__ == 'ensure'`. Fix the second line to
`ensure = _apimethod('ensure')`.

### 3. Registry drift guard is never run
`_ensure_datamethods_names_are_registered` (around `arrayapi.py:1796`) would
have caught item 2, but it is never invoked and it MUTATES the registry
(pops `'func'` at line 289). Make it non-destructive and call it from a unit
test (add to `tests/test_arrayapi.py`) so facade/registry drift fails CI.

### 4. `one_hot_embedding` mishandles negative dim (both backends)
`util_torch.py:84-85` (torch) and `:101-102` (numpy):
`dim = in_dims - dim + 1` is the wrong conversion; the correct formula for
an output of ndim `in_dims + 1` is `dim = in_dims + 1 + dim`. The wrong
index is silently clamped by `list.insert` to "append", so the class axis
always lands last (verified numpy-side: `dim=-2` on 2D labels gives shape
`(2,2,3)` instead of `(2,3,2)`; `dim=-1` only works by coincidence). Fix
both occurrences; also `new_axes = old_axes` at `:114` aliases instead of
copying (currently harmless — fix anyway). Add tests over
`dim in {-3,-2,-1,0,1,2}` comparing negative against equivalent positive
dims, both backends.

### 5. torch dtype LUT inversion → wrong result_type promotion
`arrayapi.py:1678-1680` with LUT construction at `:1829, :1861`:
`ub.invert_dict` on the many-to-one name→dtype LUT keeps the LAST key, so
`torch.float32 → 'float' → np.float64` and `torch.int32 → 'int' → int64`
(the `'float'`/`'int'` aliases are inserted after the sized names —
`ub.invert_dict({'a':1,'b':1}) == {1:'b'}` verified). Result: with torch
installed, `ArrayAPI.result_type(torch.float32, torch.float32)` returns
float64. Fix: build an explicit `torch.dtype → np.dtype` mapping instead of
inverting the coercion LUT.

### 6. `nan_to_num` torch/numpy divergence
`arrayapi.py:997-1003`: torch impl only zeroes NaN, leaves ±inf; numpy side
is `np.nan_to_num` which replaces ±inf too; kwarg signatures also differ.
Fix: `torch.nan_to_num` (available torch ≥1.8) and accept/forward
`nan=/posinf=/neginf=` on both backends.

### 7. ArrayAPI dispatch rejects plain Python scalars
`arrayapi.py:200-211`: lists/tuples/np scalars are handled, but
`ArrayAPI.sum(3.0)` raises "unknown type" (verified). Fix: route
`numbers.Number` / `bool` to the numpy impl.

### 8. `ifloor`/`iceil` return int32 while `iround` defaults to int64 **[DECISION]**
`arrayapi.py:1504-1509, 1187-1194` vs `:1516-1517, :1222-1226`. Backends
agree with each other but the family is inconsistent and int32 silently
overflows large coordinates. Recommended fix: give ifloor/iceil the same
`dtype=int` parameter iround has (default int64 on Linux) — this is a
behavior change for consumers relying on int32 output; CHANGELOG "Changed".

## Torch-side bugs (verify in the torch venv, then fix)

### 9. `TorchImpls.repeat` unconditionally raises
`arrayapi.py:527`: first statement is `raise NotImplementedError`, dead code
below, yet `ArrayAPI.repeat` is exposed and works for numpy. Fix:
`torch.repeat_interleave(data, repeats, dim=axis)` matches np.repeat
(including array-valued repeats).

### 10. `TorchImpls.asarray` uses the legacy `torch.Tensor(...)` constructor
`arrayapi.py:1150-1152`: always float32, always copies, and for a scalar int
input `torch.Tensor(n)` is the SIZE constructor returning uninitialized
memory. Line 1150 is also a bare no-op statement. Fix:
`torch.as_tensor(data)`.

### 11. `TorchImpls.take` / `compress` legacy constructors
`arrayapi.py:400-401, 441`: `torch.LongTensor(indices)` with a Python int is
the size constructor → garbage; `LongTensor`/`BoolTensor` don't respect
device. Fix: `torch.as_tensor(indices, dtype=torch.long,
device=data.device)` (same for bool flags).

### 12. `TorchImpls.tile` under-length reps
`arrayapi.py:486-493`: numpy pads reps with LEADING 1s; torch impl pads the
other way → Tensor.repeat raises. Also `torch.empty(...).to(device)`
allocates on CPU first. Fix padding direction; allocate with
`device=` kwarg directly.

### 13. `TorchImpls.dtype_kind` wrong for bool/complex/uint16+
`arrayapi.py:1166-1174`: returns 'i' for torch.bool (numpy 'b'), complex
data falls into the int branch ('i' instead of 'c'), uint16/32/64 (torch
≥2.3) → 'i' instead of 'u'. Rewrite with explicit dtype→kind mapping.

### 14. `TorchImpls.result_type` single-arg / tensor-arg behavior
`arrayapi.py:322-323`: one argument returns the argument itself (numpy
returns a dtype); tensor varargs break `torch.empty(0, dtype=a)`. Fix to
mirror `np.result_type` semantics (accept tensors and dtypes, return dtype).

### 15. `TorchImpls.numpy` uses legacy `.data`
`arrayapi.py:1124`: use `data.detach().cpu().numpy()`.

### 16. `TorchImpls.astype` double copy
`arrayapi.py:1109-1114`: `.to(dtype)` already copies on dtype change; the
unconditional `.clone()` doubles it. Only clone when `data.dtype == dtype`
and `copy=True`.

### 17. `TorchImpls.pad` shorthand support
`arrayapi.py:1135-1138`: numpy accepts scalar / `(before, after)` shorthand
`pad_width`; torch impl requires the full per-axis pairs, and mode names
differ (numpy 'edge' == torch 'replicate'). Normalize the shorthand and
translate mode names.

### 18. `view` semantic divergence
numpy impl uses `reshape` (`:1278`, silently copies when needed); torch
`.view` (`:394`) raises on non-contiguous. Use `reshape` on both sides (and
keep `view` as an alias) or document the divergence loudly.

### 19. Version detection fragility
`arrayapi.py:87-110`: versions come from `importlib.metadata` of the
DISTRIBUTION (breaks for editable/vendored installs); if numpy metadata is
missing, `_NUMPY_GE_2_0_0 = None` is falsy and `_torch_dtype_lut` (`:1837`)
reaches `np.float_` → AttributeError on numpy ≥2. The `distutils` fallback
(`:76`) is dead on py≥3.12 and `pkg_resources` (`:81`) is removed in
setuptools ≥81. Fix: parse `np.__version__` / `torch.__version__` directly
(packaging.version if available, tuple-split fallback otherwise).

### 20. `NumpyImpls.softmax` requires scipy
`arrayapi.py:1523`: a base install fails on `ArrayAPI.softmax`. Implement
natively (subtract max, exp, normalize — 3 lines) and drop the scipy import.

### 21. `argsort(descending=True)` tie-order mismatch
`arrayapi.py:1356-1360` vs torch `descending=True`: tie permutations differ
between backends. Low priority: document, or normalize by flipping consistently
on both sides. The doctest at `:641` only passes because rand rarely ties.

## Cleanups (same files)

- Dead code: commented torch import block (`:121-147`), `numpy_type_map`
  check (`:1870-1877`), duplicate namedtuple import (`:1962-1966`), unused
  ducktype instantiation whose result is discarded (`:1967-1972` — decide if
  it was meant to be a fallback; if so `info` is unbound on that path).
- `NumpyImpls.kron` (`:1526`) is an unregistered orphan with no torch twin
  and no facade entry — register + implement torch side, or remove.
- `NumpyImpls.tensor` (`:1455-1457`) has an unreachable `if torch is None`
  guard after a bare `import torch`; catch ImportError with a friendly
  message instead.
- `ArrayAPI.impl('hello')` returns NumpyImpls for any non-tensor — add
  validation or document.
- Per-call `sys.modules.get('torch')` + per-method `import torch` overhead:
  cache a module-level handle on first success (keep laziness).
- `TorchImpls.max/min` tuple-axis loop (`:694-699`): use
  `torch.amax/amin(data, dim=axis)`.

## Verification
- Fix and extend `tests/test_arrayapi.py::test_numpy_torch_compat` FIRST
  (phase 06 item 1: it currently asserts nothing). Target: every op exposed
  on `ArrayAPI` gets a numpy-vs-torch parity check over a dtype/shape grid
  including float32/float64/int32/uint8/bool, 0-d/1-d/2-d, and (where
  meaningful) NaN/inf values.
- Registry symmetry test from item 3 runs in CI.
- Full suite green in BOTH environments: with torch and without torch.
- `import kwarray` must still never import torch eagerly (verify with
  `python3 -c "import kwarray, sys; assert 'torch' not in sys.modules"`).
