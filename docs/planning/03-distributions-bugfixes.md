# Phase 03 — Confirmed bugs in distributions.py

Scope: `kwarray/distributions.py` only. All items empirically verified
(numpy 2.4.4, scipy 1.18.0). Install scipy before working this phase; the
scipy-gated doctests must run.

## Bugs

### 1. `Exponential.__scipy__` passes scale as scipy's loc
`distributions.py:1071`: `expon(self.scale)` means "exponential shifted
right by scale". Verified: `Exponential(scale=2).__scipy__().mean()` is 3.0
(should be 2.0) and `pdf(0) == 0`, which is impossible for an exponential;
meanwhile `sample()` (numpy path) correctly has mean 2.0 — so the scipy view
and the sampler disagree. Fix: `rv = expon(scale=self.scale)`. The doctest
at lines 1063–1068 baked in the wrong values — update to the correct pdf
values (`[1., 0.37, 0.3]`-ish; recompute exactly).

### 2. `Uniform.__scipy__` passes high as scipy's scale
`distributions.py:1015`: `scipy.stats.uniform(loc, scale)` covers
`[loc, loc+scale]`. Verified: `Uniform(low=5, high=10).__scipy__().support()`
→ `(5.0, 15.0)`. Fix: `uniform(self.low, self.high - self.low)`. The
doctest at 1010–1012 only passes because low=0 — strengthen it with a
nonzero-low case.

### 3. `_BinOpMixin.int()` uses removed `np.int`
`distributions.py:441`: `Uniform(rng=0).int().sample(2)` raises
`AttributeError: module 'numpy' has no attribute 'int'` (removed numpy
≥1.24). Fix: use a lambda/partial applying `.astype(int)` or `np.int64`.

### 4. `round()` broken for default ndigits
`distributions.py:420, 447`: `Composed(np.round, [self, ndigits])` with
`ndigits=None` → `np.round(x, None)` → TypeError on sampling. Fix: omit
the operand when `ndigits is None` (or default to 0).

### 5. `_trysample` swallows all exceptions
`distributions.py:959-964`: the try/except around operand sampling catches
the bug-3 AttributeError and returns the *Distribution object itself* as an
operand, so `(Uniform(rng=0).int() + 1).sample(3)` silently returns a
`Composed` object instead of raising (verified). Fix: replace try/except
with an explicit `isinstance(arg, Distribution)` check (or
`hasattr(arg, 'sample')`). This turns silent garbage into loud errors —
do it in the same commit as items 3/4.

### 6. `Distribution.random()` crashes on untyped Values
`distributions.py:626-630`: `kw[k] = sample` executes even when
`v.type is None`, but `sample` is only assigned inside the typed branch →
`UnboundLocalError` (verified via `Categorical.random(rng=0)`). Fix: move
the assignment inside the `if type is not None:` block; for untyped params
fall back to the declared default.

### 7. `Mixture.seed()` is a no-op
`distributions.py:832-856`: `_idx_choice` starts with `_` so
`Parameterized.__setattr__` (lines 182–184) never registers it as a child;
the `pdfs` list is not a `ParameterizedList`; and `Mixture.sample` never
touches `self.rng`. Verified: `m.seed(42); m.sample(5)` is not reproducible
while `Normal` (control) is. Fix: store components in a
`ParameterizedList`, rename `_idx_choice` (no leading underscore) so seeding
reaches it, and route all randomness through registered children.

### 8. `PDF.sample` uses global RNG
`distributions.py:1522`: `np.random.rand` instead of `self.rng.rand` —
samples track the global seed, not the instance rng (verified). Fix:
`r = self.rng.rand(*shape)`. Also delete the stale "BROKEN?"/"FIXME: not
actually the inverse" comments at 1471/1516 — the inverse-CDF math was
empirically verified correct.

### 9. `TruncNormal.random(rng)` drops the rng
`distributions.py:1236-1243`: parameters are reproducible but the returned
instance is constructed without `rng=rng`, so sampling is not (verified).
Fix: pass `rng=rng` to the constructor.

### 10. `Value.__init__` discards `constraints`
`distributions.py:63`: `self.constraints = None` regardless of the argument.
Related: `Distribution.__init__` builds a constraints list (548–567) that is
never stored or evaluated, and its lambdas close over loop variables
(late-binding bug if ever activated). Fix: assign the argument; then either
implement constraint checking (validate on construction) or delete the whole
constraints mechanism. Deleting is acceptable — it has never worked; pick
one and be consistent.

### 11. `Binomial.sample` needlessly requires scipy
`distributions.py:1335-1337`: sampling round-trips through
`self.__scipy__().rvs(...)`. numpy can do it:
`self.rng.binomial(self.n, self.p, size=shape)`. Verified ImportError with
scipy blocked. Fix accordingly; keep `__scipy__` for pdf/cdf.

### 12. `Uniform.__params__` order is (high, low) **[DECISION — breaking change]**
`distributions.py:997-1000`: positional `Uniform(0, 10)` sets
`high=0, low=10` (verified). Sampling still covers [0,10] by accident of
the formula, but `self.low`/`self.high` are swapped, which breaks anything
consuming them — including the fixed `__scipy__` from item 2 (test that
combination explicitly!). Fix: declare `__params__` in `(low, high)` order.
Search downstream kwarray-ecosystem usage (kwimage, kwcoco, netharn) for
positional `Uniform(` calls before landing; CHANGELOG must flag it.

### 13. `Distribution.random()` requires scipy transitively
`distributions.py:606-618`: `TruncNormal` is in `_PRIMATIVES` and its
constructor imports scipy → `Distribution.random()` ImportErrors on
scipy-less installs (verified). Fix: filter scipy-dependent classes out of
the candidate list when scipy is unavailable (try `ub.modname_to_modpath` or
a cheap `importlib.util.find_spec('scipy')` check).

## Smaller cleanups (same phase, low risk)

14. `distributions.py:700`: `_show` deprecation message says
    "plot is deprecated, use plot instead" — should name `_show`.
15. Inconsistent `coerce` signatures: `Distribution.coerce(arg, rng=None)`
    and `DiscreteUniform.coerce(arg, rng=None)` accept rng;
    `Uniform.coerce(arg)` and `Bernoulli.coerce(arg)` don't (verified
    TypeError). Unify on `(arg, rng=None)`; also make `Distribution.coerce`
    warn or reseed when given an existing Distribution plus an rng it
    currently ignores.
16. `Mixture.sample` always returns float64 (`np.zeros(shape)` at line 850),
    silently casting integer component samples. Allocate with the promoted
    dtype of component samples, or cast at the end from collected parts.
17. `_test_distributions` helper (1541–1560) instantiates abstract
    intermediates and raises — fix its class filter or delete it.
18. `_process_docstrings()` (1563–1621) mutates docstrings at import time;
    module comment says make it static. Low priority; acceptable to defer to
    phase 09 — if deferred, leave a note there.
19. Give `Mixture`, `PDF`, `NonlinearUniform` proper `__params__`
    declarations (kills the "Warning, no params" print handled cosmetically
    in phase 01, and makes `random()`/introspection work for them).
20. `TruncNormal` goes stale if mean/std/low/high are mutated after
    construction (verified). Either make the params properties that call
    `_update_internals`, or document immutability in the class docstring.

## Verification
- With scipy installed: every distribution's `__scipy__()` must agree with
  its own `sample()` — add a parametrized test comparing empirical
  mean/variance of `sample(10000, rng=...)` against `rv.mean()`/`rv.var()`
  within tolerance for Normal, Uniform, Exponential, Binomial, TruncNormal.
  This single test would have caught items 1, 2, and 12.
- Without scipy (uninstall or block via sys.modules): `import kwarray`,
  `Distribution.random()`, `Binomial(...).sample()` all work.
- Seeding test: for every concrete distribution, two instances constructed
  with `rng=0` produce identical `sample(5)` — catches items 7, 8, 9.
- `python3 -m pytest tests/ --xdoctest kwarray/distributions.py -q` green.
