from typing import Union
import random
import numpy
from typing import List
from numbers import Number
import ubelt as ub
from _typeshed import Incomplete
from collections.abc import Generator
from typing import Any

inf: Incomplete


class Value(ub.NiceRepr):
    default: Incomplete
    name: Incomplete
    min: Incomplete
    max: Incomplete
    help: Incomplete
    type: Incomplete
    constraints: Incomplete
    category: Incomplete
    numeric: Incomplete

    def __init__(self,
                 default: Incomplete | None = ...,
                 min: Incomplete | None = ...,
                 max: Incomplete | None = ...,
                 help: Incomplete | None = ...,
                 constraints: Incomplete | None = ...,
                 type: Incomplete | None = ...,
                 name: Incomplete | None = ...) -> None:
        ...

    def __nice__(self):
        ...

    def sample(self, rng):
        ...


class Parameterized(ub.NiceRepr):

    def __init__(self) -> None:
        ...

    def __setattr__(self, key, value) -> None:
        ...

    def children(self) -> Generator[Any, None, None]:
        ...

    def seed(self, rng: Incomplete | None = ...) -> None:
        ...

    def parameters(self) -> Generator[Any, None, None]:
        ...

    def idstr(self, nl: Incomplete | None = ..., thresh: int = ...):
        ...

    def json_id(self):
        ...

    def __nice__(self):
        ...


class ParameterizedList(Parameterized):

    def __init__(self, items) -> None:
        ...

    def append(self, item) -> None:
        ...

    def __iter__(self):
        ...

    def __getitem__(self, key):
        ...

    def idstr(self, nl: Incomplete | None = ..., thresh: int = ...):
        ...


class _BinOpMixin:

    def __add__(self, other):
        ...

    def __pow__(self, other):
        ...

    def __mul__(self, other):
        ...

    def __sub__(self, other):
        ...

    def __truediv__(self, other):
        ...

    def __matmul__(self, other) -> None:
        ...

    def __floordiv__(self, other) -> None:
        ...

    def __mod__(self, other) -> None:
        ...

    def __divmod__(self, other) -> None:
        ...

    def __lshift__(self, other) -> None:
        ...

    def __rshift__(self, other) -> None:
        ...

    def __and__(self, other) -> None:
        ...

    def __xor__(self, other) -> None:
        ...

    def __or__(self, other) -> None:
        ...

    def __neg__(self):
        ...

    def __pos__(self):
        ...

    def __invert__(self) -> None:
        ...

    def __abs__(self):
        ...

    def __round__(self, ndigits: Incomplete | None = ...):
        ...

    def __trunc__(self):
        ...

    def __floor__(self):
        ...

    def __ceil__(self):
        ...

    def int(self):
        ...

    def round(self, ndigits: Incomplete | None = ...):
        ...

    def clip(self,
             a_min: Incomplete | None = ...,
             a_max: Incomplete | None = ...):
        ...

    def log(self):
        ...

    def log10(self):
        ...

    def exp(self):
        ...

    def sqrt(self):
        ...

    def abs(self):
        ...


class _RBinOpMixin(_BinOpMixin):

    def __radd__(self, other):
        ...

    def __rpow__(self, other):
        ...

    def __rmul__(self, other):
        ...

    def __rsub__(self, other):
        ...

    def __rtruediv__(self, other):
        ...

    def __rmatmul__(self, other) -> None:
        ...

    def __rfloordiv__(self, other) -> None:
        ...

    def __rmod__(self, other) -> None:
        ...

    def __rdivmod__(self, other) -> None:
        ...

    def __rlshift__(self, other) -> None:
        ...

    def __rrshift__(self, other) -> None:
        ...

    def __rand__(self, other) -> None:
        ...

    def __rxor__(self, other) -> None:
        ...

    def __ror__(self, other) -> None:
        ...


class Distribution(Parameterized, _RBinOpMixin):
    __params__: Incomplete
    rng: Incomplete

    def __init__(self, *args, **kwargs):
        ...

    def __call__(self, *shape):
        ...

    def seed(self, rng: Incomplete | None = ...) -> None:
        ...

    def sample(self, *shape) -> None:
        ...

    @classmethod
    def random(cls,
               rng: Union[int, float, None, numpy.random.RandomState,
                          random.Random] = None):
        ...

    @classmethod
    def coerce(cls, arg, rng: Incomplete | None = ...):
        ...

    @classmethod
    def cast(cls, arg):
        ...

    @classmethod
    def seeded(cls, rng: int = ...):
        ...

    def plot(self,
             n: str = ...,
             bins: Union[int, List[Number], str] = 'auto',
             stat: str = 'count',
             color: Incomplete | None = ...,
             kde: bool = ...,
             ax: Incomplete | None = ...,
             **kwargs):
        ...


class DiscreteDistribution(Distribution):
    ...


class ContinuousDistribution(Distribution):
    ...


class MixedDistribution(Distribution):
    ...


class Mixture(MixedDistribution):
    pdfs: Incomplete

    def __init__(self,
                 pdfs,
                 weights: Incomplete | None = ...,
                 rng: Incomplete | None = ...) -> None:
        ...

    def sample(self, *shape):
        ...

    @classmethod
    def random(cls,
               rng: Union[int, float, None, numpy.random.RandomState,
                          random.Random] = None,
               n: int = 3):
        ...


class Composed(MixedDistribution):
    __params__: Incomplete

    def sample(self, *shape):
        ...


class CoerceError(ValueError):
    ...


CastError = CoerceError


class Uniform(ContinuousDistribution):
    __params__: Incomplete

    def sample(self, *shape):
        ...

    @classmethod
    def coerce(cls, arg):
        ...


class Exponential(ContinuousDistribution):
    __params__: Incomplete

    def sample(self, *shape):
        ...


class Constant(DiscreteDistribution):
    __params__: Incomplete

    def sample(self, *shape):
        ...


class DiscreteUniform(DiscreteDistribution):
    __params__: Incomplete

    def sample(self, *shape):
        ...

    @classmethod
    def coerce(cls, arg, rng: Incomplete | None = ...):
        ...


class Normal(ContinuousDistribution):
    __params__: Incomplete

    def sample(self, *shape):
        ...

    @classmethod
    def random(cls, rng: Incomplete | None = ...):
        ...


class TruncNormal(ContinuousDistribution):
    __params__: Incomplete

    def __init__(self, *args, **kwargs) -> None:
        ...

    @classmethod
    def random(cls, rng: Incomplete | None = ...):
        ...

    def sample(self, *shape):
        ...


class Bernoulli(DiscreteDistribution):
    __params__: Incomplete

    def sample(self, *shape):
        ...

    @classmethod
    def coerce(cls, arg):
        ...


class Binomial(DiscreteDistribution):
    __params__: Incomplete

    def sample(self, *shape):
        ...


class Categorical(DiscreteDistribution):

    def __init__(self,
                 categories,
                 weights: Incomplete | None = ...,
                 rng: Incomplete | None = ...) -> None:
        ...

    def sample(self, *shape):
        ...


class NonlinearUniform(ContinuousDistribution):

    def __init__(self,
                 min,
                 max,
                 nonlinearity: Incomplete | None = ...,
                 reverse: bool = ...,
                 rng: Incomplete | None = ...):
        ...

    def sample(self, *shape):
        ...


class CategoryUniform(DiscreteUniform):

    def __init__(self, categories=..., rng: Incomplete | None = ...) -> None:
        ...

    def sample(self, *shape):
        ...


class PDF(Distribution):
    edge_x: Incomplete
    edge_cdf: Incomplete
    inv_cdf: Incomplete

    def __init__(self, x, p, rng: Incomplete | None = ...) -> None:
        ...

    def sample(self, *shape):
        ...


class Seeded:
    rng: Incomplete
    cls: Incomplete

    def __init__(self,
                 rng: Incomplete | None = ...,
                 cls: Incomplete | None = ...) -> None:
        ...

    def __call__(self, *args, **kwargs):
        ...

    def __getattr__(self, key):
        ...
