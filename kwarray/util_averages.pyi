from numpy.typing import ArrayLike
from typing import List
from typing import Dict
import collections
from ubelt.util_const import NoParamType
import ubelt as ub


def stats_dict(
        inputs: ArrayLike,
        axis: int | None = None,
        nan: bool = False,
        sum: bool = False,
        extreme: bool = True,
        n_extreme: bool = False,
        median: bool = False,
        shape: bool = True,
        size: bool = False,
        quantile: str | bool | List[float] = 'auto'
) -> collections.OrderedDict:
    ...


class NoSupportError(RuntimeError):
    ...


class RunningStats(ub.NiceRepr):

    def __init__(run,
                 nan_policy: str = ...,
                 check_weights: bool = True,
                 **kwargs) -> None:
        ...

    def __nice__(self):
        ...

    @property
    def shape(run):
        ...

    def update_many(run, data, weights: int = ...) -> None:
        ...

    def update(run, data, weights: int = ...) -> None:
        ...

    def summarize(run,
                  axis: int | List[int] | None | NoParamType = None,
                  keepdims: bool = True) -> Dict:
        ...

    def current(run):
        ...
