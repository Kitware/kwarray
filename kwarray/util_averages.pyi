from numpy.typing import ArrayLike
from typing import Dict
import collections
from typing import Union
from typing import List
from ubelt.util_const import NoParamType
import ubelt as ub


def stats_dict(inputs: ArrayLike,
               axis: int = None,
               nan: bool = False,
               sum: bool = False,
               extreme: bool = True,
               n_extreme: bool = False,
               median: bool = False,
               shape: bool = True,
               size: bool = False,
               quantile: str = ...) -> collections.OrderedDict:
    ...


class RunningStats(ub.NiceRepr):

    def __init__(run) -> None:
        ...

    def __nice__(self):
        ...

    @property
    def shape(run):
        ...

    def update(run, data, weights: int = ...) -> None:
        ...

    def summarize(run,
                  axis: Union[int, List[int], None, NoParamType] = None,
                  keepdims: bool = True) -> Dict:
        ...

    def current(run):
        ...
