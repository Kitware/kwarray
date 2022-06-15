from typing import Union
from typing import Sequence
from typing import Dict
from typing import Tuple
import ubelt as ub
from _typeshed import Incomplete
from collections.abc import Generator
from typing import Any


class LocLight:
    parent: Incomplete

    def __init__(self, parent) -> None:
        ...

    def __getitem__(self, index):
        ...


class DataFrameLight(ub.NiceRepr):

    def __init__(self,
                 data: Incomplete | None = ...,
                 columns: Incomplete | None = ...) -> None:
        ...

    @property
    def iloc(self):
        ...

    @property
    def values(self):
        ...

    @property
    def loc(self):
        ...

    def __eq__(self, other):
        ...

    def to_string(self, *args, **kwargs):
        ...

    def to_dict(self, orient: str = 'dict', into: type = dict):
        ...

    def pandas(self):
        ...

    def __nice__(self):
        ...

    def __len__(self):
        ...

    def __contains__(self, item):
        ...

    def __normalize__(self) -> None:
        ...

    @property
    def columns(self):
        ...

    def sort_values(self, key, inplace: bool = ..., ascending: bool = ...):
        ...

    def keys(self) -> Generator[Any, None, None]:
        ...

    def get(self, key, default: Incomplete | None = ...):
        ...

    def clear(self) -> None:
        ...

    def __getitem__(self, key):
        ...

    def __setitem__(self, key, value) -> None:
        ...

    def compress(self, flags, inplace: bool = ...):
        ...

    def take(self, indices, inplace: bool = False):
        ...

    def copy(self):
        ...

    def extend(self, other: Union[DataFrameLight, dict[str,
                                                       Sequence]]) -> None:
        ...

    def union(self, *others):
        ...

    @classmethod
    def concat(cls, others):
        ...

    @classmethod
    def from_pandas(cls, df):
        ...

    @classmethod
    def from_dict(cls, records):
        ...

    def reset_index(self, drop: bool = ...):
        ...

    def groupby(self, by: str = None, *args, **kwargs):
        ...

    def rename(self,
               mapper: Incomplete | None = ...,
               columns: Incomplete | None = ...,
               axis: Incomplete | None = ...,
               inplace: bool = ...):
        ...

    def iterrows(self) -> Generator[Tuple[int, Dict], None, None]:
        ...


class DataFrameArray(DataFrameLight):

    def __normalize__(self) -> None:
        ...

    def extend(self, other) -> None:
        ...

    def compress(self, flags, inplace: bool = ...):
        ...

    def take(self, indices, inplace: bool = ...):
        ...
