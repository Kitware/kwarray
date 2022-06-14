from typing import Union
from typing import Tuple
from typing import List
from typing import Dict
from numpy import ndarray
from typing import TypeVar

Sliceable = TypeVar("Sliceable")


def padded_slice(data: Sliceable,
                 slices: Union[slice, Tuple[slice, ...]],
                 pad: List[Union[int, Tuple]] = None,
                 padkw: Dict = None,
                 return_info: bool = False) -> Tuple[Sliceable, Dict]:
    ...


__TODO__: str


def apply_embedded_slice(data: ndarray, data_slice, extra_padding, **padkw):
    ...


def embed_slice(slices: Tuple[slice, ...],
                data_dims: Tuple[int, ...],
                pad: List[Union[int, Tuple]] = None) -> Tuple:
    ...
