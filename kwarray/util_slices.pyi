from typing import Any
from typing import Tuple
from typing import List
from typing import Dict
from numpy import ndarray


def padded_slice(data: Any,
                 slices: slice | Tuple[slice, ...],
                 pad: List[int | Tuple] | None = None,
                 padkw: Dict | None = None,
                 return_info: bool = False) -> Tuple[Any, Dict]:
    ...


__TODO__: str


def apply_embedded_slice(data: ndarray, data_slice, extra_padding,
                         **padkw) -> ndarray:
    ...


def embed_slice(slices: Tuple[slice, ...],
                data_dims: Tuple[int, ...],
                pad: int | List[int | Tuple[int, int]] | None = None) -> Tuple:
    ...
