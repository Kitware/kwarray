from typing import List
from typing import Tuple
import ubelt as ub
from _typeshed import Incomplete


class FlatIndexer(ub.NiceRepr):
    lens: Incomplete
    cums: Incomplete

    def __init__(self, lens) -> None:
        ...

    @classmethod
    def fromlist(cls, items: List[list]) -> FlatIndexer:
        ...

    def __len__(self) -> int:
        ...

    def unravel(self, index: int | List[int]) -> Tuple[int, int]:
        ...

    def ravel(self, outer, inner) -> List[int]:
        ...
