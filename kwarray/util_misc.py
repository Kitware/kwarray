import numpy as np
import ubelt as ub


class FlatIndexer(ub.NiceRepr):
    """
    Creates a flat "view" of a jagged nested indexable object.
    Only supports one offset level.

    Args:
        lens (list): a list of the lengths of the nested objects.

    Doctest:
        >>> self = FlatIndexer([1, 2, 3])
        >>> len(self)
        >>> self.unravel(4)
        >>> self.ravel(2, 1)
    """
    def __init__(self, lens):
        self.lens = lens
        self.cums = np.cumsum(lens)

    @classmethod
    def fromlist(cls, items):
        """
        Convenience method to create a :class:`FlatIndexer` from the list of
        items itself instead of the array of lengths.

        Args:
            items (List[list]): a list of the lists you want to flat index over

        Returns:
            FlatIndexer
        """
        lens = list(map(len, items))
        return cls(lens)

    def __len__(self):
        return self.cums[-1] if len(self.cums) else 0

    def unravel(self, index):
        """
        Args:
            index : raveled index

        Returns:
            Tuple[int, int]: outer and inner indices
        """
        outer = np.where(self.cums > index)[0][0]
        base = self.cums[outer] - self.lens[outer]
        inner = index - base
        return (outer, inner)

    def ravel(self, outer, inner):
        """
        Args:
            outer: index into outer list
            inner: index into the list referenced by outer

        Returns:
            index: the raveled index
        """
        base = self.cums[outer] - self.lens[outer]
        return base + inner
