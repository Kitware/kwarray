"""
Misc tools that should find a better home
"""
import numpy as np
import ubelt as ub
import numbers


class FlatIndexer(ub.NiceRepr):
    """
    Creates a flat "view" of a jagged nested indexable object.
    Only supports one offset level.

    Args:
        lens (List[int]):
            a list of the lengths of the nested objects.

    Doctest:
        >>> self = FlatIndexer([1, 2, 3])
        >>> len(self)
        >>> self.unravel(4)
        >>> self.ravel(2, 1)
    """
    def __init__(self, lens):
        self.lens = np.asarray(lens)
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
        """
        Returns:
            int
        """
        return self.cums[-1] if len(self.cums) else 0

    def unravel(self, index):
        """
        Args:
            index (int | List[int]) : raveled index

        Returns:
            Tuple[int, int]: outer and inner indices

        Example:
            >>> import kwarray
            >>> rng = kwarray.ensure_rng(0)
            >>> items = [rng.rand(rng.randint(0, 10)) for _ in range(10)]
            >>> self = kwarray.FlatIndexer.fromlist(items)
            >>> index = np.arange(0, len(self))
            >>> outer, inner = self.unravel(index)
            >>> recon = self.ravel(outer, inner)
            >>> # This check is only possible because index is an arange
            >>> check1 = np.hstack(list(map(sorted, kwarray.group_indices(outer)[1])))
            >>> check2 = np.hstack(kwarray.group_consecutive_indices(inner))
            >>> assert np.all(check1 == index)
            >>> assert np.all(check2 == index)
            >>> assert np.all(index == recon)
        """
        if isinstance(index, numbers.Integral):
            outer = np.where(self.cums > index)[0][0]
            base = self.cums[outer] - self.lens[outer]
            inner = index - base
            return (outer, inner)
        else:
            # index = np.asarray(index)
            # outers = np.where(self.cums[None, :] > index[:, None])
            outer = np.searchsorted(self.cums, index, side='right')
            base = self.cums[outer] - self.lens[outer]
            inner = index - base
            return (outer, inner)

    def ravel(self, outer, inner):
        """
        Args:
            outer: index into outer list
            inner: index into the list referenced by outer

        Returns:
            List[int]:
                the raveled index
        """
        base = self.cums[outer] - self.lens[outer]
        return base + inner
