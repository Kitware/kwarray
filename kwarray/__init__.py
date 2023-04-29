"""
The ``kwarray`` module implements a small set of pure-python extensions to
numpy and torch along with a few select algorithms. Each module contains
module level docstring that gives a rough idea of the utilities in each module,
and each function or class itself contains a docstring with more details and
examples.

KWarray is part of Kitware's computer vision Python suite:

    https://gitlab.kitware.com/computer-vision

"""
__mkinit__ = """
AutogenInit:
    mkinit kwarray --relative --nomods --diff
    mkinit kwarray --relative --nomods -w
"""

__private__ = [
    'distributions',
    'arrayapi'
]

__explicit__ = [
    'ArrayAPI', 'dtype_info',
]

__version__ = '0.6.13'

from kwarray.arrayapi import ArrayAPI, dtype_info
# Everything after this point is autogenerated
from .algo_assignment import (maxvalue_assignment, mincost_assignment,
                              mindist_assignment,)
from .algo_setcover import (setcover,)
from .dataframe_light import (DataFrameArray, DataFrameLight, LocLight,)
from .fast_rand import (standard_normal, standard_normal32, standard_normal64,
                        uniform, uniform32,)
from .util_averages import (NoSupportError, RunningStats, stats_dict,)
from .util_groups import (apply_grouping, group_consecutive,
                          group_consecutive_indices, group_indices,
                          group_items,)
from .util_misc import (FlatIndexer,)
from .util_numpy import (arglexmax, argmaxima, argminima, atleast_nd, boolmask,
                         equal_with_nan, generalized_logistic, isect_flags,
                         iter_reduce_ufunc, unique_rows,)
from .util_random import (ensure_rng, random_combinations, random_product,
                          seed_global, shuffle,)
from .util_robust import (find_robust_normalizers, normalize,
                          robust_normalize,)
from .util_slices import (apply_embedded_slice, embed_slice, padded_slice,)
from .util_slider import (SlidingWindow, Stitcher,)
from .util_torch import (one_hot_embedding, one_hot_lookup,)

__all__ = ['ArrayAPI', 'DataFrameArray', 'DataFrameLight', 'FlatIndexer',
           'LocLight', 'NoSupportError', 'RunningStats', 'SlidingWindow',
           'Stitcher', 'apply_embedded_slice', 'apply_grouping', 'arglexmax',
           'argmaxima', 'argminima', 'atleast_nd', 'boolmask', 'dtype_info',
           'embed_slice', 'ensure_rng', 'equal_with_nan',
           'find_robust_normalizers', 'generalized_logistic',
           'group_consecutive', 'group_consecutive_indices', 'group_indices',
           'group_items', 'isect_flags', 'iter_reduce_ufunc',
           'maxvalue_assignment', 'mincost_assignment', 'mindist_assignment',
           'normalize', 'one_hot_embedding', 'one_hot_lookup', 'padded_slice',
           'random_combinations', 'random_product', 'robust_normalize',
           'seed_global', 'setcover', 'shuffle', 'standard_normal',
           'standard_normal32', 'standard_normal64', 'stats_dict', 'uniform',
           'uniform32', 'unique_rows']
