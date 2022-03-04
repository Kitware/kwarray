# -*- coding: utf-8 -*-
"""
The ``kwarray`` module implements a small set of pure-python extensions to
numpy and torch.
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals
"""
AutogenInit:
    mkinit ~/code/kwarray/kwarray/__init__.py --relative --nomods
    mkinit ~/code/kwarray/kwarray/__init__.py --relative --nomods -w
"""

__private__ = [
    'distributions',
    'arrayapi'
]

__explicit__ = [
    'ArrayAPI', 'dtype_info',
]

__version__ = '0.6.1'

from kwarray.arrayapi import ArrayAPI, dtype_info
# Everything after this point is autogenerated
from .algo_assignment import (maxvalue_assignment, mincost_assignment,
                              mindist_assignment,)
from .algo_setcover import (setcover,)
from .dataframe_light import (DataFrameArray, DataFrameLight, LocLight,)
from .fast_rand import (standard_normal, standard_normal32, standard_normal64,
                        uniform, uniform32,)
from .util_averages import (RunningStats, stats_dict,)
from .util_groups import (apply_grouping, group_consecutive,
                          group_consecutive_indices, group_indices,
                          group_items,)
from .util_misc import (FlatIndexer,)
from .util_numpy import (arglexmax, argmaxima, argminima, atleast_nd, boolmask,
                         isect_flags, iter_reduce_ufunc, normalize,
                         unique_rows,)
from .util_random import (ensure_rng, random_combinations, random_product,
                          seed_global, shuffle,)
from .util_robust import (find_robust_normalizers, robust_normalize,)
from .util_slices import (apply_embedded_slice, embed_slice, padded_slice,)
from .util_slider import (SlidingWindow, Stitcher,)
from .util_torch import (one_hot_embedding, one_hot_lookup,)

__all__ = ['ArrayAPI', 'DataFrameArray', 'DataFrameLight', 'FlatIndexer',
           'LocLight', 'RunningStats', 'SlidingWindow', 'Stitcher',
           'apply_embedded_slice', 'apply_grouping', 'arglexmax', 'argmaxima',
           'argminima', 'atleast_nd', 'boolmask', 'dtype_info', 'embed_slice',
           'ensure_rng', 'find_robust_normalizers', 'group_consecutive',
           'group_consecutive_indices', 'group_indices', 'group_items',
           'isect_flags', 'iter_reduce_ufunc', 'maxvalue_assignment',
           'mincost_assignment', 'mindist_assignment', 'normalize',
           'one_hot_embedding', 'one_hot_lookup', 'padded_slice',
           'random_combinations', 'random_product', 'robust_normalize',
           'seed_global', 'setcover', 'shuffle', 'standard_normal',
           'standard_normal32', 'standard_normal64', 'stats_dict', 'uniform',
           'uniform32', 'unique_rows']
