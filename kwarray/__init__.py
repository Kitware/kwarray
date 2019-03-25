"""
mkinit ~/code/kwarray/kwarray/__init__.py --relative --nomods -w
"""

__protected__ = [
    'distributions',
    'arrayapi'
]

__explicit__ = [
    'ArrayAPI'
]

__version__ = '0.3.0'

from kwarray.arrayapi import ArrayAPI


# Everything after this point is autogenerated
from .algo_assignment import (maxvalue_assignment, mincost_assignment,
                              mindist_assignment,)
from .dataframe_light import (DataFrameArray, DataFrameLight, LocLight,)
from .fast_rand import (standard_normal, standard_normal32, standard_normal64,
                        uniform, uniform32,)
from .util_averages import (stats_dict,)
from .util_groups import (apply_grouping, group_consecutive,
                          group_consecutive_indices, group_indices,
                          group_items,)
from .util_numpy import (arglexmax, argmaxima, argminima, atleast_nd, boolmask,
                         isect_flags, iter_reduce_ufunc,)
from .util_random import (ensure_rng, random_combinations, random_product,
                          seed_global, shuffle,)
from .util_torch import (one_hot_embedding,)

__all__ = ['ArrayAPI', 'DataFrameArray', 'DataFrameLight', 'LocLight',
           'apply_grouping', 'arglexmax', 'argmaxima', 'argminima',
           'atleast_nd', 'boolmask', 'ensure_rng', 'group_consecutive',
           'group_consecutive_indices', 'group_indices', 'group_items',
           'isect_flags', 'iter_reduce_ufunc', 'maxvalue_assignment',
           'mincost_assignment', 'mindist_assignment', 'one_hot_embedding',
           'random_combinations', 'random_product', 'seed_global', 'shuffle',
           'standard_normal', 'standard_normal32', 'standard_normal64',
           'stats_dict', 'uniform', 'uniform32']
