The Kitware Array Module
========================

The ``kwarray`` module implements a small set of pure-python extensions to numpy and torch.

Here is the complete API:

.. code:: python

    from .arrayapi import ArrayAPI
    from . import distributions
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


The ArrayAPI
------------

On of the most useful features in ``kwarray`` is the ``kwarray.ArrayAPI`` --- a
class that helps bridge between numpy and torch. This class consists of static
methods that implement part of the numpy API and operate equivalently on either
torch.Tensor or numpy.ndarray objects. 

This works because every function call checks if the input is a torch tensor or
a numpy array and then takes the appropriate action.

As you can imagine, it can be slow to validate your inputs on each function
call. Therefore the recommended way of using the array API is via the
``kwarray.ArrayAPI.impl`` function. This function does the check once and then
returns another object that directly performs the correct operations on
subsequent data items of the same type. 

The following example demonstrates both modes of usage.

.. code:: python

        import torch
        import numpy as np
        data1 = torch.rand(10, 10)
        data2 = data1.numpy()
        # Method 1: grab the appropriate sub-impl
        impl1 = ArrayAPI.impl(data1)
        impl2 = ArrayAPI.impl(data2)
        result1 = impl1.sum(data1, axis=0)
        result2 = impl2.sum(data2, axis=0)
        assert np.all(impl1.numpy(result1) == impl2.numpy(result2))
        # Method 2: choose the impl on the fly
        result1 = ArrayAPI.sum(data1, axis=0)
        result2 = ArrayAPI.sum(data2, axis=0)
        assert np.all(ArrayAPI.numpy(result1) == ArrayAPI.numpy(result2))


Other Notes:
------------

The ``kwarray.ensure_rng`` function helps you properly maintain and control local
seeded random number generation. This means that you wont clobber the random
state of another library / get your random state clobbered.

``DataFrameArray`` and ``DataFrameLight`` implement a subset of the pandas API.
They are less powerful, but orders of magnitude faster. The main drawback is
that you lose ``loc``, but ``iloc`` is available.

``uniform32`` and ``standard_normal32`` are faster 32-bit random number generators
(compared to their 64-bit numpy counterparts).

``mincost_assignment`` is the Munkres / Hungarian algorithm. It solves the
assignment problem.

``one_hot_embedding`` is a fast numpy / torch way to perform the often needed OHE
deep-learning trick.

``group_items`` is a fast way to group a numpy array by another numpy array.  For
fine grained control we also expose ``group_indices``, which groups the indices
of a numpy array, and ``apply_grouping``, which partitions a numpy array by those
indices.

``boolmask`` effectively inverts ``np.where``.
