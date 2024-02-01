The Kitware Array Module
========================

.. # TODO Get CI services running on gitlab

|GitlabCIPipeline| |GitlabCICoverage| |Appveyor| |Pypi| |Downloads| |ReadTheDocs|


+------------------+-------------------------------------------------------+
| Read the docs    | https://kwarray.readthedocs.io                        |
+------------------+-------------------------------------------------------+
| Gitlab (main)    | https://gitlab.kitware.com/computer-vision/kwarray    |
+------------------+-------------------------------------------------------+
| Github (mirror)  | https://github.com/Kitware/kwarray                    |
+------------------+-------------------------------------------------------+
| Pypi             | https://pypi.org/project/kwarray                      |
+------------------+-------------------------------------------------------+

The main webpage for this project is: https://gitlab.kitware.com/computer-vision/kwarray

The ``kwarray`` module implements a small set of pure-python extensions to numpy and torch.

The ``kwarray`` module started as extensions for numpy + a simplified
pandas-like DataFrame object with much faster item row and column access. But
it also include an ArrayAPI, which is a wrapper that allows 100%
interoperability between torch and numpy. It also contains a few algorithms
like setcover and mincost_assignment.


The top-level API is:

.. code:: python

    from kwarray.arrayapi import ArrayAPI, dtype_info
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
                             isect_flags, iter_reduce_ufunc, normalize,)
    from .util_random import (ensure_rng, random_combinations, random_product,
                              seed_global, shuffle,)
    from .util_slices import (embed_slice, padded_slice,)
    from .util_slider import (SlidingWindow, Stitcher,)
    from .util_torch import (one_hot_embedding, one_hot_lookup,)



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

``setcover`` - solves the minimum weighted set cover problem using either an
approximate or an exact solution.

``one_hot_embedding`` is a fast numpy / torch way to perform the often needed OHE
deep-learning trick.

``group_items`` is a fast way to group a numpy array by another numpy array.  For
fine grained control we also expose ``group_indices``, which groups the indices
of a numpy array, and ``apply_grouping``, which partitions a numpy array by those
indices.

``boolmask`` effectively inverts ``np.where``.

Usefulness:
-----------

This is the frequency that I've used various components of this library with in
my projects:


======================================================================================================================================================== ================
 Function name                                                                                                                                                 Usefulness
======================================================================================================================================================== ================
`kwarray.ensure_rng <https://kwarray.readthedocs.io/en/latest/kwarray.util_random.html#kwarray.util_random.ensure_rng>`__                                             239
`kwarray.ArrayAPI <https://kwarray.readthedocs.io/en/latest/kwarray.arrayapi.html#kwarray.arrayapi.ArrayAPI>`__                                                       148
`kwarray.atleast_nd <https://kwarray.readthedocs.io/en/latest/kwarray.util_numpy.html#kwarray.util_numpy.atleast_nd>`__                                                50
`kwarray.DataFrameArray <https://kwarray.readthedocs.io/en/latest/kwarray.dataframe_light.html#kwarray.dataframe_light.DataFrameArray>`__                              43
`kwarray.group_indices <https://kwarray.readthedocs.io/en/latest/kwarray.util_groups.html#kwarray.util_groups.group_indices>`__                                        40
`kwarray.stats_dict <https://kwarray.readthedocs.io/en/latest/kwarray.util_averages.html#kwarray.util_averages.stats_dict>`__                                          34
`kwarray.normalize <https://kwarray.readthedocs.io/en/latest/kwarray.util_robust.html#kwarray.util_robust.normalize>`__                                                28
`kwarray.embed_slice <https://kwarray.readthedocs.io/en/latest/kwarray.util_slices.html#kwarray.util_slices.embed_slice>`__                                            21
`kwarray.shuffle <https://kwarray.readthedocs.io/en/latest/kwarray.util_random.html#kwarray.util_random.shuffle>`__                                                    17
`kwarray.padded_slice <https://kwarray.readthedocs.io/en/latest/kwarray.util_slices.html#kwarray.util_slices.padded_slice>`__                                          14
`kwarray.SlidingWindow <https://kwarray.readthedocs.io/en/latest/kwarray.util_slider.html#kwarray.util_slider.SlidingWindow>`__                                        14
`kwarray.isect_flags <https://kwarray.readthedocs.io/en/latest/kwarray.util_numpy.html#kwarray.util_numpy.isect_flags>`__                                              12
`kwarray.RunningStats <https://kwarray.readthedocs.io/en/latest/kwarray.util_averages.html#kwarray.util_averages.RunningStats>`__                                      12
`kwarray.standard_normal <https://kwarray.readthedocs.io/en/latest/kwarray.fast_rand.html#kwarray.fast_rand.standard_normal>`__                                        10
`kwarray.setcover <https://kwarray.readthedocs.io/en/latest/kwarray.algo_setcover.html#kwarray.algo_setcover.setcover>`__                                               8
`kwarray.robust_normalize <https://kwarray.readthedocs.io/en/latest/kwarray.util_robust.html#kwarray.util_robust.robust_normalize>`__                                   7
`kwarray.boolmask <https://kwarray.readthedocs.io/en/latest/kwarray.util_numpy.html#kwarray.util_numpy.boolmask>`__                                                     7
`kwarray.one_hot_embedding <https://kwarray.readthedocs.io/en/latest/kwarray.util_torch.html#kwarray.util_torch.one_hot_embedding>`__                                   7
`kwarray.uniform <https://kwarray.readthedocs.io/en/latest/kwarray.fast_rand.html#kwarray.fast_rand.uniform>`__                                                         6
`kwarray.find_robust_normalizers <https://kwarray.readthedocs.io/en/latest/kwarray.util_robust.html#kwarray.util_robust.find_robust_normalizers>`__                     6
`kwarray.Stitcher <https://kwarray.readthedocs.io/en/latest/kwarray.util_slider.html#kwarray.util_slider.Stitcher>`__                                                   6
`kwarray.apply_grouping <https://kwarray.readthedocs.io/en/latest/kwarray.util_groups.html#kwarray.util_groups.apply_grouping>`__                                       6
`kwarray.group_consecutive <https://kwarray.readthedocs.io/en/latest/kwarray.util_groups.html#kwarray.util_groups.group_consecutive>`__                                 5
`kwarray.argmaxima <https://kwarray.readthedocs.io/en/latest/kwarray.util_numpy.html#kwarray.util_numpy.argmaxima>`__                                                   4
`kwarray.seed_global <https://kwarray.readthedocs.io/en/latest/kwarray.util_random.html#kwarray.util_random.seed_global>`__                                             4
`kwarray.FlatIndexer <https://kwarray.readthedocs.io/en/latest/kwarray.util_misc.html#kwarray.util_misc.FlatIndexer>`__                                                 3
`kwarray.group_items <https://kwarray.readthedocs.io/en/latest/kwarray.util_groups.html#kwarray.util_groups.group_items>`__                                             3
`kwarray.arglexmax <https://kwarray.readthedocs.io/en/latest/kwarray.util_numpy.html#kwarray.util_numpy.arglexmax>`__                                                   2
`kwarray.DataFrameLight <https://kwarray.readthedocs.io/en/latest/kwarray.dataframe_light.html#kwarray.dataframe_light.DataFrameLight>`__                               2
`kwarray.group_consecutive_indices <https://kwarray.readthedocs.io/en/latest/kwarray.util_groups.html#kwarray.util_groups.group_consecutive_indices>`__                 1
`kwarray.equal_with_nan <https://kwarray.readthedocs.io/en/latest/kwarray.util_numpy.html#kwarray.util_numpy.equal_with_nan>`__                                         1
`kwarray.dtype_info <https://kwarray.readthedocs.io/en/latest/kwarray.arrayapi.html#kwarray.arrayapi.dtype_info>`__                                                     1
`kwarray.unique_rows <https://kwarray.readthedocs.io/en/latest/kwarray.util_numpy.html#kwarray.util_numpy.unique_rows>`__                                               0
`kwarray.uniform32 <https://kwarray.readthedocs.io/en/latest/kwarray.fast_rand.html#kwarray.fast_rand.uniform32>`__                                                     0
`kwarray.standard_normal64 <https://kwarray.readthedocs.io/en/latest/kwarray.fast_rand.html#kwarray.fast_rand.standard_normal64>`__                                     0
`kwarray.standard_normal32 <https://kwarray.readthedocs.io/en/latest/kwarray.fast_rand.html#kwarray.fast_rand.standard_normal32>`__                                     0
`kwarray.random_product <https://kwarray.readthedocs.io/en/latest/kwarray.util_random.html#kwarray.util_random.random_product>`__                                       0
`kwarray.random_combinations <https://kwarray.readthedocs.io/en/latest/kwarray.util_random.html#kwarray.util_random.random_combinations>`__                             0
`kwarray.one_hot_lookup <https://kwarray.readthedocs.io/en/latest/kwarray.util_torch.html#kwarray.util_torch.one_hot_lookup>`__                                         0
`kwarray.mindist_assignment <https://kwarray.readthedocs.io/en/latest/kwarray.algo_assignment.html#kwarray.algo_assignment.mindist_assignment>`__                       0
`kwarray.mincost_assignment <https://kwarray.readthedocs.io/en/latest/kwarray.algo_assignment.html#kwarray.algo_assignment.mincost_assignment>`__                       0
`kwarray.maxvalue_assignment <https://kwarray.readthedocs.io/en/latest/kwarray.algo_assignment.html#kwarray.algo_assignment.maxvalue_assignment>`__                     0
`kwarray.iter_reduce_ufunc <https://kwarray.readthedocs.io/en/latest/kwarray.util_numpy.html#kwarray.util_numpy.iter_reduce_ufunc>`__                                   0
`kwarray.generalized_logistic <https://kwarray.readthedocs.io/en/latest/kwarray.util_numpy.html#kwarray.util_numpy.generalized_logistic>`__                             0
`kwarray.argminima <https://kwarray.readthedocs.io/en/latest/kwarray.util_numpy.html#kwarray.util_numpy.argminima>`__                                                   0
`kwarray.apply_embedded_slice <https://kwarray.readthedocs.io/en/latest/kwarray.util_slices.html#kwarray.util_slices.apply_embedded_slice>`__                           0
`kwarray.NoSupportError <https://kwarray.readthedocs.io/en/latest/kwarray.util_averages.html#kwarray.util_averages.NoSupportError>`__                                   0
`kwarray.LocLight <https://kwarray.readthedocs.io/en/latest/kwarray.dataframe_light.html#kwarray.dataframe_light.LocLight>`__                                           0
======================================================================================================================================================== ================



.. |Pypi| image:: https://img.shields.io/pypi/v/kwarray.svg
   :target: https://pypi.python.org/pypi/kwarray

.. |Downloads| image:: https://img.shields.io/pypi/dm/kwarray.svg
   :target: https://pypistats.org/packages/kwarray

.. |ReadTheDocs| image:: https://readthedocs.org/projects/kwarray/badge/?version=release
    :target: https://kwarray.readthedocs.io/en/release/

.. # See: https://ci.appveyor.com/project/jon.crall/kwarray/settings/badges
.. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/py3s2d6tyfjc8lm3/branch/main?svg=true
   :target: https://ci.appveyor.com/project/jon.crall/kwarray/branch/main

.. |GitlabCIPipeline| image:: https://gitlab.kitware.com/computer-vision/kwarray/badges/main/pipeline.svg
   :target: https://gitlab.kitware.com/computer-vision/kwarray/-/jobs

.. |GitlabCICoverage| image:: https://gitlab.kitware.com/computer-vision/kwarray/badges/main/coverage.svg
    :target: https://gitlab.kitware.com/computer-vision/kwarray/-/commits/main
