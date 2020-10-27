# -*- coding: utf-8 -*-
"""
A convinient interface to solving assignment problems with the Hungarian
algorithm (also known as Munkres or maximum linear-sum-assignment).

The core implementation of munkres in in scipy. Recent versions are written in
C, so their speed should be reflected here.

TODO:
   - [ ] Implement linear-time maximum weight matching approximation algorithm
     from this paper: https://web.eecs.umich.edu/~pettie/papers/ApproxMWM-JACM.pdf
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
# from distutils.version import LooseVersion
# import scipy
# SCIPY_GE_1_4 = LooseVersion(scipy.__version__) >= LooseVersion('1.4.0')


def mindist_assignment(vecs1, vecs2, p=2):
    """
    Finds minimum cost assignment between two sets of D dimensional vectors.

    Args:
        vecs1 (np.ndarray): NxD array of vectors representing items in vecs1
        vecs2 (np.ndarray): MxD array of vectors representing items in vecs2
        p (float): L-p norm to use. Default is 2 (aka Eucliedean)

    Returns:
        Tuple[list, float]: tuple containing assignments of rows in vecs1 to
            rows in vecs2, and the total distance between assigned pairs.

    Notes:
        Thin wrapper around mincost_assignment

    CommandLine:
        xdoctest -m ~/code/kwarray/kwarray/algo_assignment.py mindist_assignment

    CommandLine:
        xdoctest -m ~/code/kwarray/kwarray/algo_assignment.py mindist_assignment

    Example:
        >>> # xdoctest: +REQUIRES(module:scipy)
        >>> # Rows are detections in img1, cols are detections in img2
        >>> rng = np.random.RandomState(43)
        >>> vecs1 = rng.randint(0, 10, (5, 2))
        >>> vecs2 = rng.randint(0, 10, (7, 2))
        >>> ret = mindist_assignment(vecs1, vecs2)
        >>> print('Total error: {:.4f}'.format(ret[1]))
        Total error: 8.2361
        >>> print('Assignment: {}'.format(ret[0]))  # xdoc: +IGNORE_WANT
        Assignment: [(0, 0), (1, 3), (2, 5), (3, 2), (4, 6)]
    """
    from scipy.spatial import distance_matrix
    cost = distance_matrix(vecs1, vecs2, p=p)
    assignment, dist_tot = mincost_assignment(cost)
    return assignment, dist_tot


def mincost_assignment(cost):
    """
    Finds the minimum cost assignment based on a NxM cost matrix, subject to
    the constraint that each row can match at most one column and each column
    can match at most one row. Any pair with a cost of infinity will not be
    assigned.

    Args:
        cost (ndarray): NxM matrix, cost[i, j] is the cost to match i and j

    Returns:
        Tuple[list, float]: tuple containing a list of assignment of rows
            and columns, and the total cost of the assignment.

    CommandLine:
        xdoctest -m ~/code/kwarray/kwarray/algo_assignment.py mincost_assignment


    Example:
        >>> # xdoctest: +REQUIRES(module:scipy)
        >>> # Costs to match item i in set1 with item j in set2.
        >>> cost = np.array([
        >>>     [9, 2, 1, 9],
        >>>     [4, 1, 5, 5],
        >>>     [9, 9, 2, 4],
        >>> ])
        >>> ret = mincost_assignment(cost)
        >>> print('Assignment: {}'.format(ret[0]))
        >>> print('Total cost: {}'.format(ret[1]))
        Assignment: [(0, 2), (1, 1), (2, 3)]
        Total cost: 6

    Example:
        >>> # xdoctest: +REQUIRES(module:scipy)
        >>> cost = np.array([
        >>>     [0, 0, 0, 0],
        >>>     [4, 1, 5, -np.inf],
        >>>     [9, 9, np.inf, 4],
        >>>     [9, -2, np.inf, 4],
        >>> ])
        >>> ret = mincost_assignment(cost)
        >>> print('Assignment: {}'.format(ret[0]))
        >>> print('Total cost: {}'.format(ret[1]))
        Assignment: [(0, 2), (1, 3), (2, 0), (3, 1)]
        Total cost: -inf

    Example:
        >>> # xdoctest: +REQUIRES(module:scipy)
        >>> cost = np.array([
        >>>     [0, 0, 0, 0],
        >>>     [4, 1, 5, -3],
        >>>     [1, 9, np.inf, 0.1],
        >>>     [np.inf, np.inf, np.inf, 100],
        >>> ])
        >>> ret = mincost_assignment(cost)
        >>> print('Assignment: {}'.format(ret[0]))
        >>> print('Total cost: {}'.format(ret[1]))
        Assignment: [(0, 2), (1, 1), (2, 0), (3, 3)]
        Total cost: 102.0
    """
    from scipy.optimize import linear_sum_assignment
    n1, n2 = cost.shape
    n = max(n1, n2)
    # Embed the [n1 x n2] matrix in a padded (with inf) [n x n] matrix
    cost_matrix = np.full((n, n), fill_value=np.inf)
    cost_matrix[0:n1, 0:n2] = cost

    # Find an effective infinite value for infeasible assignments
    is_infinte = np.isinf(cost_matrix)
    is_finite = ~is_infinte
    is_positive = cost_matrix > 0
    is_negative = cost_matrix < 0
    # Note: in scipy 1.4 input costs may be infinte, should fix for this case
    # (also note, they don't allow a budgeted solution, so maybe we have to use
    # effective values)
    feasible_pos_vals = cost_matrix[(is_finite & is_positive)]
    feasible_neg_vals = cost_matrix[(is_finite & is_negative)]
    feasible_extent = feasible_pos_vals.sum() - feasible_neg_vals.sum()
    # Find a value that is approximately pos/neg infinity wrt this matrix
    approx_pos_inf = (n + feasible_extent) * 2
    approx_neg_inf = -approx_pos_inf
    # replace infinite values with effective infinite values
    cost_matrix[(is_infinte & is_positive)] = approx_pos_inf
    cost_matrix[(is_infinte & is_negative)] = approx_neg_inf

    # Solve munkres problem for minimum weight assignment
    indexes = list(zip(*linear_sum_assignment(cost_matrix)))
    # Return only the feasible assignments
    assignment = [(i, j) for (i, j) in indexes
                  if cost_matrix[i, j] < approx_pos_inf]
    # assert len(assignment) == min(cost.shape)
    cost_tot = sum([cost[i, j] for i, j in assignment])
    return assignment, cost_tot


def maxvalue_assignment(value):
    """
    Finds the maximum value assignment based on a NxM value matrix. Any pair
    with a non-positive value will not be assigned.

    Args:
        value (ndarray): NxM matrix, value[i, j] is the value of matching i and j

    Returns:
        Tuple[list, float]: tuple containing a list of assignment of rows
            and columns, and the total value of the assignment.

    CommandLine:
        xdoctest -m ~/code/kwarray/kwarray/algo_assignment.py maxvalue_assignment

    Example:
        >>> # xdoctest: +REQUIRES(module:scipy)
        >>> # Costs to match item i in set1 with item j in set2.
        >>> value = np.array([
        >>>     [9, 2, 1, 3],
        >>>     [4, 1, 5, 5],
        >>>     [9, 9, 2, 4],
        >>>     [-1, -1, -1, -1],
        >>> ])
        >>> ret = maxvalue_assignment(value)
        >>> # Note, depending on the scipy version the assignment might change
        >>> # but the value should always be the same.
        >>> print('Total value: {}'.format(ret[1]))
        Total value: 23.0
        >>> print('Assignment: {}'.format(ret[0]))  # xdoc: +IGNORE_WANT
        Assignment: [(0, 0), (1, 3), (2, 1)]

        >>> ret = maxvalue_assignment(np.array([[np.inf]]))
        >>> print('Assignment: {}'.format(ret[0]))
        >>> print('Total value: {}'.format(ret[1]))
        Assignment: [(0, 0)]
        Total value: inf

        >>> ret = maxvalue_assignment(np.array([[0]]))
        >>> print('Assignment: {}'.format(ret[0]))
        >>> print('Total value: {}'.format(ret[1]))
        Assignment: []
        Total value: 0
    """
    cost = (-value).astype(np.float)
    cost[value <= 0] = np.inf  # dont take anything with non-positive value
    assignment, cost_tot = mincost_assignment(cost)
    value_tot = -cost_tot
    return assignment, value_tot


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m kwarray.algo_assignment
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
