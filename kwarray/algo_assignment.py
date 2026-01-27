"""
A convenient interface to solving assignment problems with the Hungarian
algorithm (also known as Munkres or maximum linear-sum-assignment).

The core implementation of munkres in in scipy. Recent versions are written in
C, so their speed should be reflected here.

TODO:
   - [ ] Implement linear-time maximum weight matching approximation algorithm
     from this paper: https://web.eecs.umich.edu/~pettie/papers/ApproxMWM-JACM.pdf
"""
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union  # NOQA


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

    Note:
        Thin wrapper around mincost_assignment

    CommandLine:
        xdoctest -m kwarray.algo_assignment mindist_assignment

    Example:
        >>> # xdoctest: +REQUIRES(module:scipy)
        >>> # Rows are detections in img1, cols are detections in img2
        >>> import kwarray
        >>> import numpy as np
        >>> rng = np.random.RandomState(43)
        >>> vecs1 = rng.randint(0, 10, (5, 2))
        >>> vecs2 = rng.randint(0, 10, (7, 2))
        >>> ret = kwarray.mindist_assignment(vecs1, vecs2)
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
        xdoctest -m kwarray.algo_assignment mincost_assignment


    Example:
        >>> # xdoctest: +REQUIRES(module:scipy)
        >>> # Costs to match item i in set1 with item j in set2.
        >>> import kwarray
        >>> cost = np.array([
        >>>     [9, 2, 1, 9],
        >>>     [4, 1, 5, 5],
        >>>     [9, 9, 2, 4],
        >>> ])
        >>> ret = kwarray.mincost_assignment(cost)
        >>> assignment = [(int(a), int(b)) for a, b in ret[0]]
        >>> print('Assignment: {}'.format(assignment))
        >>> print('Total cost: {}'.format(ret[1]))
        Assignment: [(0, 2), (1, 1), (2, 3)]
        Total cost: 6

    Example:
        >>> # xdoctest: +REQUIRES(module:scipy)
        >>> import kwarray
        >>> cost = np.array([
        >>>     [0, 0, 0, 0],
        >>>     [4, 1, 5, -np.inf],
        >>>     [9, 9, np.inf, 4],
        >>>     [9, -2, np.inf, 4],
        >>> ])
        >>> ret = kwarray.mincost_assignment(cost)
        >>> assignment = [(int(a), int(b)) for a, b in ret[0]]
        >>> print('Assignment: {}'.format(assignment))
        >>> print('Total cost: {}'.format(ret[1]))
        Assignment: [(0, 2), (1, 3), (2, 0), (3, 1)]
        Total cost: -inf

    Example:
        >>> # xdoctest: +REQUIRES(module:scipy)
        >>> import kwarray
        >>> cost = np.array([
        >>>     [0, 0, 0, 0],
        >>>     [4, 1, 5, -3],
        >>>     [1, 9, np.inf, 0.1],
        >>>     [np.inf, np.inf, np.inf, 100],
        >>> ])
        >>> ret = kwarray.mincost_assignment(cost)
        >>> assignment = [(int(a), int(b)) for a, b in ret[0]]
        >>> print('Assignment: {}'.format(assignment))
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
    # Note: in scipy 1.4 input costs may be infinite, should fix for this case
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
        xdoctest -m kwarray.algo_assignment maxvalue_assignment

    Example:
        >>> # xdoctest: +REQUIRES(module:scipy)
        >>> # Costs to match item i in set1 with item j in set2.
        >>> import kwarray
        >>> value = np.array([
        >>>     [9, 2, 1, 3],
        >>>     [4, 1, 5, 5],
        >>>     [9, 9, 2, 4],
        >>>     [-1, -1, -1, -1],
        >>> ])
        >>> ret = kwarray.maxvalue_assignment(value)
        >>> # Note, depending on the scipy version the assignment might change
        >>> # but the value should always be the same.
        >>> print('Total value: {}'.format(ret[1]))
        Total value: 23.0
        >>> print('Assignment: {}'.format(ret[0]))  # xdoc: +IGNORE_WANT
        Assignment: [(0, 0), (1, 3), (2, 1)]

        >>> ret = kwarray.maxvalue_assignment(np.array([[np.inf]]))
        >>> assignment = [(int(a), int(b)) for a, b in ret[0]]
        >>> print('Assignment: {}'.format(assignment))
        >>> print('Total value: {}'.format(ret[1]))
        Assignment: [(0, 0)]
        Total value: inf

        >>> ret = kwarray.maxvalue_assignment(np.array([[0]]))
        >>> assignment = [(int(a), int(b)) for a, b in ret[0]]
        >>> print('Assignment: {}'.format(assignment))
        >>> print('Total value: {}'.format(ret[1]))
        Assignment: []
        Total value: 0
    """
    cost = (-value).astype(float)
    cost[value <= 0] = np.inf  # dont take anything with non-positive value
    assignment, cost_tot = mincost_assignment(cost)
    value_tot = -cost_tot
    return assignment, value_tot


def _benchmark_sparse_vs_dense():
    ...


def _demo_assign_problem(n, m, num_vals, maximize, rng):
    ids1 = rng.randint(0, n, size=num_vals).tolist()
    ids2 = rng.randint(0, m, size=num_vals).tolist()
    # values = rng.rand(num_vals)
    values = rng.randint(1, 10, num_vals)

    maximize = rng.rand() > 0.5

    if maximize:
        fill_value = -1
    else:
        fill_value = np.inf

    value_matrix = np.full((n, m), fill_value=fill_value, dtype=float)

    data = {
        'ids1': ids1,
        'ids2': ids2,
        'values': values,
        'value_matrix': value_matrix,
    }
    return data


def _test_sparse_vs_dense_consistency(n, m, num_vals):
    import kwarray
    import ubelt as ub
    rng = kwarray.ensure_rng(0)

    max_size = 5
    max_size = 100
    num_trials = 100
    maximize = False

    def randint_loguniform(low: int, high: int, rng) -> int:
        import math
        if low < 1 or high < low:
            raise ValueError("low must be >= 1 and high >= low")

        log_low = math.log10(low)
        log_high = math.log10(high)

        rand_log = rng.uniform(log_low, log_high)
        result = int(10 ** rand_log)

        # Ensure result falls within the desired bounds due to rounding
        return min(max(result, low), high)

    for _ in ub.ProgIter(range(num_trials), desc='testing'):
        size = randint_loguniform(2, max_size, rng)
        n = rng.randint(1, size)
        m = rng.randint(1, size)

        num_vals = rng.randint(0, n * m)

        ids1 = rng.randint(0, n, size=num_vals).tolist()
        ids2 = rng.randint(0, m, size=num_vals).tolist()
        # values = rng.rand(num_vals)
        values = rng.randint(1, 10, num_vals)

        maximize = rng.rand() > 0.5

        if maximize:
            fill_value = -1
        else:
            fill_value = np.inf

        value_matrix = np.full((n, m), fill_value=fill_value, dtype=float)

        if len(ids1):
            flat_idxs = np.ravel_multi_index((ids1, ids2), dims=(n, m))
            value_matrix.ravel()[flat_idxs] = values

        if maximize:
            assign1, total1 = kwarray.maxvalue_assignment(value_matrix)
        else:
            assign1, total1 = kwarray.mincost_assignment(value_matrix)
        # normalize
        assign1 = [(int(u), int(v)) for u, v in assign1]

        assign3, total3 = _sparse_assignment_networkx(ids1, ids2, values, maximize=maximize)
        assert np.isclose(total3, total1)

        assign2, total2 = sparse_assignment(ids1, ids2, values, maximize=maximize)
        assert np.isclose(total2, total1)

        # assign4, total4 = sparse_partial_assignment(ids1, ids2, values, maximize=maximize)
        # assert np.isclose(total4, total1)


def _sparse_assignment_networkx(ids1, ids2, values, maximize: bool = False):
    import networkx as nx
    graph = nx.Graph()
    nodes1 = [('row', k) for k in ids1]
    nodes2 = [('col', k) for k in ids2]

    if maximize:
        edges = [(u, v, {'weight': w}) for u, v, w in zip(nodes1, nodes2, values) if w > 0]
    else:
        edges = [(u, v, {'weight': w}) for u, v, w in zip(nodes1, nodes2, values)]

    graph.add_nodes_from(nodes1)
    graph.add_nodes_from(nodes2)
    graph.add_edges_from(edges)

    if maximize:
        edge_assignments = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=True)
    else:
        edge_assignments = nx.algorithms.matching.min_weight_matching(graph)

    assignments = []
    total = 0
    for u, v in edge_assignments:
        if u[0] == 'row':
            k1 = u[1]
            k2 = v[1]
        else:
            k1 = v[1]
            k2 = u[1]
        w = graph.edges[(u, v)]['weight']
        total += w
        assignments.append((k1, k2))
    return assignments, total


# Not working
# def sparse_partial_assignment(ids1, ids2, values, maximize=False):
#     """
#     References:
#         https://github.com/scipy/scipy/issues/14041#issuecomment-875530763
#     """
#     import numpy as np
#     import scipy.sparse
#     from scipy.sparse.csgraph import min_weight_full_bipartite_matching

#     if not (len(ids1) == len(ids2) == len(values)):
#         raise ValueError("Input lists must have equal length")

#     def unique(seq):
#         return list(dict.fromkeys(seq))

#     id1_unique = unique(ids1)
#     id2_unique = unique(ids2)

#     id1_to_idx = {k: i for i, k in enumerate(id1_unique)}
#     id2_to_idx = {k: i for i, k in enumerate(id2_unique)}
#     idx_to_id1 = {i: k for k, i in id1_to_idx.items()}
#     idx_to_id2 = {i: k for k, i in id2_to_idx.items()}

#     n1, n2 = len(id1_unique), len(id2_unique)
#     values = np.asarray(values, dtype=float)

#     # Build cost matrix (non-square)
#     cost = scipy.sparse.lil_matrix((n1, n2), dtype=float)
#     for a, b, v in zip(ids1, ids2, values):
#         i = id1_to_idx[a]
#         j = id2_to_idx[b]
#         cost[i, j] = -v if maximize else v

#     # --- Now we augment this cost matrix to support partial match ---
#     maxval = np.max(np.abs(values)) if values.size else 1.0
#     sentinel = 4 * maxval * max(n1, n2)

#     if maximize:
#         sentinel = -sentinel

#     upper_right = scipy.sparse.eye(n1, format="coo") * 0  # optional
#     lower_left = scipy.sparse.eye(n2, format="coo") * sentinel

#     bigraph = scipy.sparse.bmat(
#         [[cost, upper_right],
#          [lower_left, cost.T]],
#         format="csr"
#     )

#     # Shift costs so everything is > 0
#     bigraph.data -= bigraph.data.min()
#     bigraph.data += 1

#     # Run full matching
#     row_idx, col_idx = min_weight_full_bipartite_matching(bigraph)

#     # Extract real matches
#     assignments = []
#     total = 0
#     for r, c in zip(row_idx, col_idx):
#         if r < n1 and c < n2:
#             a = idx_to_id1[r]
#             b = idx_to_id2[c]
#             assignments.append((a, b))
#             total += cost[r, c]

#     return assignments, total


def sparse_assignment(ids1, ids2, values, maximize: bool = False, dense_fallback='auto'):
    """
    Solves the sparse assignment problem using a padded square matrix.

    Args:
        ids1 (List[Any]): Row identifiers.
        ids2 (List[Any]): Column identifiers.
        values (ndarray[float]): Scores or costs associated with (ids1, ids2) pairs.
        maximize (bool): If True, solve for maximum score. Else, minimize cost.

    Returns:
        Tuple[List[Tuple[Any, Any]], float]:
            - A list of (ids1, ids2) assignments (partial if some cannot be matched)
            - The total cost or score

    TODO:
        * Abstract the different assignment algorithms into a class with better
          backends to select the most efficient one for the type of inputs and
          use-case.

    Example:
        >>> # xdoctest: +REQUIRES(module:scipy)
        >>> from kwarray.algo_assignment import *  # NOQA
        >>> ids1 = ['a', 'a', 'b']
        >>> ids2 = ['x', 'y', 'x']
        >>> values = [5, 2, 4]
        >>> result = sparse_assignment(ids1, ids2, values, maximize=True)
        >>> print(f'result = {ub.urepr(result, nl=0)}')
        result = ([('a', 'y'), ('b', 'x')], 6.0)

    Example:
        >>> # xdoctest: +REQUIRES(module:scipy)
        >>> ids1 = ['a', 'b', 'c', 'c']
        >>> ids2 = ['x', 'y', 'p', 'q']
        >>> values = [2, 3, 1, 10]
        >>> sparse_assignment(ids1, ids2, values, maximize=True)
        >>> sparse_assignment(ids1, ids2, values, maximize=False)
        >>> sparse_assignment(ids2, ids1, values, maximize=False)

    Example:
        >>> # xdoctest: +REQUIRES(module:scipy)
        >>> ids1 = [  1,  2,  4,  5,   6,  7]
        >>> ids2 = [  1,  2,  3,  4,   3,  5]
        >>> value = values = [ 29,  4, 12, 20,  10, 21]
        >>> sparse_assignment(ids1, ids2, values, maximize=True)
        >>> sparse_assignment(ids1, ids2, values, maximize=False)
    """
    # from scipy.sparse import csr_matrix
    from scipy.sparse import lil_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching
    # from scipy.sparse.csgraph import maximum_bipartite_matching

    if not (len(ids1) == len(ids2) == len(values)):
        raise ValueError("Input lists must have equal length")

    def unique(seq):
        return list(dict.fromkeys(seq))

    id1_unique = unique(ids1)
    id2_unique = unique(ids2)
    values = np.asarray(values)

    id1_to_idx = {k: i for i, k in enumerate(id1_unique)}
    id2_to_idx = {k: i for i, k in enumerate(id2_unique)}
    idx_to_id1 = {i: k for k, i in id1_to_idx.items()}
    idx_to_id2 = {i: k for k, i in id2_to_idx.items()}

    n1, n2 = len(id1_unique), len(id2_unique)
    n = max(n1, n2)

    # Build sparse cost matrix (square n x n)
    cost_lil = lil_matrix((n, n), dtype=float)
    for a, b, v in zip(ids1, ids2, values):
        i = id1_to_idx[a]
        j = id2_to_idx[b]
        cost_lil[i, j] = v

    # Replace implicit inf with a very large number for padded area
    # We set large values for missing entries in padded space
    if len(values):
        approx_inf = 2 * (n + values.max())
    else:
        approx_inf = 9001

    approx_fill = -approx_inf if maximize else approx_inf

    # Fill padded rows/cols with approx_inf so the solver will assign them
    for i in range(n1, n):
        for j in range(n):
            cost_lil[i, j] = approx_fill
    for j in range(n2, n):
        for i in range(n):
            cost_lil[i, j] = approx_fill
    cost_csr = cost_lil.tocsr()

    # idxs = maximum_bipartite_matching(cost_csr)
    # cost_csr.toarray()[idxs]

    # See issue about forcing a full solution
    # https://github.com/scipy/scipy/issues/14041
    try:
        row_ind, col_ind = min_weight_full_bipartite_matching(cost_csr, maximize=maximize)
    except ValueError:
        if dense_fallback == 'auto':
            # probably better to just use dense
            dense_fallback = n < 1_000_000

        if dense_fallback:
            # HACK: fallback to the dense version, which does work.
            fill_value = -np.inf if maximize else np.inf
            dense = np.full((n, n), fill_value=fill_value)
            for a, b, v in zip(ids1, ids2, values):
                i = id1_to_idx[a]
                j = id2_to_idx[b]
                dense[i, j] = v
            if maximize:
                _assigns, _ = maxvalue_assignment(dense)
            else:
                _assigns, _ = mincost_assignment(dense)
            row_ind, col_ind = list(zip(*_assigns))
        else:
            # networkx fallback
            return _sparse_assignment_networkx(ids1, ids2, values, maximize=maximize)

    assignments = []
    total = 0.0
    for r, c in zip(row_ind, col_ind):
        if r < n1 and c < n2:
            cost = cost_csr[r, c]
            if cost < approx_inf:
                a = idx_to_id1[r]
                b = idx_to_id2[c]
                assignments.append((a, b))
                total += cost
    return assignments, total

if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m kwarray.algo_assignment
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
