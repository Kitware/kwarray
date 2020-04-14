# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import ubelt as ub
import itertools as it
from collections import OrderedDict


def setcover(candidate_sets_dict, items=None, set_weights=None,
             item_values=None, max_weight=None, algo='approx'):
    """
    Finds a feasible solution to the minimum weight maximum value set cover.
    The quality and runtime of the solution will depend on the backend
    algorithm selected.

    Args:
        candidate_sets_dict (Dict[Hashable, List[Hashable]]):
            a dictionary where keys are the candidate set ids and each value is
            a candidate cover set.

        items (Hashable, optional): the set of all items to be covered,
            if not specified, it is infered from the candidate cover sets

        set_weights (Dict, optional): maps candidate set ids to a cost
            for using this candidate cover in the solution. If not specified
            the weight of each candiate cover defaults to 1.

        item_values (Dict, optional): maps each item to a value we get for
            returning this item in the solution. If not specified the value
            of each item defaults to 1.

        max_weight (float): if specified, the total cost of the returned cover
            is constrained to be less than this number.

        algo (str): specifies which algorithm to use. Can either be
            'approx' for the greedy solution or 'exact' for the globally
            optimal solution. Note the 'exact' algorithm solves an
            integer-linear-program, which can be very slow and requires
            the `pulp` package to be installed.

    Returns:
        Dict: a subdict of candidate_sets_dict containing the chosen solution.

    Example:
        >>> candidate_sets_dict = {
        >>>     'a': [1, 2, 3, 8, 9, 0],
        >>>     'b': [1, 2, 3, 4, 5],
        >>>     'c': [4, 5, 7],
        >>>     'd': [5, 6, 7],
        >>>     'e': [6, 7, 8, 9, 0],
        >>> }
        >>> greedy_soln = setcover(candidate_sets_dict, algo='greedy')
        >>> print('greedy_soln = {}'.format(ub.repr2(greedy_soln, nl=0)))
        greedy_soln = {'a': [1, 2, 3, 8, 9, 0], 'c': [4, 5, 7], 'd': [5, 6, 7]}
        >>> # xdoc: +REQUIRES(module:pulp)
        >>> exact_soln = setcover(candidate_sets_dict, algo='exact')
        >>> print('exact_soln = {}'.format(ub.repr2(exact_soln, nl=0)))
        exact_soln = {'b': [1, 2, 3, 4, 5], 'e': [6, 7, 8, 9, 0]}
    """
    if algo in ['approx', 'greedy']:
        return _setcover_greedy_old(candidate_sets_dict, items=items,
                                    set_weights=set_weights,
                                    item_values=item_values,
                                    max_weight=max_weight)
    elif algo in ['exact', 'ilp']:
        return _setcover_ilp(candidate_sets_dict, items=items,
                             set_weights=set_weights, item_values=item_values,
                             max_weight=max_weight)
    else:
        raise KeyError(algo)


def _setcover_greedy_old(candidate_sets_dict, items=None, set_weights=None,
                         item_values=None, max_weight=None):
    """
    Benchmark:
        items = np.arange(10000)
        candidate_sets_dict = {}
        for i in range(1000):
            candidate_sets_dict[i] = np.random.choice(items, 200).tolist()

        _setcover_greedy_new(candidate_sets_dict) == _setcover_greedy_old(candidate_sets_dict)
        _ = nh.util.profile_onthefly(_setcover_greedy_new)(candidate_sets_dict)
        _ = nh.util.profile_onthefly(_setcover_greedy_old)(candidate_sets_dict)

        import ubelt as ub
        for timer in ub.Timerit(3, bestof=1, label='time'):
            with timer:
                len(_setcover_greedy_new(candidate_sets_dict))

        import ubelt as ub
        for timer in ub.Timerit(3, bestof=1, label='time'):
            with timer:
                len(_setcover_greedy_old(candidate_sets_dict))
    """
    solution_cover = {}

    if len(candidate_sets_dict) == 0:
        # O(1) optimal solution, we did it!
        return solution_cover

    # If candset_weights or item_values not given use the length as defaults
    if set_weights is None:
        get_weight = len
    else:
        def get_weight(solution_cover):
            return sum(set_weights[key] for key in solution_cover.keys())
    if item_values is None:
        get_value = len
    else:
        def get_value(vals):
            return sum(item_values[v] for v in vals)
    if max_weight is None:
        max_weight = get_weight(candidate_sets_dict)

    avail_covers = {key: set(val) for key, val in candidate_sets_dict.items()}
    avail_keys, avail_vals = zip(*sorted(avail_covers.items()))
    avail_keys = list(avail_keys)
    avail_vals = list(avail_vals)

    # While we still need covers
    while get_weight(solution_cover) < max_weight and len(avail_keys) > 0:
        # Find candiate set with the most uncovered items
        uncovered_values = list(map(get_value, avail_vals))
        chosen_idx = ub.argmax(uncovered_values)
        if uncovered_values[chosen_idx] <= 0:
            # needlessly adding value-less items
            break
        chosen_key = avail_keys[chosen_idx]
        # Add values in this key to the cover
        chosen_set = avail_covers[chosen_key]
        solution_cover[chosen_key] = candidate_sets_dict[chosen_key]
        # Remove chosen set from available options and covered items
        # from remaining available sets
        del avail_keys[chosen_idx]
        del avail_vals[chosen_idx]
        for vals in avail_vals:
            vals.difference_update(chosen_set)
    return solution_cover


def _setcover_greedy_new(candidate_sets_dict, items=None, set_weights=None,
                         item_values=None, max_weight=None):
    """
    Implements Johnson's / Chvatal's greedy set-cover approximation algorithms.

    The approximation gaurentees depend on specifications of set weights and
    item values

    Running time:
        N = number of universe items
        C = number of candidate covering sets

        Worst case running time is: O(C^2 * CN)
            (note this is via simple analysis, the big-oh might be better)

    Set Cover: log(len(items) + 1) approximation algorithm
    Weighted Maximum Cover: 1 - 1/e == .632 approximation algorithm
    Generalized maximum coverage is not implemented

    References:
        https://en.wikipedia.org/wiki/Maximum_coverage_problem

    Notes:
        # pip install git+git://github.com/tangentlabs/django-oscar.git#egg=django-oscar.
        # TODO: wrap https://github.com/martin-steinegger/setcover/blob/master/SetCover.cpp
        # pip install SetCoverPy
        # This is actually much slower than my implementation
        from SetCoverPy import setcover
        g = setcover.SetCover(full_overlaps, cost=np.ones(len(full_overlaps)))
        g.greedy()
        keep = np.where(g.s)[0]

    Example:
        >>> candidate_sets_dict = {
        >>>     'a': [1, 2, 3, 8, 9, 0],
        >>>     'b': [1, 2, 3, 4, 5],
        >>>     'c': [4, 5, 7],
        >>>     'd': [5, 6, 7],
        >>>     'e': [6, 7, 8, 9, 0],
        >>> }
        >>> greedy_soln = _setcover_greedy_new(candidate_sets_dict)
        >>> #print(repr(greedy_soln))
        ...
        >>> print('greedy_soln = {}'.format(ub.repr2(greedy_soln, nl=0)))
        greedy_soln = {'a': [1, 2, 3, 8, 9, 0], 'c': [4, 5, 7], 'd': [5, 6, 7]}

    Example:
        >>> candidate_sets_dict = {
        >>>     'a': [1, 2, 3, 8, 9, 0],
        >>>     'b': [1, 2, 3, 4, 5],
        >>>     'c': [4, 5, 7],
        >>>     'd': [5, 6, 7],
        >>>     'e': [6, 7, 8, 9, 0],
        >>> }
        >>> items = list(set(it.chain(*candidate_sets_dict.values())))
        >>> set_weights = {i: 1 for i in candidate_sets_dict.keys()}
        >>> item_values = {e: 1 for e in items}
        >>> greedy_soln = _setcover_greedy_new(candidate_sets_dict,
        >>>                             item_values=item_values,
        >>>                             set_weights=set_weights)
        >>> print('greedy_soln = {}'.format(ub.repr2(greedy_soln, nl=0)))
        greedy_soln = {'a': [1, 2, 3, 8, 9, 0], 'c': [4, 5, 7], 'd': [5, 6, 7]}

    Example:
        >>> candidate_sets_dict = {}
        >>> greedy_soln = _setcover_greedy_new(candidate_sets_dict)
        >>> print('greedy_soln = {}'.format(ub.repr2(greedy_soln, nl=0)))
        greedy_soln = {}
    """
    if len(candidate_sets_dict) == 0:
        # O(1) optimal solution, we did it!
        return {}

    solution_cover = {}
    solution_weight = 0

    if items is None:
        items = list(set(it.chain(*candidate_sets_dict.values())))

    # Inverted index
    item_to_keys = {item: set() for item in items}
    # This is actually a fair bit faster than the non-comprehension version
    [item_to_keys[item].add(key)
     for key, vals in candidate_sets_dict.items()
     for item in vals]

    # If set_weights or item_values not given use the length as defaults
    if set_weights is None:
        get_weight = len
    else:
        # TODO: we can improve this with bookkeeping
        def get_weight(solution_cover):
            return sum(set_weights[key] for key in solution_cover.keys())

    if item_values is None:
        get_value = len
    else:
        def get_value(vals):
            return sum(item_values[v] for v in vals)
    if max_weight is None:
        max_weight = get_weight(candidate_sets_dict)

    avail_covers = OrderedDict([
        (key, set(vals))
        for key, vals in sorted(candidate_sets_dict.items())
    ])
    avail_totals = OrderedDict([
        (key, get_value(vals))
        for key, vals in avail_covers.items()
    ])

    print('avail_covers = {}'.format(ub.repr2(avail_covers, nl=1)))
    print('avail_totals = {}'.format(ub.repr2(avail_totals, nl=1)))

    # While we still need covers
    while solution_weight < max_weight and len(avail_covers) > 0:
        # Find candiate set with the most valuable uncovered items
        chosen_key = ub.argmax(avail_totals)
        if avail_totals[chosen_key] <= 0:
            # needlessly adding value-less covering set
            break

        print('-----')
        print('CHOOSE COVER SET = {!r}'.format(chosen_key))

        # Add values in this key to the cover
        chosen_items = avail_covers[chosen_key]
        solution_cover[chosen_key] = candidate_sets_dict[chosen_key]

        # Update the solution weight
        chosen_weight = (1 if set_weights is None else set_weights[chosen_key])
        solution_weight += chosen_weight

        # Remove chosen covering set from available options
        del avail_covers[chosen_key]
        del avail_totals[chosen_key]

        # For each chosen item, find the other sets that it belongs to
        modified_keys = set()
        for item in chosen_items:
            # Update the inverted index
            new_keys = item_to_keys[item]
            new_keys.remove(chosen_key)
            item_to_keys[item] = new_keys
            # And mark the non-chosen reamining cover sets as modified
            modified_keys.update(new_keys)
        # Then update and recompute the value of the modified sets
        for key in modified_keys:
            avail_covers[key].difference_update(chosen_items)
            newval = get_value(avail_covers[key])
            avail_totals[key] = newval

        print('avail_covers = {}'.format(ub.repr2(avail_covers, nl=1)))
        print('avail_totals = {}'.format(ub.repr2(avail_totals, nl=1)))

    print('solution_cover = {!r}'.format(solution_cover))
    return solution_cover


def _setcover_ilp(candidate_sets_dict, items=None, set_weights=None,
                  item_values=None, max_weight=None, verbose=False):
    """
    Set cover / Weighted Maximum Cover exact algorithm using an integer linear
    program.

    TODO:
        - [ ] Use CPLEX solver if available

    https://en.wikipedia.org/wiki/Maximum_coverage_problem

    Example:
        >>> # xdoc: +REQUIRES(module:pulp)
        >>> candidate_sets_dict = {}
        >>> exact_soln = _setcover_ilp(candidate_sets_dict)
        >>> print('exact_soln = {}'.format(ub.repr2(exact_soln, nl=0)))
        exact_soln = {}

    Example:
        >>> # xdoc: +REQUIRES(module:pulp)
        >>> candidate_sets_dict = {
        >>>     'a': [1, 2, 3, 8, 9, 0],
        >>>     'b': [1, 2, 3, 4, 5],
        >>>     'c': [4, 5, 7],
        >>>     'd': [5, 6, 7],
        >>>     'e': [6, 7, 8, 9, 0],
        >>> }
        >>> items = list(set(it.chain(*candidate_sets_dict.values())))
        >>> set_weights = {i: 1 for i in candidate_sets_dict.keys()}
        >>> item_values = {e: 1 for e in items}
        >>> exact_soln1 = _setcover_ilp(candidate_sets_dict,
        >>>                             item_values=item_values,
        >>>                             set_weights=set_weights)
        >>> exact_soln2 = _setcover_ilp(candidate_sets_dict)
        >>> assert exact_soln1 == exact_soln2
    """
    try:
        import pulp
    except ImportError:
        print('ERROR: must install pulp to use ILP setcover solver')
        raise

    if len(candidate_sets_dict) == 0:
        return {}

    if items is None:
        items = list(set(it.chain(*candidate_sets_dict.values())))

    if item_values is None and set_weights is None and max_weight is None:
        # This is the most basic set cover problem
        # Formulate integer program
        prob = pulp.LpProblem("Set Cover", pulp.LpMinimize)
        # Solution variable indicates if set it chosen or not
        set_indices = candidate_sets_dict.keys()
        x = pulp.LpVariable.dicts(name='x', indexs=set_indices,
                                  lowBound=0, upBound=1, cat=pulp.LpInteger)
        # minimize the number of sets
        prob.objective = sum(x[i] for i in set_indices)
        # subject to
        for e in items:
            # each element is covered
            containing_sets = [i for i in set_indices if e in candidate_sets_dict[i]]
            prob.add(sum(x[i] for i in containing_sets) >= 1)
        # Solve using with solver like CPLEX, GLPK, or SCIP.
        #pulp.CPLEX().solve(prob)
        pulp.PULP_CBC_CMD().solve(prob)
        # Read solution
        solution_keys = [i for i in set_indices if x[i].varValue]
        solution_cover = {i: candidate_sets_dict[i] for i in solution_keys}
        # Print summary
        if verbose:
            print(prob)
            print('OPT:')
            print('\n'.join(['    %s = %s' % (x[i].name, x[i].varValue) for i in set_indices]))
            print('solution_cover = %r' % (solution_cover,))
    else:
        if set_weights is None:
            set_weights = {i: 1 for i in candidate_sets_dict.keys()}
        if item_values is None:
            item_values = {e: 1 for e in items}
        if max_weight is None:
            max_weight = sum(set_weights[i] for i in candidate_sets_dict.keys())
        prob = pulp.LpProblem("Maximum Cover", pulp.LpMaximize)
        # Solution variable indicates if set it chosen or not
        item_indicies = items
        set_indices = candidate_sets_dict.keys()
        x = pulp.LpVariable.dicts(name='x', indexs=set_indices,
                                  lowBound=0, upBound=1, cat=pulp.LpInteger)
        y = pulp.LpVariable.dicts(name='y', indexs=item_indicies,
                                  lowBound=0, upBound=1, cat=pulp.LpInteger)
        r = pulp.LpVariable.dicts(name='r', indexs=item_indicies)
        # maximize the value of the covered items
        primary_objective = sum(item_values[e] * y[e] for e in item_indicies)
        # minimize the number of sets used (make sure it does not influence the chosen primary objective)
        # This is only possible when values are non-negative
        # TODO: minimize redundency
        min_influence = min(item_values.values())
        secondary_weight = min_influence / (1.1 * len(set_indices))
        secondary_objective = (sum(-x[i] for i in set_indices)) * secondary_weight
        #
        prob.objective = primary_objective + secondary_objective
        # subject to
        # no more than the maximum weight
        prob.add(sum(x[i] * set_weights[i] for i in set_indices) <= max_weight)
        # If an item is chosen than at least one set containing it is chosen
        for e in item_indicies:
            containing_sets = [i for i in set_indices if e in candidate_sets_dict[i]]
            if len(containing_sets) > 0:
                prob.add(sum(x[i] for i in containing_sets) >= y[e])
                # record number of times each item is covered
                prob.add(sum(x[i] for i in containing_sets) == r[e])
        # Solve using with solver like CPLEX, GLPK, or SCIP.
        #pulp.CPLEX().solve(prob)
        pulp.PULP_CBC_CMD().solve(prob)
        # Read solution
        solution_keys = [i for i in set_indices if x[i].varValue]
        solution_cover = {i: candidate_sets_dict[i] for i in solution_keys}
        # Print summary
        if verbose:
            print(prob)
            print('OPT:')
            print('\n'.join(['    %s = %s' % (x[i].name, x[i].varValue) for i in set_indices]))
            print('\n'.join(['    %s = %s' % (y[i].name, y[i].varValue) for i in item_indicies]))
            print('solution_cover = %r' % (solution_cover,))
    return solution_cover
