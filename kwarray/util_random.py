# -*- coding: utf-8 -*-
"""
Handle and interchange between different random number generators (numpy,
python, torch, ...). Also defines useful random iterator functions and
:func:`ensure_rng`.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import random
import itertools as it

_SEED_MAX = int(2 ** 32 - 1)


def seed_global(seed, offset=0):
    """
    Seeds the python, numpy, and torch global random states

    Args:
        seed (int): seed to use
        offset (int, optional): if specified, uses a different seed for each
            global random state separated by this offset.
    """
    random.seed((seed) % _SEED_MAX)
    np.random.seed((seed + offset) % _SEED_MAX)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.random.manual_seed((seed + 2 * offset) % _SEED_MAX)
        torch.cuda.manual_seed_all((seed + 3 * offset) % _SEED_MAX)


def shuffle(items, rng=None):
    """
    Shuffles a list inplace and then returns it for convinience

    Args:
        items (list or ndarray): list to shuffle
        rng (RandomState or int): seed or random number gen

    Returns:
        list: this is the input, but returned for convinience

    Example:
        >>> list1 = [1, 2, 3, 4, 5, 6]
        >>> list2 = shuffle(list(list1), rng=1)
        >>> assert list1 != list2
        >>> result = str(list2)
        >>> print(result)
        [3, 2, 5, 1, 4, 6]
    """
    rng = ensure_rng(rng)
    rng.shuffle(items)
    return items


def random_combinations(items, size, num=None, rng=None):
    """
    Yields ``num`` combinations of length ``size`` from items in random order

    Args:
        items (List): pool of items to choose from
        size (int): number of items in each combination
        num (None, default=None): number of combinations to generate
        rng (int | RandomState, default=None): seed or random number generator

    Yields:
        Tuple: a random combination of ``items`` of length ``size``.

    Example:
        >>> import ubelt as ub
        >>> items = list(range(10))
        >>> size = 3
        >>> num = 5
        >>> rng = 0
        >>> # xdoctest: +IGNORE_WANT
        >>> combos = list(random_combinations(items, size, num, rng))
        >>> print('combos = {}'.format(ub.repr2(combos, nl=1)))
        combos = [
            (0, 6, 9),
            (4, 7, 8),
            (4, 6, 7),
            (2, 3, 5),
            (1, 2, 4),
        ]

    Example:
        >>> import ubelt as ub
        >>> items = list(zip(range(10), range(10)))
        >>> # xdoctest: +IGNORE_WANT
        >>> combos = list(random_combinations(items, 3, num=5, rng=0))
        >>> print('combos = {}'.format(ub.repr2(combos, nl=1)))
        combos = [
            ((0, 0), (6, 6), (9, 9)),
            ((4, 4), (7, 7), (8, 8)),
            ((4, 4), (6, 6), (7, 7)),
            ((2, 2), (3, 3), (5, 5)),
            ((1, 1), (2, 2), (4, 4)),
        ]

    """
    import scipy.special
    rng = ensure_rng(rng, api='python')
    num_ = np.inf if num is None else num
    # Ensure we dont request more than is possible
    n_max = int(scipy.special.comb(len(items), size))
    num_ = min(n_max, num_)
    if num is not None and num_ > n_max // 2:
        # If num is too big just generate all combinations and shuffle them
        combos = list(it.combinations(items, size))
        rng.shuffle(combos)
        for combo in combos[:num]:
            yield combo
    else:
        # Otherwise yield randomly until we get something we havent seen
        items = list(items)
        combos = set()
        while len(combos) < num_:
            # combo = tuple(sorted(rng.choice(items, size, replace=False)))
            combo = tuple(sorted(rng.sample(items, size)))
            if combo not in combos:
                # TODO: store indices instead of combo values
                combos.add(combo)
                yield combo


def random_product(items, num=None, rng=None):
    """
    Yields ``num`` items from the cartesian product of items in a random order.

    Args:
        items (List[Sequence]):
            items to get caresian product of packed in a list or tuple.
            (note this deviates from api of :func:`itertools.product`)

        num (int, default=None):
            maximum number of items to generate. If None, all

        rng (random.Random | np.random.RandomState | int):
            random number generator

    Yields:
        Tuple: a random item in the cartesian product

    Example:
        >>> import ubelt as ub
        >>> items = [(1, 2, 3), (4, 5, 6, 7)]
        >>> rng = 0
        >>> # xdoctest: +IGNORE_WANT
        >>> products = list(random_product(items, rng=0))
        >>> print(ub.repr2(products, nl=0))
        [(3, 4), (1, 7), (3, 6), (2, 7),... (1, 6), (2, 5), (2, 4)]
        >>> products = list(random_product(items, num=3, rng=0))
        >>> print(ub.repr2(products, nl=0))
        [(3, 4), (1, 7), (3, 6)]

    Example:
        >>> # xdoctest: +REQUIRES(--profile)
        >>> rng = ensure_rng(0)
        >>> items = [np.array([15, 14]), np.array([27, 26]),
        >>>          np.array([21, 22]), np.array([32, 31])]
        >>> num = 2
        >>> for _ in range(100):
        >>>     list(random_product(items, num=num, rng=rng))
    """
    # NUMPY_RNG = True  # toggle new speedup on
    try:
        if not isinstance(items, (list, tuple)):
            raise TypeError
        idx_cards = np.array([len(g) for g in items], dtype=np.uint32)
    except (TypeError, AttributeError):
        items = [list(g) for g in items]
        idx_cards = np.array([len(g) for g in items], dtype=np.uint32)

    ndims = len(items)
    # max_num = np.prod(idx_cards.astype(np.float))
    max_num = np.multiply.reduce(idx_cards.astype(np.float32))

    if num is None:
        num = max_num
    else:
        num = min(num, max_num)
        # if num > max_num:
        #     raise ValueError('num exceedes maximum number of products')

    # TODO: make this more efficient when num is large
    if max_num > 100 and num > max_num // 2:

        rng = ensure_rng(rng, 'python')

        for prod in shuffle(list(it.product(*items)), rng=rng):
            yield prod
    else:
        if True:  # NUMPY_RNG
            rng = ensure_rng(rng, 'numpy')
            # Need to use least-common-multiple so the mod of all idxs
            # are equally likely
            card_lcm = np.lcm.reduce(idx_cards)
        else:
            rng = ensure_rng(rng, 'python')

        seen = set()
        while len(seen) < num:

            if True:  # NUMPY_RNG
                idxs = rng.randint(0, card_lcm, size=ndims, dtype=idx_cards.dtype)
                idxs %= idx_cards
                idxs = tuple(idxs.tolist())
            else:
                idxs = tuple(rng.randint(0, n - 1) for n in idx_cards)

            if idxs not in seen:
                seen.add(idxs)
                prod = tuple(g[x] for g, x in zip(items, idxs))
                yield prod


def _npstate_to_pystate(npstate):
    """
    Convert state of a NumPy RandomState object to a state
    that can be used by Python's Random. Derived from [1]_.

    References:
        .. [1] https://stackoverflow.com/questions/44313620/convert-randomstate

    Example:
        >>> py_rng = random.Random(0)
        >>> np_rng = np.random.RandomState(seed=0)
        >>> npstate = np_rng.get_state()
        >>> pystate = _npstate_to_pystate(npstate)
        >>> py_rng.setstate(pystate)
        >>> assert np_rng.rand() == py_rng.random()
    """
    PY_VERSION = 3
    version, keys, pos, has_gauss, cached_gaussian_ = npstate
    keys_pos = tuple(map(int, keys)) + (int(pos),)
    cached_gaussian_ = cached_gaussian_ if has_gauss else None
    pystate = (PY_VERSION, keys_pos, cached_gaussian_)
    return pystate


def _pystate_to_npstate(pystate):
    """
    Convert state of a Python Random object to state usable
    by NumPy RandomState. Derived from [2]_.

    References:
        .. [2] https://stackoverflow.com/questions/44313620/convert-randomstate

    Example:
        >>> py_rng = random.Random(0)
        >>> np_rng = np.random.RandomState(seed=0)
        >>> pystate = py_rng.getstate()
        >>> npstate = _pystate_to_npstate(pystate)
        >>> np_rng.set_state(npstate)
        >>> assert np_rng.rand() == py_rng.random()
    """
    NP_VERSION = 'MT19937'
    version, keys_pos_, cached_gaussian_ = pystate
    keys, pos = keys_pos_[:-1], keys_pos_[-1]
    keys = np.array(keys, dtype=np.uint32)
    has_gauss = cached_gaussian_ is not None
    cached_gaussian = cached_gaussian_ if has_gauss else 0.0
    npstate = (NP_VERSION, keys, pos, has_gauss, cached_gaussian)
    return npstate


def _coerce_rng_type(rng):
    """
    Internal method that transforms input seeds into an integer form.
    """
    if rng is None or isinstance(rng, (random.Random, np.random.RandomState)):
        pass
    elif rng is random:
        rng = rng._inst
    elif rng is np.random:
        rng = np.random.mtrand._rand
    elif isinstance(rng, (float, np.floating)):
        rng = float(rng)
        # Coerce the float into an integer
        a, b = rng.as_integer_ratio()
        if b == 1:
            rng = a
        else:
            s = max(a.bit_length(), b.bit_length())
            rng = (b << s) | a
    elif isinstance(rng, (int, np.integer)):
        rng = int(rng)
    else:
        raise TypeError(
            'Cannot coerce {!r} to a random object'.format(type(rng)))
    return rng


def ensure_rng(rng, api='numpy'):
    """
    Coerces input into a random number generator.

    This function is useful for ensuring that your code uses a controlled
    internal random state that is independent of other modules.

    If the input is None, then a global random state is returned.

    If the input is a numeric value, then that is used as a seed to construct a
    random state.

    If the input is a random number generator, then another random number
    generator with the same state is returned. Depending on the api, this
    random state is either return as-is, or used to construct an equivalent
    random state with the requested api.

    Args:
        rng (int | float | numpy.random.RandomState | random.Random | None):
            if None, then defaults to the global rng. Otherwise this can
            be an integer or a RandomState class

        api (str, default='numpy'): specify the type of random number
            generator to use. This can either be 'numpy' for a
            :class:`numpy.random.RandomState` object or 'python' for a
            :class:`random.Random` object.

    Returns:
        (numpy.random.RandomState | random.Random) : rng -
            either a numpy or python random number generator, depending on the
            setting of ``api``.

    Example:
        >>> rng = ensure_rng(None)
        >>> ensure_rng(0).randint(0, 1000)
        684
        >>> ensure_rng(np.random.RandomState(1)).randint(0, 1000)
        37

    Example:
        >>> num = 4
        >>> print('--- Python as PYTHON ---')
        >>> py_rng = random.Random(0)
        >>> pp_nums = [py_rng.random() for _ in range(num)]
        >>> print(pp_nums)
        >>> print('--- Numpy as PYTHON ---')
        >>> np_rng = ensure_rng(random.Random(0), api='numpy')
        >>> np_nums = [np_rng.rand() for _ in range(num)]
        >>> print(np_nums)
        >>> print('--- Numpy as NUMPY---')
        >>> np_rng = np.random.RandomState(seed=0)
        >>> nn_nums = [np_rng.rand() for _ in range(num)]
        >>> print(nn_nums)
        >>> print('--- Python as NUMPY---')
        >>> py_rng = ensure_rng(np.random.RandomState(seed=0), api='python')
        >>> pn_nums = [py_rng.random() for _ in range(num)]
        >>> print(pn_nums)
        >>> assert np_nums == pp_nums
        >>> assert pn_nums == nn_nums

    Example:
        >>> # Test that random modules can be coerced
        >>> import random
        >>> import numpy as np
        >>> ensure_rng(random, api='python')
        >>> ensure_rng(random, api='numpy')
        >>> ensure_rng(np.random, api='python')
        >>> ensure_rng(np.random, api='numpy')

    Ignore:
        >>> np.random.seed(0)
        >>> np.random.randint(0, 10000)
        2732
        >>> np.random.seed(0)
        >>> np.random.mtrand._rand.randint(0, 10000)
        2732
        >>> np.random.seed(0)
        >>> ensure_rng(None).randint(0, 10000)
        2732
        >>> np.random.randint(0, 10000)
        9845
        >>> ensure_rng(None).randint(0, 10000)
        3264
    """
    rng = _coerce_rng_type(rng)

    if api == 'numpy':
        if rng is None:
            # This is the underlying random state of the np.random module
            rng = np.random.mtrand._rand
            # Dont do this because it seeds using dev/urandom
            # rng = np.random.RandomState(seed=None)
        elif isinstance(rng, int):
            rng = np.random.RandomState(seed=rng % _SEED_MAX)
        elif isinstance(rng, random.Random):
            # Convert python to numpy random state
            py_rng = rng
            pystate = py_rng.getstate()
            npstate = _pystate_to_npstate(pystate)
            rng = np_rng = np.random.RandomState(seed=0)
            np_rng.set_state(npstate)
    elif api == 'python':
        if rng is None:
            # This is the underlying random state of the random module
            rng = random._inst
        elif isinstance(rng, int):
            rng = random.Random(rng % _SEED_MAX)
        elif isinstance(rng, np.random.RandomState):
            # Convert numpy to python random state
            np_rng = rng
            npstate = np_rng.get_state()
            pystate = _npstate_to_pystate(npstate)
            rng = py_rng = random.Random(0)
            py_rng.setstate(pystate)
    else:
        raise KeyError('unknown rng api={}'.format(api))
    return rng


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m kwarray.util_random
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
