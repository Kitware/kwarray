import numpy as np
import ubelt as ub
import kwarray
from kwarray import distributions as dmod


def test_rng_case1():
    """
    Reproduce a bug from kwarray.__version__ < 0.6.13
    """
    rng = 0
    rng = kwarray.ensure_rng(rng)
    a = dmod.Distribution.random(rng=rng)
    print(a)
    values1 = a.sample(10)

    rng = 0
    rng = kwarray.ensure_rng(rng)
    a = dmod.Distribution.random(rng=rng)
    print(a)
    values2 = a.sample(10)
    assert np.allclose(values1, values2)


def test_normal_distribution_with_random_seed():
    rng = kwarray.ensure_rng(0)
    distri = dmod.Normal(rng=rng)
    values1 = distri.sample(10)
    print('values1 = {}'.format(ub.urepr(values1, nl=1)))

    rng = kwarray.ensure_rng(0)
    distri = dmod.Normal(rng=rng)
    values2 = distri.sample(10)
    print('values2 = {}'.format(ub.urepr(values2, nl=1)))
    assert np.allclose(values1, values2)
