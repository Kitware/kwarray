import kwarray
import numpy as np
import ubelt as ub


def test_run_avg_cases():
    for use_weights in [0, 1]:
        base = np.array([1, 1, 1, 1, 0, 0, 0, 1])
        run0 = kwarray.RunningStats()
        data0 = base.reshape(8, 1)
        weights0 = np.ones_like(data0) if use_weights else 1
        for _ in range(3):
            run0.update(data0, weights=weights0)
        run1 = kwarray.RunningStats()
        data1 = base.reshape(4, 2)
        weights1 = np.ones_like(data1) if use_weights else 1
        for _ in range(3):
            run1.update(data1, weights=weights1)
        run2 = kwarray.RunningStats()
        data2 = base.reshape(2, 2, 2)
        weights2 = np.ones_like(data2) if use_weights else 1
        for _ in range(3):
            run2.update(data2, weights=weights2)

        for keepdims in [False, True]:
            # Summarizing over everything should be exactly the same
            s0N = run0.summarize(axis=None, keepdims=keepdims)
            s1N = run1.summarize(axis=None, keepdims=keepdims)
            s2N = run2.summarize(axis=None, keepdims=keepdims)
            print('s2N = {}'.format(ub.repr2(s2N, nl=1)))
            # assert ub.util_indexable.indexable_allclose(s0N, s1N, rel_tol=0.0, abs_tol=0.0)
            # assert ub.util_indexable.indexable_allclose(s1N, s2N, rel_tol=0.0, abs_tol=0.0)
            assert s0N['mean'] == 0.625

        for keepdims in [False, True]:
            # Summarizing over everything should be exactly the same
            s1N = run1.summarize(axis=1, keepdims=keepdims)
            s2N = run2.summarize(axis=2, keepdims=keepdims)
            print('s1N = {}'.format(ub.repr2(s1N, nl=1)))
            print('s2N = {}'.format(ub.repr2(s2N, nl=1)))
            s1N = ub.map_vals(np.ravel, s1N)
            s2N = ub.map_vals(np.ravel, s2N)
            assert np.allclose(s1N['mean'], s2N['mean'], rtol=0.0, atol=0.0)
            assert np.allclose(s1N['mean'], np.array([1. , 1. , 0. , 0.5]))


def test_run_avg_with_zero_weight():
    data    = np.array([1, 1, 1, 0, 0, 0, 1, 1.]).reshape(4, 2)
    weights = np.array([1, 0, 0, 1, 0, 1, 1, 1.]).reshape(data.shape)
    run = kwarray.RunningStats()
    stacked_data = []
    stacked_weights = []
    for _ in range(3):
        stacked_data.append(data)
        stacked_weights.append(weights)
        run.update(data, weights=weights)

    C = run.current()
    SN = run.summarize(axis=None)
    S1 = run.summarize(axis=1)
    S0 = run.summarize(axis=0)
    print('C = {}'.format(ub.repr2(C, nl=1)))
    print('SN = {}'.format(ub.repr2(SN, nl=1)))
    print('S1 = {}'.format(ub.repr2(S1, nl=1)))
    print('S0 = {}'.format(ub.repr2(S0, nl=1)))

    stack_d = np.stack(stacked_data, axis=-1)
    stack_w = np.stack(stacked_weights, axis=-1)
    ave0 = np.average(stack_d, weights=stack_w)
    assert np.allclose(SN['mean'].ravel(), ave0)

    ave0 = np.average(stack_d, weights=stack_w, axis=(0, -1))
    assert np.allclose(S0['mean'].ravel(), ave0)

    ave1 = np.average(stack_d, weights=stack_w, axis=(1, -1))
    assert np.allclose(S1['mean'].ravel(), ave1)


def test_fuzzed_random_running():
    """
    We should get close to the same result as np.average in all cases where we
    stack up our data along a new axis and compute the explict average along
    the chosen axis AND -1.
    """
    grid = ub.named_product({
        'shape': [(2, 3, 5), (5,)],
        'use_weights': [0, 1],
        'keepdims': [False, True],
    })

    rng = kwarray.ensure_rng()

    from packaging.version import parse as Version

    for params in grid:
        shape = params['shape']
        use_weights = params['use_weights']
        keepdims = params['keepdims']

        run = kwarray.RunningStats()
        stacked_data = []
        stacked_weights = []
        num = 3
        for _ in range(num):
            data = rng.randint(0, 2, size=shape).astype(float)
            if use_weights:
                weights = rng.rand(*shape)
            else:
                weights = 1

            stacked_data.append(data)
            stacked_weights.append(weights)

            run.update(data, weights=weights)

        C = run.current()
        print('C = {}'.format(ub.repr2(C, nl=1)))
        SN = run.summarize(axis=None, keepdims=keepdims)
        S0 = run.summarize(axis=0, keepdims=keepdims)

        stack_d = np.stack(stacked_data, axis=-1)
        if params['use_weights']:
            stack_w = np.stack(stacked_weights, axis=-1)
            # stack_w = np.broadcast_to(stack_w, stack_d.shape)
            assert np.allclose(run.n, stack_w.sum(axis=-1))
        else:
            stack_w = None

        ave_kw = dict()
        if Version(np.__version__) < Version('1.23.0'):
            if keepdims:
                continue
        else:
            ave_kw = dict(keepdims=keepdims)

        ourN = SN['mean']
        aveN = np.average(stack_d, weights=stack_w, **ave_kw)
        print('ourN = {}'.format(ub.repr2(ourN, nl=1, precision=4)))
        print('aveN = {}'.format(ub.repr2(aveN, nl=1, precision=4)))
        assert np.allclose(ourN, aveN)

        our0 = S0['mean']
        ave0 = np.average(stack_d, weights=stack_w, axis=(0, -1), **ave_kw)
        if keepdims:
            ave0 = ave0[..., 0]
        print('our0 = {}'.format(ub.repr2(our0, nl=1, precision=4)))
        print('ave0 = {}'.format(ub.repr2(ave0, nl=1, precision=4)))
        assert np.allclose(our0, ave0)

        if len(params['shape']) > 1:
            S1 = run.summarize(axis=1, keepdims=keepdims)
            our1 = S1['mean']
            ave1 = np.average(stack_d, weights=stack_w, axis=(1, -1), **ave_kw)
            if keepdims:
                ave1 = ave1[..., 0]
            print('our1 = {}'.format(ub.repr2(our1, nl=1, precision=4)))
            print('ave1 = {}'.format(ub.repr2(ave1, nl=1, precision=4)))
            assert np.allclose(our1, ave1)
