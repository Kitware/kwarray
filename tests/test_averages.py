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
            print('s2N = {}'.format(ub.urepr(s2N, nl=1)))
            # assert ub.util_indexable.indexable_allclose(s0N, s1N, rel_tol=0.0, abs_tol=0.0)
            # assert ub.util_indexable.indexable_allclose(s1N, s2N, rel_tol=0.0, abs_tol=0.0)
            assert s0N['mean'] == 0.625

        for keepdims in [False, True]:
            # Summarizing over everything should be exactly the same
            s1N = run1.summarize(axis=1, keepdims=keepdims)
            s2N = run2.summarize(axis=2, keepdims=keepdims)
            print('s1N = {}'.format(ub.urepr(s1N, nl=1)))
            print('s2N = {}'.format(ub.urepr(s2N, nl=1)))
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
    print('C = {}'.format(ub.urepr(C, nl=1)))
    print('SN = {}'.format(ub.urepr(SN, nl=1)))
    print('S1 = {}'.format(ub.urepr(S1, nl=1)))
    print('S0 = {}'.format(ub.urepr(S0, nl=1)))

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
        print('C = {}'.format(ub.urepr(C, nl=1)))
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
        print('ourN = {}'.format(ub.urepr(ourN, nl=1, precision=4)))
        print('aveN = {}'.format(ub.urepr(aveN, nl=1, precision=4)))
        assert np.allclose(ourN, aveN)

        our0 = S0['mean']
        ave0 = np.average(stack_d, weights=stack_w, axis=(0, -1), **ave_kw)
        if keepdims:
            ave0 = ave0[..., 0]
        print('our0 = {}'.format(ub.urepr(our0, nl=1, precision=4)))
        print('ave0 = {}'.format(ub.urepr(ave0, nl=1, precision=4)))
        assert np.allclose(our0, ave0)

        if len(params['shape']) > 1:
            S1 = run.summarize(axis=1, keepdims=keepdims)
            our1 = S1['mean']
            ave1 = np.average(stack_d, weights=stack_w, axis=(1, -1), **ave_kw)
            if keepdims:
                ave1 = ave1[..., 0]
            print('our1 = {}'.format(ub.urepr(our1, nl=1, precision=4)))
            print('ave1 = {}'.format(ub.urepr(ave1, nl=1, precision=4)))
            assert np.allclose(our1, ave1)


def test_with_nan_case():
    import kwarray
    run = kwarray.RunningStats(nan_policy='omit')
    n = float('nan')
    im1 = np.array([[[0, 0, 1], [1, 0, 1]], [[1, 2, 3], [4, 5, 6]]])
    im2 = np.array([[[2, 3, 2], [1, 3, 2]], [[0, 8, 8], [8, 4, 3]]])
    im3 = np.array([[[0, 0, 1], [1, 0, 1]], [[n, n, n], [n, n, n]]])
    im4 = np.array([[[n, n, n], [n, n, n]], [[n, n, n], [n, n, n]]])

    # summary = run.summarize(axis=ub.NoParam, keepdims=True)
    # print('summary = {}'.format(ub.urepr(summary, nl=1)))

    run.update(im1)
    summary = run.summarize(axis=ub.NoParam, keepdims=True)
    print('summary = {}'.format(ub.urepr(summary, nl=1)))

    run.update(im2)
    summary = run.summarize(axis=ub.NoParam, keepdims=True)
    print('summary = {}'.format(ub.urepr(summary, nl=1)))

    run.update(im3)
    summary = run.summarize(axis=ub.NoParam, keepdims=True)
    print('summary = {}'.format(ub.urepr(summary, nl=1)))

    run.update(im4)
    summary = run.summarize(axis=ub.NoParam, keepdims=True)
    print('summary = {}'.format(ub.urepr(summary, nl=1)))


def combine_mean_std_zero_nums():
    """
    Test for bug that occurs when nums are zero and bessel correction causes
    the numerator to be negative.
    """
    from kwarray.util_averages import _combine_mean_stds
    means = np.array([[0.154509, 0.192753, 0.182485],
                      [0.      , 0.      , 0.      ],
                      [0.122889, 0.170761, 0.15622 ],
                      [0.202546, 0.24321 , 0.282333]], dtype=np.float64)
    stds = np.array([[1.26812e-01, 1.33810e-01, 1.30416e-01],
                     [1.00000e+08, 1.00000e+08, 1.00000e+08],
                     [1.31318e-01, 1.38126e-01, 1.28543e-01],
                     [9.27550e-02, 8.76810e-02, 1.21056e-01]], dtype=np.float64)
    nums = np.array([[200704., 200704., 200704.],
                     [     0.,      0.,      0.],
                     [200704., 200704., 200704.],
                     [150528., 150528., 150528.]], dtype=np.float64)

    cm1, cs1, _ = _combine_mean_stds(means, stds, nums, axis=0)
    assert not np.any(np.isnan(cs1))

    means = np.array([[0.1, 0.1, 0.1],
                      [  0, 0.,   0.],
                      [0.1, 0.1, 0.1],
                      [0.2, 0.2, 0.2]], dtype=np.float64)
    stds = np.array([[1.2, 1.3, 1.3],
                     [1e8, 3.0, 1e1],
                     [1.3, 1.3, 1.2],
                     [9.2, 8.7, 1.2]], dtype=np.float64)
    nums = np.array([[20., 20., 20.],
                     [ 0.,  0., 0.],
                     [20., 20., 20.],
                     [15., 15., 15.]], dtype=np.float64)

    cm1, cs1, _ = _combine_mean_stds(means, stds, nums, axis=0, bessel=False)
    assert not np.any(np.isnan(cs1))
