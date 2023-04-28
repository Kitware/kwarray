
def benchmark_atleast_nd():
    # Hmm, mine is actually faster
    # %timeit atleast_nd(arr, 3)
    # %timeit np.atleast_3d(arr)
    import ubelt as ub
    import kwarray
    import numpy as np

    data_basis = {}
    data_basis['shape'] = []

    for n in [0, 1, 8, 16, 32]:
        data_basis['shape'] += [
                tuple(),
                (n,),
                (n, n),
                (n, n, n),
                (n, n, n, n),
        ]

    data_grid = list(ub.named_product(data_basis))

    method_lut = {
        'atleast_1d': (np.atleast_1d, {}),
        'atleast_2d': (np.atleast_2d, {}),
        'atleast_3d': (np.atleast_3d, {}),

        'atleast_nd-1b': (kwarray.atleast_nd, {'n': 1, 'front': False}),
        'atleast_nd-2b': (kwarray.atleast_nd, {'n': 2, 'front': False}),
        'atleast_nd-3b': (kwarray.atleast_nd, {'n': 3, 'front': False}),

        # 'atleast_nd-1f': (kwarray.atleast_nd, {'n': 1, 'front': True}),
        # 'atleast_nd-2f': (kwarray.atleast_nd, {'n': 2, 'front': True}),
        # 'atleast_nd-3f': (kwarray.atleast_nd, {'n': 3, 'front': True}),
    }

    method_grid = list(ub.named_product({
        'method': list(method_lut.keys()),
    }))

    import timerit
    ti = timerit.Timerit(10000, bestof=10, verbose=2)

    rows = []
    for datakw in data_grid:
        arr = np.empty(datakw['shape'])
        datakey = ub.urepr(datakw, compact=1)

        # Extra summary info
        datarow = datakw.copy()
        datarow['dims'] = len(datakw['shape'])
        datarow['datakey'] = datakey
        datarow['numel'] = arr.size

        for methodkw in method_grid:
            methodkey = ub.urepr(methodkw, compact=1)
            key = methodkey + ':' + datakey

            # Extra summary info
            methodrow = methodkw.copy()
            if 'nd' in methodrow['method']:
                methodrow['package'] = 'kwarray'
            else:
                methodrow['package'] = 'numpy'
            # methodrow['methodkey'] = methodkey

            func, funckw = method_lut[methodkw['method']]
            for timer in ti.reset(key):
                func(arr, **funckw)
            row = {
                **datarow,
                **methodrow,
                'key': key,
                'min': ti.min(),
                'mean': ti.mean(),
            }
            rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows).sort_values('min')
    print(df)

    kwarray_ave_mean = df[df.package == 'kwarray']['mean'].mean()
    numpy_ave_mean = df[df.package == 'numpy']['mean'].mean()
    kwarray_ave_min = df[df.package == 'kwarray']['min'].mean()
    numpy_ave_min = df[df.package == 'numpy']['min'].mean()
    print('kwarray_ave_mean = {!r}'.format(kwarray_ave_mean))
    print('kwarray_ave_min  = {!r}'.format(kwarray_ave_min))
    print('numpy_ave_mean   = {!r}'.format(numpy_ave_mean))
    print('numpy_ave_min    = {!r}'.format(numpy_ave_min))

    import kwplot
    sns = kwplot.autosns()
    # sns.lineplot(data=df, x='dims', y='min', hue='method')

    longform = df.melt(['dims', 'method', 'numel', 'package'], value_vars=['min', 'mean'])
    kwplot.figure(fnum=1, doclf=True)
    ax = sns.lineplot(data=longform, x='dims', y='value', hue='package', size='variable', style='method')
    ax.set_yscale('log')

    kwplot.figure(fnum=2)
    sns.lineplot(data=longform, x='numel', y='value', hue='package', size='variable', style='method')
