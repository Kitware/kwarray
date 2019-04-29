
def bench_typecheck():
    import numpy as np
    datas = {
        'u': np.random.rand(10).astype(np.uint32),
        'i': np.random.rand(10).astype(np.int32),
        'f': np.random.rand(10).astype(np.float32),
    }

    import ubelt as ub
    ti = ub.Timerit(50000, bestof=200, label='time')

    for timer in ti.reset('in list'):
        with timer:
            for data in datas.values():
                data.dtype.kind in ['i', 'u']

    for timer in ti.reset('in set'):
        with timer:
            for data in datas.values():
                data.dtype.kind in {'i', 'u'}

    for timer in ti.reset('in tuple'):
        with timer:
            for data in datas.values():
                data.dtype.kind in ('i', 'u')

    for timer in ti.reset('two =='):
        with timer:
            for data in datas.values():
                data.dtype.kind == 'i' or data.dtype.kind == 'u'

    # They are all very close, and it probably doesnt matter, but it does seem
    # like set is slightly better on average.
