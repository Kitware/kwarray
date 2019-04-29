def _benchmark_distinguish_tensor_ndarray():
    import timerit
    ti = timerit.Timerit(10000, bestof=1000, verbose=1)

    array = np.arange(100)
    tensor = torch.arange(100)

    totals = ub.ddict(lambda: 0)

    for data in [array, tensor]:
        for timer in ti.reset(label='is_tensor(data)'):
            with timer:
                torch.is_tensor(data)
        totals[ti.label] += ti.mean()

        for timer in ti.reset(label='isinstance(data, np.ndarray)'):
            with timer:
                isinstance(data, np.ndarray)
        totals[ti.label] += ti.mean()

        for timer in ti.reset(label='isinstance(data, torch.Tensor)'):
            with timer:
                isinstance(data, torch.Tensor)
        totals[ti.label] += ti.mean()

