"""
References:
    https://en.wikipedia.org/wiki/Hodges%E2%80%93Lehmann_estimator
    https://www.youtube.com/watch?v=PaRZge3njm4
    https://github.com/borisvish/Median-Polish
    https://jerryzli.github.io/robust-ml-fall19/lec3.pdf
    https://www.mwsug.org/proceedings/2006/stats/MWSUG-2006-SD01.pdf
    *** > Very good -> https://cseweb.ucsd.edu/~slovett/workshops/robust-statistics-2019/slides/donoho-univariate.pdf
"""


def hodgest_lehmann_estimator():
    from kwarray.distributions import Mixture
    import numpy as np
    data = Mixture.random(6).sample(3000)

    pairwise_mean = (data[:, None] + data[None, :]) / 2.0
    hl_mean = np.median(pairwise_mean)
    return hl_mean


def other():
    # Other central location estimators
    median = np.median(data)
    arithmetic_mean = np.mean(data)

    import sklearn
    sklearn.linear_model.RANSACRegressor

    import numpy as np
    from sklearn.covariance import EmpiricalCovariance
    cov = EmpiricalCovariance().fit(data[:, None])


    import numpy as np
    from sklearn.covariance import GraphicalLassoCV
    lass = GraphicalLassoCV().fit(data[:, None])

    import numpy as np
    import statsmodels.formula.api as smf
    import statsmodels as sm

    modelspec = ('cost ~ np.log(units) + np.log(units):item + item') #where item is a categorical variable
    results = smf.rlm(modelspec, data = dataset, M = sm.robust.norms.TukeyBiweight()).fit()


def m_estimator():
    from kwarray.distributions import Mixture
    import numpy as np
    import scipy
    data = Mixture.random(6).sample(3000)

    def tukey_bisquare(x):
        c = 4.685
        absx = np.abs(x)
        x2 = (x * x)
        x4 = (x2 * x2)
        x6 = (x4 * x2)
        c2 = c * c
        c4 = c2 * c2
        r0 = x2 / 2 - (x4 / (2 * c2)) + x6 / (6 * c4)
        return np.where(absx <= c, r0, c2 / 6)

    robustbase.Qn(data)

    from scipy.stats import rv_continuous, norm
    from scipy import stats
    stats.exponnorm.fit(data)
    stats.truncnorm.fit(data)
    stats.genlogistic.fit(data)
    norm.fit(data)

    #https://github.com/deepak7376/robustbase
    import robustbase

    def robust_center(data, robust_function='tukey'):
        # Maximum liklihood estimator
        if robust_function == 'tukey':
            rho = tukey_bisquare
            phi = None # TODO
        elif robust_function == 'mean':
            rho = lambda x: (x * x) / 2  # NOQA
            phi = lambda x: x
        elif robust_function == 'median':
            rho = lambda x: np.abs(x)  # NOQA
            phi = lambda x: np.sign(x)
        else:
            raise KeyError
        def mle_objective(center):
            return np.sum(np.log(rho(data - center)))
        x0 = np.median(data)
        # result = scipy.optimize.newton(mle_objective, x0, fprime=phi)
        result = scipy.optimize.minimize(mle_objective, x0, jac=phi)
        # method='BFGS')
        return result.x


    def irls():
        sig = data.std()
        mean = data.mean()
        med = np.median(data)
        print('med = {!r}'.format(med))
        print('sig = {!r}'.format(sig))
        print('mean = {!r}'.format(mean))
        # Start with a guess
        x0 = med
        x0 = mean
        uk = x0
        rho = lambda x: (x * x) / 2  # NOQA
        # rho = lambda x: np.abs(x)  # NOQA
        for k in range(10):
            # Compute "weighted" average
            residual = (data - uk) / sig
            w = rho(residual)
            # Get next center estimate
            # u_next = (data * w).sum() / w.sum()
            u_next = np.average(data, weights=w)
            print('u_next = {!r}'.format(u_next))
            # Why does this oscilate?
            if abs(u_next - uk) < 0.0001 * sig:
                break
            uk = u_next
        print('uk = {!r}'.format(uk))

    center = robust_center(data, 'tukey')
    print('center = {!r}'.format(center))

    print(robust_center(data, 'mean'))
    print(robust_center(data, 'median'))
    print(robust_center(data, 'tukey'))

    # Scale estimator

    # Important to compute the scale estimator first
    # This is because location m-estimators are not scale equivariant
    # but we can adjust them to be by computing scale first.

    # Location estimator


    pass


def tukey_biweight_loss(r, c=4.685):
    """
    Beaton Tukey Biweight

    Computes the function :
        L(r) = (
            (c ** 2) / 6 * (1 - 1 * (r / c) ** 2) ** 3) if abs(r) <= c else
            (c ** 2)
        )

    Args:
        r (float | ndarray): residual parameter
        c (float): tuning constant (defaults to 4.685 which is 95% efficient
            for normal distributions of residuals)

    TODO:
        - [ ] Move elsewhere or find a package that provides it
        - [ ] Move elsewhere (kwarray?) or find a package that provides it

    Returns:
        float | ndarray

    References:
        https://en.wikipedia.org/wiki/Robust_statistics
        https://mathworld.wolfram.com/TukeysBiweight.html
        https://statisticaloddsandends.wordpress.com/2021/04/23/what-is-the-tukey-loss-function/
        https://arxiv.org/pdf/1505.06606.pdf

    Example:
        >>> from watch.utils.util_kwarray import *  # NOQA
        >>> import ubelt as ub
        >>> r = np.linspace(-20, 20, 1000)
        >>> data = {'r': r}
        >>> grid = ub.named_product({
        >>>     'c': [4.685, 2, 6],
        >>> })
        >>> for kwargs in grid:
        >>>     key = ub.repr2(kwargs, compact=1)
        >>>     loss = tukey_biweight_loss(r, **kwargs)
        >>>     data[key] = loss
        >>> import pandas as pd
        >>> melted = pd.DataFrame(data).melt(['r'])
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> sns = kwplot.autosns()
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> ax = sns.lineplot(data=melted, x='r', y='value', hue='variable', style='variable')
        >>> #ax.set_ylim(*robust_limits(melted.value))
    """
    # https://statisticaloddsandends.wordpress.com/2021/04/23/what-is-the-tukey-loss-function/
    is_inside = np.abs(r) < c
    c26 = (c ** 2) / 6
    loss = np.full_like(r, fill_value=c26, dtype=np.float32)
    r_inside = r[is_inside]
    loss_inside = c26 * (1 - (1 - (r_inside / c) ** 2) ** 3)
    loss[is_inside] = loss_inside
    return loss


def outline_of_an_algorithm():
    """

    * First procedure: Find "important" points in the data. Thes are typically
      going to be centers of Gaussian distributions. We assume this is
      univariate, so we can order these once we are done. The min and max are
      also included as control points.

      Do piecewise stretching of the histograms between control points

    Take a histogram of the data

    * Find the quartiles of the data.

        * Perform a normality test on the inner quartile of the data.
           * While the data is not normal, subdivide
           * In this way attempt to "cluster" to find the "central point" of
             any mixture of Gaussians present in the data.
           * maybe some other cluster method for this is fine


    *

    """
