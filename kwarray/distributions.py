"""
Defines data structures for efficient repeated sampling of specific
distributions (e.g. Normal, Uniform, Binomial) with specific parameters.

Inspired by ~/code/imgaug/imgaug/parameters.py

Similar Libraries:
    * https://docs.pymc.io/api/distributions.html
    * https://github.com/phobson/paramnormal

TODO:
    - [ ] change sample shape to just a single num.
    - [ ] Some Distributions will output vectors. Maybe we could just postpend the dimensions?
    - [ ] Expose as kwstats?
    - [ ] Improve coerce syntax for concise distribution specification

References:
    https://stackoverflow.com/questions/21100716/fast-arbitrary-distribution-random-sampling
    https://stackoverflow.com/questions/4265988/generate-random-numbers-with-a-given-numerical-distribution
    https://codereview.stackexchange.com/questions/196286/inverse-transform-sampling
"""
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import ubelt as ub
import functools
import builtins
import numbers
import fractions  # NOQA
from kwarray.util_random import ensure_rng

inf = float('inf')
# __all__ =  [


class Value(ub.NiceRepr):
    """
    Container for class `__params__` values.

    Used to store metadata about distribution arguments, including default
    values, numeric constraints, typing, and help text.

    Example:
        >>> from kwarray.distributions import *  # NOQA
        >>> self = Value(43.5)
        >>> print(Value(name='lucy'))
        >>> print(Value(name='jeff', default=1))
        >>> self = Value(name='fred', default=1.0)
        >>> print('self = {}'.format(ub.repr2(self, nl=1)))
        >>> print(Value(name='bob', default=1.0, min=-5, max=5))
        >>> print(Value(name='alice', default=1.0, min=-5))
    """
    def __init__(self, default=None, min=None, max=None, help=None,
                 constraints=None, type=None, name=None):
        if type is None and default is not None:
            type = builtins.type(default)
        if help is None:
            help = 'no help given'

        self.default = default
        self.name = name
        self.min = min
        self.max = max
        self.help = help
        self.type = type
        self.constraints = None

        # if self.min is None:
        #     self.min = -1024
        # if self.max is None:
        #     self.max = 1024

        if self.type is None:
            self.category = None
            self.numeric = None
        else:
            self.numeric = issubclass(self.type, numbers.Number)
            if self.numeric:
                self.category = (
                    (issubclass(self.type, numbers.Integral) and 'integral') or
                    (issubclass(self.type, numbers.Rational) and 'rational') or
                    (issubclass(self.type, float) and 'floating') or
                    (issubclass(self.type, numbers.Rational) and 'rational') or
                    # (issubclass(self.type, numbers.Real) and 'real') or # is it though?
                    ('numeric')
                )
                # TODO: add sequence, etc.
                #  (issubclass(self.type, numbers.Complex) and 'complex') # maybe someday
            else:
                self.category = 'unknown'

    def __nice__(self):
        parts = []
        parts.append('{name}={default}')
        # if self.numeric:
        #     parts.append('{sym_elementof}')
        #     # symbol = MathSymbols.__dict__.get('sym_' + self.category, '?')
        #     symbol_parts = []
        #     symbol_parts.append(symbol)
        #     if self.min == 0 or self.max == 0:
        #         symbol_parts.append('0')
        #     minmax_constraint_parts = []
        #     if self.min is not None:
        #         if self.min > 0:
        #             symbol_parts.append('+')
        #         if self.min != 0:
        #             minmax_constraint_parts.append('{min}<=')
        #     if self.max is not None:
        #         if self.max < 0:
        #             symbol_parts.append('-')
        #         if self.max != 0:
        #             minmax_constraint_parts.append('<={max}')
        #     parts.append(''.join(symbol_parts))
        #     parts.append(''.join(minmax_constraint_parts))
        kw = {}
        # kw.update(MathSymbols.__dict__)
        kw.update(self.__dict__)
        template = ' '.join(parts)
        text = template.format(**kw)
        return text

    def sample(self, rng):
        """
        Get a random value for this parameter.
        """
        # TODO: special min and max values for classmethod random values
        _min = -1024 if self.min is None else self.min
        _max =  1024 if self.max is None else self.max
        if self.category == 'integral':
            assert _min <= _max
            sample = rng.randint(_min, _max)
        elif self.category == 'floating':
            assert _min <= _max
            sample = rng.rand() * (_max - _min) + _min
        else:
            raise NotImplementedError
        return sample


def _issubclass2(child, parent):
    """
    Uses string comparisons to avoid ipython reload errors.
    Much less robust though.
    """
    # String comparison
    if child.__name__ == parent.__name__:
        if child.__module__ == parent.__module__:
            return True
    # Recurse through classes of obj
    return any(_issubclass2(base, parent) for base in child.__bases__)


def _isinstance2(obj, cls):
    """
    Internal hacked version is isinstance for debugging

    obj = self
    cls = distributions.Distribution

    child = obj.__class__
    parent = cls
    """
    return isinstance(obj, cls)
    # try:
    #     return _issubclass2(obj.__class__, cls)
    # except Exception:
    #     return False


class Parameterized(ub.NiceRepr):
    """
    Keeps track of all registered params and classes with registered params
    """
    def __init__(self):
        self._called_init = True
        self._params = ub.odict()
        self._children = ub.odict()

    def __setattr__(self, key, value):
        if not getattr(self, '_called_init', key == '_called_init'):
            raise Exception(
                'Need to call super().__init__() before setting '
                'attribute "{}" of Parameterized classes'.format(key))

        if not key.startswith('_'):
            if key in self._children or _isinstance2(value, Parameterized):
                self._children[key] = value
        super(Parameterized, self).__setattr__(key, value)

    def _setparam(self, key, value):
        setattr(self, key, value)
        self._params[key] = value

    def _setchild(self, key, value):
        assert isinstance(key, str)
        self._children[key] = value

    def children(self):
        for key, value in self._children.items():
            yield key, value

    def seed(self, rng=None):
        rng = ensure_rng(rng)
        for key, child in self.children():
            if _isinstance2(child, Parameterized):
                child.seed(rng)

    def parameters(self):
        """
        Returns parameters in this object and its children
        """
        for key, value in self._params.items():
            yield key, value
        for prefix, child in self.children():
            for subkey, value in child.parameters():
                key = prefix + '.' + subkey
                yield key,  value

    def _body_str():
        pass

    def idstr(self, nl=None, thresh=80):
        """
        Example:
            >>> self = TruncNormal()
            >>> self.idstr()
            >>> #
            >>> #
            >>> class Dummy(Distribution):
            >>>     def __init__(self):
            >>>         super(Dummy, self).__init__()
            >>>         self._setparam('a', 3)
            >>>         self.b = Normal()
            >>>         self.c = Uniform()
            >>> self = Dummy()
            >>> print(self.idstr())
            >>> #
            >>> class Tail5(Distribution):
            >>>     def __init__(self):
            >>>         super(Tail5, self).__init__()
            >>>         self._setparam('a_parameter', 3)
            >>>         for i in range(5):
            >>>             self._setparam(chr(i + 97), i)
            >>> #
            >>> class Tail6(Distribution):
            >>>     def __init__(self):
            >>>         super(Tail6, self).__init__()
            >>>         for i in range(9):
            >>>             self._setparam(chr(i + 97) + '_parameter', i)
            >>> #
            >>> class Dummy2(Distribution):
            >>>     def __init__(self):
            >>>         super(Dummy2, self).__init__()
            >>>         self._setparam('x', 3)
            >>>         self._setparam('y', 3)
            >>>         self.d = Dummy()
            >>>         self.f = Tail6()
            >>>         self.y = Tail5()
            >>> self = Dummy2()
            >>> print(self.idstr())
            >>> print(ub.repr2(self.json_id()))
        """
        classname = self.__class__.__name__
        self_part = ['{}={}'.format(key, ub.repr2(value, precision=2, si=True, nl=0))
                     for key, value in self._params.items()]
        child_part = ['{}={}'.format(key, child.idstr(nl, thresh=thresh - 2))
                      for key, child in self.children()]

        body, nl = self._make_body(self_part, child_part, nl, thresh - len(classname) - 2)
        if nl:
            body = ub.indent(body, '  ')
        return '{}({})'.format(classname, body.rstrip(' '))

    def json_id(self):
        children = ub.odict([(key, child.json_id())
                             for key, child in self.children()])
        params = ub.odict([
            (key, value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in self._params.items()])
        return ub.dict_union(ub.odict([('__class__', self.__class__.__name__)]),
                             params,
                             children)

    def _make_body(self, self_part, child_part, nl=None, thresh=80):
        parts = self_part + child_part
        if len(parts) == 0:
            body = ''
        else:
            sep = ', '
            oneline_body = sep.join(parts)
            if nl is None:
                nl = len(oneline_body) > thresh or len(child_part) > 0
            if nl:
                sep = ',\n'
                body = '\n' + sep.join(parts) + '\n'
            else:
                body = oneline_body
        return body, nl

    def __nice__(self):
        try:
            params = ub.odict(self.parameters())
            return ub.repr2(params, nl=0, precision=2, si=True, nobr=True,
                            explicit=True)
        except Exception:
            return '?'


class ParameterizedList(Parameterized):
    """

    Example:
        >>> from kwarray import distributions as stoch
        >>> self1 = stoch.ParameterizedList([
        >>>     stoch.Normal(),
        >>>     stoch.Uniform(),
        >>> ])
        >>> print(self1.idstr())
        >>> self = stoch.ParameterizedList([stoch.ParameterizedList([
        >>>     stoch.Normal(),
        >>>     stoch.Uniform(),
        >>>     self1,
        >>> ])])
        >>> print(self.idstr())
        >>> print(self.idstr(0))

    """
    def __init__(self, items):
        super(ParameterizedList, self).__init__()
        self._items = []
        for t in items:
            self.append(t)

    def _setparam(self, key, value):
        raise NotImplementedError('lists cannot have params')

    def append(self, item):
        key = str(len(self._items))
        self._items.append(item)
        self._setchild(key, item)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return self._items[key]
        else:
            return self._children[key]

    def idstr(self, nl=None, thresh=80):
        assert len(self._items) == len(self._children)
        assert len(self._params) == 0
        child_part = ['{}'.format(child.idstr(nl, thresh=thresh - 2))
                      for key, child in self.children()]
        body, nl = self._make_body([], child_part, nl, thresh - 2)
        if nl:
            body = ub.indent(body, '  ')
        return '[{}]'.format(body.rstrip(' '))


class _BinOpMixin(object):
    """
    Allows binary operations to be performed on distributions to create
    composed distributions.
    """
    def __add__(self, other):
        return Composed(np.add, [self, other])

    def __pow__(self, other):
        return Composed(np.power, [self, other])

    def __mul__(self, other):
        return Composed(np.multiply, [self, other])

    def __sub__(self, other):
        return Composed(np.subtract, [self, other])

    def __truediv__(self, other):
        return Composed(np.divide, [self, other])

    def __matmul__(self, other):
        raise NotImplementedError

    def __floordiv__(self, other):
        raise NotImplementedError

    def __mod__(self, other):
        raise NotImplementedError

    def __divmod__(self, other):
        raise NotImplementedError

    def __lshift__(self, other):
        raise NotImplementedError

    def __rshift__(self, other):
        raise NotImplementedError

    def __and__(self, other):
        raise NotImplementedError

    def __xor__(self, other):
        raise NotImplementedError

    def __or__(self, other):
        raise NotImplementedError

    def __neg__(self):
        return Composed(np.negative, [self])

    def __pos__(self):
        return Composed(np.positive, [self])

    def __invert__(self):
        raise NotImplementedError

    def __abs__(self):
        return Composed(np.abs, [self])

    def __round__(self, ndigits=None):
        return Composed(np.round, [self, ndigits])

    def __trunc__(self):
        return Composed(np.trunc, [self])

    def __floor__(self):
        return Composed(np.floor, [self])

    def __ceil__(self):
        return Composed(np.ceil, [self])

    # Numpy mixins

    # def __getattr__(self, attr):
    #     # Handle application of general numpy functions
    #     operation = getattr(np, attr)
    #     return Composed(operation, [self])

    def int(self):
        def _cast_int(x):
            if isinstance(x, np.ndarray):
                return x.astype(np.int)
            else:
                return int(x)
        return Composed(_cast_int, [self])

    def round(self, ndigits=None):
        return Composed(np.round, [self, ndigits])

    def clip(self, a_min=None, a_max=None):
        return Composed(np.clip, [self, a_min, a_max])

    def log(self):
        return Composed(np.log, [self])

    def log10(self):
        return Composed(np.log10, [self])

    def exp(self):
        return Composed(np.exp, [self])

    def sqrt(self):
        return Composed(np.sqrt, [self])

    def abs(self):
        return Composed(np.abs, [self])


class _RBinOpMixin(_BinOpMixin):
    """
    https://docs.python.org/3/reference/datamodel.html
    """
    def __radd__(self, other):
        return Composed(np.add, [other, self])

    def __rpow__(self, other):
        return Composed(np.power, [other, self])

    def __rmul__(self, other):
        return Composed(np.multiply, [other, self])

    def __rsub__(self, other):
        return Composed(np.subtract, [other, self])

    def __rtruediv__(self, other):
        return Composed(np.divide, [other, self])

    def __rmatmul__(self, other):
        raise NotImplementedError

    def __rfloordiv__(self, other):
        raise NotImplementedError

    def __rmod__(self, other):
        raise NotImplementedError

    def __rdivmod__(self, other):
        raise NotImplementedError

    def __rlshift__(self, other):
        raise NotImplementedError

    def __rrshift__(self, other):
        raise NotImplementedError

    def __rand__(self, other):
        raise NotImplementedError

    def __rxor__(self, other):
        raise NotImplementedError

    def __ror__(self, other):
        raise NotImplementedError


class Distribution(Parameterized, _RBinOpMixin):
    """
    Base class for all distributions.

    There are 3 main subtypes:
        ContinuousDistribution
        DiscreteDistribution
        MixedDistribution

    Note:
        In [DiscVsCont]_ notes that there are only 3 types of random variables:
        discrete, continuous, or mixed. And these types are mutually exclusive.

    Note:
        When inheriting from this class, you typically do not  need to define
        an `__init__` method.  Instead, overwrite the `__params__` class
        attribute with an `OrderedDict[str, Value]` to indicate what the
        signature of the `__init__` method should be. This allows for (1)
        concise expression of new distributions and (2) for new distributions
        to inherit a `random` classmethod that works according to constraints
        specified in each parameter Value.

        If you do overwrite `__init__`, be sure to call `super()`.

    References:
        .. [DiscVsCont] https://www.quora.com/Can-a-random-variable-be-both-discrete-and-continuous
    """
    __params__ = NotImplemented  # overwrite!

    def __init__(self, *args, **kwargs):
        super(Distribution, self).__init__()
        rng = kwargs.pop('rng', None)
        self.rng = ensure_rng(rng)
        constraints = []
        __params__ = self.__params__
        if __params__ is NotImplemented:
            print('Warning, no params: self={!r}'.format(self))
        elif isinstance(__params__, dict):
            for paramx, (key, info) in enumerate(__params__.items()):
                info.name = key
                if paramx < len(args):
                    val = args[paramx]
                    if key in kwargs:
                        raise ValueError('arg pased as both positional and keyword')
                else:
                    val = kwargs.get(key, info.default)
                self._setparam(key, val)
                if info.min is not None:
                    constraints.append(lambda: val >= info.min)
                if info.max is not None:
                    constraints.append(lambda: val <= info.max)
                if info.constraints:
                    constraints.extend(info.constraints)
        else:
            raise TypeError(__params__)

    def __call__(self, *shape):
        return self.sample(*shape)

    def seed(self, rng=None):
        super(Distribution, self).seed(rng)
        self.rng = ensure_rng(rng)

    def sample(self, *shape):
        raise NotImplementedError('overwrite me')

    @classmethod
    def random(cls, rng=None):
        """
        Returns a random distribution

        Args:
            rng (int | float | None | numpy.random.RandomState | random.Random):
                random coercable

        CommandLine:
            xdoctest -m /home/joncrall/code/kwarray/kwarray/distributions.py Distribution.random --show

        Example:
            >>> from kwarray.distributions import *  # NOQA
            >>> self = Distribution.random()
            >>> print('self = {!r}'.format(self))
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> self.plot('0.001s', bins=256)
            >>> kwplot.show_if_requested()
        """
        rng = ensure_rng(rng)
        if cls is Distribution:
            _PRIMATIVES = [
                # Bernoulli,
                # Binomial,
                # Categorical,
                # Constant,
                # DiscreteUniform,
                Normal,
                Exponential,
                TruncNormal,
                Uniform,
            ]
            chosen = rng.choice(_PRIMATIVES)
            self = chosen.random(rng=rng)
        else:
            try:
                # TODO:
                # - [ ] Suport for notations like a "min" param must be smaller
                # than a "max" param.
                kw = {}
                for k, v in cls.__params__.items():
                    type = v.type
                    if type is not None:
                        sample = v.sample(rng=rng)
                    kw[k] = sample
                self = cls(rng=rng, **kw)
            except Exception:
                print('Error constructing random cls = {!r}'.format(cls))
                raise
        return self

    @classmethod
    def coerce(cls, arg, rng=None):
        if isinstance(arg, cls):
            self = arg
        elif isinstance(arg, Distribution):
            # allow distribution substitution
            self = arg
        elif isinstance(arg, (list, tuple)):
            self = cls(*arg, rng=rng)
        elif isinstance(arg, dict):
            self = cls(rng=rng, **arg)
        else:
            raise CoerceError('cannot coerce {} as {}'.format(arg, cls))
        return self

    @classmethod
    def cast(cls, arg):
        import warnings
        warnings.warn('Distributions.cast is deprecated. Use coerce',
                      DeprecationWarning)
        return cls.coerce(arg)

    @classmethod
    def seeded(cls, rng=0):
        return Seeded(rng, cls)

    def plot(self, n='0.01s', bins='auto', stat='count', color=None, kde=True,
             ax=None, **kwargs):
        """
        Plots ``n`` samples from the distribution.

        Args:
            bins (int | List[Number] | str):
                number of bins, bin edges, or special numpy method for finding
                the number of bins.

            stat (str):
                density, count, probability, frequency

            **kwargs: other args passed to :func:`seaborn.histplot`

        Example:
            >>> from kwarray.distributions import Normal  # NOQA
            >>> self = Normal()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> self.plot(n=1000)
        """
        import seaborn as sns
        if isinstance(n, str):
            data = _generate_on_a_time_budget(func=lambda: self.sample(100), maxiters=1000, budget=n)
            print(len(data))
        else:
            data = self.sample(n)
        return sns.histplot(data, ax=ax, bins=bins, color=color, stat=stat,
                            kde=kde, **kwargs)

    def _show(self, n, bins=None, ax=None, color=None, label=None):
        """ plot samples monte-carlo style """
        import warnings
        warnings.warn('plot is deprecated, use plot instead')
        if ax is None:
            from matplotlib import pyplot as plt
            ax = plt.gca()
        data = self.sample(n)
        ax.hist(data, bins=bins, color=color, label=label)


def _coerce_timedelta(data):
    import datetime
    if isinstance(data, datetime.timedelta):
        delta = data
        num_seconds = delta.total_seconds()
    elif isinstance(data, numbers.Number):
        num_seconds = data
    elif isinstance(data, str):
        if data.endswith('s'):
            try:
                num_seconds = float(data[:-1])
            except Exception:
                num_seconds = None
        if num_seconds is None:
            import pytimeparse
            num_seconds = pytimeparse.parse(data)
            if num_seconds is None:
                raise ValueError('{} is not a parsable delta'.format(data))
    else:
        raise TypeError(data)
    return num_seconds


def _generate_on_a_time_budget(func, maxiters, budget):
    """
    budget = 60
    """
    seconds = _coerce_timedelta(budget)
    timer = ub.Timer().tic()
    accum = []
    accum.append(func())
    if timer.toc() < seconds / 2.0:
        for i in range(1, maxiters):
            if timer.toc() >= seconds:
                break
            accum.append(func())
    data = np.hstack(accum)
    return data


class DiscreteDistribution(Distribution):
    # @classmethod
    # def coerce(cls, arg)
    pass


class ContinuousDistribution(Distribution):
    pass


class MixedDistribution(Distribution):
    pass


class Mixture(MixedDistribution):
    """
    Creates a mixture model of multiple distributions

    Contains a set of distributions with associated weights. Sampling is done
    by first choosing a distribution with probability proportional to its
    weighthing, and then sampling from the chosen distribution.

    In general, a mixture model generates data by first first we sample from z,
    and then we sample the observables x from a distribution which depends on
    z.  , i.e. p(z, x) = p(z) p(x | z) [GrosseMixture]_ [StephensMixture]_.

    Args:
        pdfs (List[Distribution]):
            list of distributions

        weights (List[float]): corresponding weights of each distribution

        rng (np.random.RandomState): seed random number generator

    References:
        .. [StephensMixture] https://stephens999.github.io/fiveMinuteStats/intro_to_mixture_models.html
        .. [GrosseMixture] https://www.cs.toronto.edu/~rgrosse/csc321/mixture_models.pdf

    CommandLine:
        xdoctest -m kwarray.distributions Mixture:0 --show

    Example:
        >>> # In this examle we create a bimodal mixture of normals
        >>> from kwarray.distributions import *  # NOQA
        >>> pdfs = [Normal(mean=10, std=2), Normal(18, 2)]
        >>> self = Mixture(pdfs)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> self.plot(500, bins=25)
        >>> kwplot.show_if_requested()

    Example:
        >>> # Compare Composed versus Mixture Distributions
        >>> # Given two normal distributions,
        >>> from kwarray.distributions import Normal  # NOQA
        >>> from kwarray.distributions import *  # NOQA
        >>> n1 = Normal(mean=11, std=3)
        >>> n2 = Normal(mean=53, std=5)
        >>> composed = (n1 * 0.3) + (n2 * 0.7)
        >>> mixture = Mixture([n1, n2], [0.3, 0.7])
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.figure(fnum=1, pnum=(2, 2, 1))
        >>> ax = kwplot.figure(pnum=(2, 1, 1), title='n1 & n2').gca()
        >>> n = 10000
        >>> plotkw = dict(stat='density', kde=1, bins=1000)
        >>> plotkw = dict(stat='count', kde=1, bins=1000)
        >>> #plotkw = dict(stat='frequency', kde=1, bins='auto')
        >>> n1.plot(n, ax=ax, **plotkw)
        >>> n2.plot(n, ax=ax, **plotkw)
        >>> ax=kwplot.figure(pnum=(2, 2, 3), title='composed').gca()
        >>> composed.plot(n, ax=ax, **plotkw)
        >>> ax=kwplot.figure(pnum=(2, 2, 4), title='mixture').gca()
        >>> mixture.plot(n, ax=ax, **plotkw)
        >>> kwplot.show_if_requested()
    """
    def __init__(self, pdfs, weights=None, rng=None):
        super(Mixture, self).__init__(rng=rng)
        self.pdfs = pdfs
        self._setparam('pdfs', pdfs)
        self._setparam('weights', weights)
        idxs = np.arange(len(pdfs))
        self._idx_choice = Categorical(idxs, weights, rng=rng)

    def sample(self, *shape):
        """
        Sampling from a mixture of k distributions with weights w_k is
        equivalent to picking a distribution with probability w_k, and then
        sampling from the picked distribution.
        `SOuser6655984 <https://stackoverflow.com/a/47762586/887074>`
        """
        # Choose which distributions are picked for each sample
        idxs = self._idx_choice.sample(*shape)
        idx_to_nsamples = ub.dict_hist(idxs.ravel())
        out = np.zeros(*shape)
        for idx, n in idx_to_nsamples.items():
            # Sample the from the distribution we picked
            mask = (idx == idxs)
            subsample = self.pdfs[idx].sample(n)
            out[mask] = subsample
        return out

    @classmethod
    def random(cls, rng=None, n=3):
        """
        Args:
            rng (int | float | None | numpy.random.RandomState | random.Random):
                random coercable

            n (int):
                number of random distributions in the mixture

        Example:
            >>> from kwarray.distributions import *  # NOQA
            >>> print('Mixture = {!r}'.format(Mixture))
            >>> print('Mixture = {!r}'.format(dir(Mixture)))
            >>> self = Mixture.random(3)
            >>> print('self = {!r}'.format(self))
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> self.plot('0.1s', bins=256)
            >>> kwplot.show_if_requested()
        """
        rng = ensure_rng(rng)
        components = []
        for _ in range(n):
            item = Distribution.random(rng=rng)
            components.append(item)
        weights = rng.rand(n)
        weights = weights / weights.sum()
        self = cls(components, weights)
        return self


class Composed(MixedDistribution):
    """
    A distribution generated by composing different base distributions or
    numbers (which are considered as constant distributions).

    Given the operation and its arguments, the sampling process of a "Composed"
    distribution will sample from each of the operands, and then apply the
    operation to the sampled points. For instance if we add two Normal
    distributions, this will first sample from each distribution and then add
    the results.

    Note:
        This is not the same as mixing distributions!

    Attributes:
        self.operation (Function) :
            operation (add / sub / mult / div) to perform on operands

        self.operands (Sequence[Distribution | Number]) :
            arguments passed to operation

    Example:
        >>> # In this examle you can see that the sum of two Normal random
        >>> # variables is also normal
        >>> from kwarray.distributions import *  # NOQA
        >>> operands = [Normal(mean=10, std=2), Normal(15, 2)]
        >>> operation = np.add
        >>> self = Composed(operation, operands)
        >>> data = self.sample(5)
        >>> print(ub.repr2(list(data), nl=0, precision=5))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> self.plot(1000, bins=100)

    Example:
        >>> # Binary operations result in composed distributions
        >>> # We can make a (bounded) exponential distribution using a uniform
        >>> from kwarray.distributions import *  # NOQA
        >>> X = Uniform(.001, 7)
        >>> lam = .7
        >>> e = np.exp(1)
        >>> self = lam * e ** (-lam * X)
        >>> data = self.sample(5)
        >>> print(ub.repr2(list(data), nl=0, precision=5))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> self.plot(5000, bins=100)

    """
    __params__ = ub.odict([
        ('operation', Value()),
        ('operands', Value()),
    ])

    def sample(self, *shape):
        # resolved_args = [arg.sample(*shape) for arg in self.operands]
        resolved_args = [_trysample(arg, shape) for arg in self.operands]
        return self.operation(*resolved_args)


def _trysample(arg, shape):
    """ samples if arg is a distribution, otherwise returns arg """
    try:
        return arg.sample(*shape)
    except Exception:
        return arg


class CoerceError(ValueError):
    pass


# Temporary backwards compat
CastError = CoerceError


class Uniform(ContinuousDistribution):
    """
    Defaults to a uniform distribution over floats between 0 and 1

    Example:
        >>> from kwarray.distributions import *  # NOQA
        >>> self = Uniform(rng=0)
        >>> self.sample()
        0.548813...
        >>> float(self.sample(1))
        0.7151...

    Benchmark:
        >>> import ubelt as ub
        >>> self = Uniform()
        >>> for timer in ub.Timerit(100, bestof=10):
        >>>     with timer:
        >>>         [self() for _ in range(100)]
        >>> for timer in ub.Timerit(100, bestof=10):
        >>>     with timer:
        >>>         self(100)
    """
    __params__ = ub.odict([
        ('high', Value(default=1)),
        ('low', Value(default=0)),
    ])

    def sample(self, *shape):
        return self.rng.rand(*shape) * (self.high - self.low) + self.low

    @classmethod
    def coerce(cls, arg):
        try:
            self = super(Uniform, cls).coerce(arg)
        except CoerceError:
            if isinstance(arg, (int, float)):
                self = cls(low=0, high=arg)
            else:
                raise
        return self


class Exponential(ContinuousDistribution):
    """
    The exponential distribution is the probability distribution of the time
    between events in a Poisson point process, i.e., a process in which events
    occur continuously and independently at a constant average rate [1]_.

    Referencs:
        .. [1] https://en.wikipedia.org/wiki/Exponential_distribution

    Example:
        >>> from kwarray.distributions import *  # NOQA
        >>> self = Exponential(rng=0)
        >>> self.sample()
        >>> self.sample(2, 3)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> self.plot(500, bins=25)
    """
    __params__ = ub.odict([
        ('scale', Value(1, min=0)),
    ])
    def sample(self, *shape):
        return self.rng.exponential(self.scale, size=shape)


class Constant(DiscreteDistribution):
    """
    Example:
        >>> self = Constant(42, rng=0)
        >>> self.sample()
        42
        >>> self.sample(3)
        array([42, 42, 42])
    """
    __params__ = ub.odict([
        ('value', Value(1, help='constant value')),
    ])
    def sample(self, *shape):
        if shape:
            return np.full(shape, fill_value=self.value)
        else:
            return self.value


class DiscreteUniform(DiscreteDistribution):
    """
    Uniform distribution over integers.

    Args:
        min (int): inclusive minimum
        max (int): exclusive maximum

    Example:
        >>> self = DiscreteUniform.coerce(4)
        >>> self.sample(100)
    """
    __params__ = ub.odict([
        ('min', Value(0)),
        ('max', Value(1)),
    ])

    def sample(self, *shape):
        if len(shape) == 0:
            shape = None
        idx = self.rng.randint(self.min, self.max, size=shape)
        return idx

    @classmethod
    def coerce(cls, arg, rng=None):
        try:
            self = super(DiscreteUniform, cls).coerce(arg, rng=rng)
        except CoerceError:
            if isinstance(arg, (int, float)):
                self = cls(max=arg, rng=rng)
            else:
                raise
        return self


class Normal(ContinuousDistribution):
    """
    A normal distribution. See [WikiNormal]_ [WikiCLT]_.

    References:
        .. [WikiNormal] https://en.wikipedia.org/wiki/Normal_distribution
        .. [WikiCLT] https://en.wikipedia.org/wiki/Central_limit_theorem

    Example:
        >>> from kwarray.distributions import *  # NOQA
        >>> self = Normal(mean=100, rng=0)
        >>> self.sample()
        >>> self.sample(100)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> self.plot(500, bins=25)
    """
    __params__ = ub.odict([
        ('mean', Value(0.0)),
        ('std', Value(1.0, min=1e-3)),
    ])

    def sample(self, *shape):
        return self.rng.randn(*shape) * self.std + self.mean

    @classmethod
    def random(cls, rng=None):
        rng = ensure_rng(rng)
        mean = (rng.rand() * 1024) - 512
        std = np.abs((rng.randn() * 32)) + 1
        return cls(mean=mean, std=std)


class TruncNormal(ContinuousDistribution):
    """
    A truncated normal distribution.

    A normal distribution, but bounded by low and high values. Note this is
    much different from just using a clipped normal.

    Args:
        mean (float): mean of the distribution
        std (float): standard deviation of the distribution
        low (float): lower bound
        high (float): upper bound
        rng (np.random.RandomState):

    CommandLine:
        xdoctest -m /home/joncrall/code/kwarray/kwarray/distributions.py TruncNormal

    Example:
        >>> self = TruncNormal(rng=0)
        >>> self()  # output of this changes before/after scipy version 1.5
        ...0.1226...

    Example:
        >>> from kwarray.distributions import *  # NOQA
        >>> low = -np.pi / 16
        >>> high = np.pi / 16
        >>> std = np.pi / 8
        >>> self = TruncNormal(low=low, high=high, std=std, rng=0)
        >>> shape = (3, 3)
        >>> data = self(*shape)
        >>> print(ub.repr2(data, precision=5))
        np.array([[ 0.01841,  0.0817 ,  0.0388 ],
                  [ 0.01692, -0.0288 ,  0.05517],
                  [-0.02354,  0.15134,  0.18098]], dtype=np.float64)
    """
    __params__ = ub.odict([
        ('mean', Value(0.0)),
        ('std', Value(1.0)),
        ('low', Value(-inf)),
        ('high', Value(inf)),
    ])

    def __init__(self, *args, **kwargs):
        super(TruncNormal, self).__init__(*args, **kwargs)
        self._update_internals()

    def _update_internals(self):
        from scipy.stats import truncnorm
        # Convert high and low values to be wrt the standard normal range
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
        self.a = (self.low - self.mean) / self.std
        self.b = (self.high - self.mean) / self.std
        self.rv = truncnorm(a=self.a, b=self.b, loc=self.mean, scale=self.std)

    @classmethod
    def random(cls, rng=None):
        rng = ensure_rng(rng)
        mean = (rng.rand() * 1024) - 512
        std = np.abs((rng.randn() * 32)) + 1
        low = mean - (std * rng.rand() * 6)
        high = mean + (std * rng.rand() * 6)
        return cls(mean=mean, std=std, low=low, high=high)

    def sample(self, *shape):
        arr = self.rv.rvs(size=shape, random_state=self.rng)
        return arr


class Bernoulli(DiscreteDistribution):
    """
    The Bernoulli distribution is the discrete probability distribution of a
    random variable which takes the value 1 with probability `p` and the value
    0 with probability `q = 1 - p`.

    References:
        https://en.wikipedia.org/wiki/Bernoulli_distribution
    """
    __params__ = ub.odict([
        ('p', Value(0.5, help='probability of success', min=0, max=1)),
    ])

    def sample(self, *shape):
        return self.rng.rand(*shape) < self.p

    @classmethod
    def coerce(cls, arg):
        try:
            self = super(Bernoulli, cls).coerce(arg)
        except CoerceError:
            if isinstance(arg, (int, float)):
                self = cls(p=arg)
            else:
                raise
        return self


class Binomial(DiscreteDistribution):
    """
    The Binomial distribution represents the discrete probabilities of
    obtaining some number of successes in n "binary-experiments" each with a
    probability of success p and a probability of failure 1 - p.

    References:
        https://en.wikipedia.org/wiki/Binomial_distribution
    """
    __params__ = ub.odict([
        ('p', Value(0.5, min=0, max=1, help='probability of success')),
        ('n', Value(1, min=0, help='probability of success')),
    ])

    def sample(self, *shape):
        return self.rng.rand(*shape) > self.p


# class Multinomial(Distribution):
#     """
#     Generalization of binomial such that instead of a binary experiment,
#     the experiment could have k outcomes. Ie the experiment is rolling dice.
#     """
#     __params__ = {
#         'p': {'default': 0.5, 'max': 1, 'min': 0, 'help': 'probability of each side'},
#         'n': {'default': 1, 'min': 0, 'help': 'number of trials'},
#         'k': {'default': 2, 'min': 2, 'help': 'sides on the dice'},

#     }
#     def __init__(self, n=1, p=.5, rng=None):
#         super().__init__(**locals())


class Categorical(DiscreteDistribution):
    """
    Example:
        >>> categories = [3, 5, 1]
        >>> weights = [.05, .5, .45]
        >>> self = Categorical(categories, weights, rng=0)
        >>> self.sample()
        5
        >>> list(self.sample(2))
        [1, 1]
        >>> self.sample(2, 3)
        array([[5, 5, 1],
               [5, 1, 1]])
    """
    def __init__(self, categories, weights=None, rng=None):
        super(Categorical, self).__init__(rng=rng)
        self._setparam('categories', np.array(categories))
        self._setparam('weights', weights)
        self._idxs = np.arange(len(self.categories))

    def sample(self, *shape):
        if len(shape) == 0:
            shape = None
        idxs = self.rng.choice(self._idxs, size=shape, p=self.weights)
        # if isinstance(self.categories, np.ndarray):
        return self.categories[idxs]
        # return self.rng.choice(self.categories, size=shape, p=self.weights)


class NonlinearUniform(ContinuousDistribution):
    """
    Weighted sample between two points depending on some nonlinearity

    TODO:
        could refactor part of this into a PowerLaw distribution

    Args:
        nonlinearity (func or str): needs to be a function that maps
             the range 0-1 to the range 0-1

    Example:
        >>> self = NonlinearUniform(0, 100, np.sqrt, rng=0)
        >>> print(ub.repr2(list(self.sample(2)), precision=2, nl=0))
        [74.08, 84.57]
        >>> print(ub.repr2(self.sample(2, 3), precision=2, nl=1))
        np.array([[77.64, 73.82, 65.09],
                  [80.37, 66.15, 94.43]], dtype=np.float64)

    Ignore:
        bins = np.linspace(0, 100, 100)
        NonlinearUniform(0, 100, None, rng=0).plot(10000, bins=bins)
        NonlinearUniform(0, 100, 'sqrt', rng=0).plot(10000, bins=bins)
        NonlinearUniform(0, 100, 'squared', rng=0).plot(10000, bins=bins)
        NonlinearUniform(0, 100, 'log', rng=0).plot(10000, bins=bins)
        NonlinearUniform(0, 100, 'exp', rng=0).plot(10000, bins=bins)
        NonlinearUniform(0, 100, 0.25, rng=0).plot(10000, bins=bins)
        NonlinearUniform(0, 100, 4.0, rng=0).plot(10000, bins=bins)
        NonlinearUniform(0, 100, 2.0, rng=0).plot(10000, bins=bins)
    """
    def __init__(self, min, max, nonlinearity=None, reverse=False, rng=None):
        super(NonlinearUniform, self).__init__(rng=rng)
        self._setparam('min', min)
        self._setparam('max', max)
        if nonlinearity == 'linear' or nonlinearity is None:
            nonlinearity = None
        elif nonlinearity == 'exp':
            nonlinearity = lambda x: (np.exp(x) - 1) / (np.exp(1) - 1)  # NOQA
        elif nonlinearity == 'log':
            nonlinearity = lambda x: np.log(x + 1) / np.log(2)  # NOQA
        elif nonlinearity == 'squared':
            nonlinearity = lambda x: np.power(x, 2)  # NOQA
        elif nonlinearity == 'sqrt':
            nonlinearity = np.sqrt
        elif isinstance(nonlinearity, float):
            # Use a powerlaw
            param = nonlinearity
            nonlinearity = lambda x: np.power(x, param)  # NOQA
        elif not callable(nonlinearity):
            raise KeyError(nonlinearity)
        self._setparam('nonlinearity', nonlinearity)
        self._setparam('reverse', reverse)

    def sample(self, *shape):
        if len(shape) == 0:
            base = self.rng.rand()
        else:
            base = self.rng.rand(*shape)
        if self.reverse:
            base = 1 - base
        if self.nonlinearity is not None:
            base = self.nonlinearity(base)
        return base * (self.max - self.min) + self.min


class CategoryUniform(DiscreteUniform):
    """
    Discrete Uniform over a list of categories
    """
    def __init__(self, categories=[None], rng=None):
        super(CategoryUniform, self).__init__(rng=rng)
        self._setparam('categories', np.array(categories))
        self._num = len(self.categories)

    def sample(self, *shape):
        if len(shape) == 0:
            shape = None
        idx = self.rng.randint(0, self._num, size=shape)
        return self.categories[idx]


class PDF(Distribution):
    """
    BROKEN?

    Similar to Categorical, but interpolates to approximate a continuous random
    variable.

    Returns a value x with probability p.

    References:
        http://www.nehalemlabs.net/prototype/blog/2013/12/16/how-to-do-inverse-transformation-sampling-in-scipy-and-numpy/

    Args:
        x (list or tuple): domain in which this PDF is defined
        p (list): probability sample for each domain sample

    Example:
        >>> from kwarray.distributions import PDF # NOQA
        >>> x = np.linspace(800, 4500)
        >>> p = np.log10(x)
        >>> p = x ** 2
        >>> self = PDF(x, p)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> self.plot(5000, bins=50)
    """
    def __init__(self, x, p, rng=None):
        import scipy.interpolate as interpolate
        super(PDF, self).__init__(rng=rng)

        if isinstance(x, tuple):
            self.edge_x = np.linspace(x[0], x[1], len(p) + 1)
        else:
            # Need to convert point samples to edges
            self.edge_x = interpolate.interp1d(np.linspace(0, 1, len(x)), x)(np.linspace(0, 1, len(x) + 1))
            # FIXME: might have a bug causing probability to be shifted slightly

        # Normalize probabilities
        p = p / p.sum()

        # Calculate CDF and interpolate edges
        self.edge_cdf = np.hstack([[0], np.cumsum(p)])

        # FIXME: This is not actually the inverse. Like, at all.
        # In fact its the inverse of the inverse.
        self.inv_cdf = interpolate.interp1d(self.edge_cdf, self.edge_x)

    def sample(self, *shape):
        # Right idea, but wrong implementation
        r = np.random.rand(*shape)
        return self.inv_cdf(r)


class Seeded(object):
    """
    Helper for grabbing pre-seeded distributions
    """
    def __init__(self, rng=None, cls=None):
        self.rng = rng
        self.cls = cls

    def __call__(self, *args, **kwargs):
        return self.cls(*args, rng=self.rng, **kwargs)

    def __getattr__(self, key):
        return functools.partial(globals()[key], rng=self.rng)


def _test_distributions():
    from kwarray import distributions as stoch
    cls_list = []
    for cls in vars(stoch).values():
        if isinstance(cls, type):
            if issubclass(cls, stoch.Distribution):
                if cls is not stoch.Distribution:
                    cls_list.append(cls)

    for cls in cls_list:
        rng = np.random.RandomState(0)
        inst = cls(rng=rng)
        scalar_sample = inst.sample()
        vector_sample = inst.sample(1)
        print('inst = {!r}'.format(inst))
        print('scalar_sample = {!r}'.format(scalar_sample))
        print('vector_sample = {!r}'.format(vector_sample))
        assert not ub.iterable(scalar_sample)
        assert ub.iterable(vector_sample)
        assert vector_sample.shape == (1,)
