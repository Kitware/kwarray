# -*- coding: utf-8 -*-
"""
A faster-than-pandas pandas-like interface to column-major data, in the case
where the data only needs to be accessed by index.

For data where more complex ids are needed you must use pandas.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import ubelt as ub
import numpy as np
import copy

try:
    import pandas as pd
except Exception:
    pd = None


__version__ = '0.0.1'


class LocLight(object):
    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, index):
        return self.parent._getrow(index)


class DataFrameLight(ub.NiceRepr):
    r"""
    Implements a subset of the pandas.DataFrame API

    The API is restricted to facilitate speed tradeoffs

    Notes:
        Assumes underlying data is Dict[list|ndarray]. If the data is known
        to be a Dict[ndarray] use DataFrameArray instead, which has faster
        implementations for some operations.

    Notes:
        pandas.DataFrame is slow. DataFrameLight is faster.
        It is a tad more restrictive though.

    Example:
        >>> self = DataFrameLight({})
        >>> print('self = {!r}'.format(self))
        >>> self = DataFrameLight({'a': [0, 1, 2], 'b': [2, 3, 4]})
        >>> print('self = {!r}'.format(self))
        >>> item = self.iloc[0]
        >>> print('item = {!r}'.format(item))

    Benchmark:
        >>> # BENCHMARK
        >>> # xdoc: +REQUIRES(--bench)
        >>> from kwarray.dataframe_light import *  # NOQA
        >>> import ubelt as ub
        >>> NUM = 1000
        >>> print('NUM = {!r}'.format(NUM))
        >>> # to_dict conversions
        >>> print('==============')
        >>> print('====== to_dict conversions =====')
        >>> _keys = ['list', 'dict', 'series', 'split', 'records', 'index']
        >>> results = []
        >>> df = DataFrameLight._demodata(num=NUM).pandas()
        >>> ti = ub.Timerit(verbose=False, unit='ms')
        >>> for key in _keys:
        >>>     result = ti.reset(key).call(lambda: df.to_dict(orient=key))
        >>>     results.append((result.mean(), result.report()))
        >>> key = 'series+numpy'
        >>> result = ti.reset(key).call(lambda: {k: v.values for k, v in df.to_dict(orient='series').items()})
        >>> results.append((result.mean(), result.report()))
        >>> print('\n'.join([t[1] for t in sorted(results)]))
        >>> print('==============')
        >>> print('====== DFLight Conversions =======')
        >>> ti = ub.Timerit(verbose=True, unit='ms')
        >>> key = 'self.pandas'
        >>> self = DataFrameLight(df)
        >>> ti.reset(key).call(lambda: self.pandas())
        >>> key = 'light-from-pandas'
        >>> ti.reset(key).call(lambda: DataFrameLight(df))
        >>> key = 'light-from-dict'
        >>> ti.reset(key).call(lambda: DataFrameLight(self._data))
        >>> print('==============')
        >>> print('====== BENCHMARK: .LOC[] =======')
        >>> ti = ub.Timerit(num=20, bestof=4, verbose=True, unit='ms')
        >>> df_light = DataFrameLight._demodata(num=NUM)
        >>> # xdoctest: +REQUIRES(module:pandas)
        >>> df_heavy = df_light.pandas()
        >>> series_data = df_heavy.to_dict(orient='series')
        >>> list_data = df_heavy.to_dict(orient='list')
        >>> np_data = {k: v.values for k, v in df_heavy.to_dict(orient='series').items()}
        >>> for timer in ti.reset('DF-heavy.iloc'):
        >>>     with timer:
        >>>         for i in range(NUM):
        >>>             df_heavy.iloc[i]
        >>> for timer in ti.reset('DF-heavy.loc'):
        >>>     with timer:
        >>>         for i in range(NUM):
        >>>             df_heavy.iloc[i]
        >>> for timer in ti.reset('dict[SERIES].loc'):
        >>>     with timer:
        >>>         for i in range(NUM):
        >>>             {key: series_data[key].loc[i] for key in series_data.keys()}
        >>> for timer in ti.reset('dict[SERIES].iloc'):
        >>>     with timer:
        >>>         for i in range(NUM):
        >>>             {key: series_data[key].iloc[i] for key in series_data.keys()}
        >>> for timer in ti.reset('dict[SERIES][]'):
        >>>     with timer:
        >>>         for i in range(NUM):
        >>>             {key: series_data[key][i] for key in series_data.keys()}
        >>> for timer in ti.reset('dict[NDARRAY][]'):
        >>>     with timer:
        >>>         for i in range(NUM):
        >>>             {key: np_data[key][i] for key in np_data.keys()}
        >>> for timer in ti.reset('dict[list][]'):
        >>>     with timer:
        >>>         for i in range(NUM):
        >>>             {key: list_data[key][i] for key in np_data.keys()}
        >>> for timer in ti.reset('DF-Light.iloc/loc'):
        >>>     with timer:
        >>>         for i in range(NUM):
        >>>             df_light.iloc[i]
        >>> for timer in ti.reset('DF-Light._getrow'):
        >>>     with timer:
        >>>         for i in range(NUM):
        >>>             df_light._getrow(i)
        NUM = 1000
        ==============
        ====== to_dict conversions =====
        Timed best=0.022 ms, mean=0.022 ± 0.0 ms for series
        Timed best=0.059 ms, mean=0.059 ± 0.0 ms for series+numpy
        Timed best=0.315 ms, mean=0.315 ± 0.0 ms for list
        Timed best=0.895 ms, mean=0.895 ± 0.0 ms for dict
        Timed best=2.705 ms, mean=2.705 ± 0.0 ms for split
        Timed best=5.474 ms, mean=5.474 ± 0.0 ms for records
        Timed best=7.320 ms, mean=7.320 ± 0.0 ms for index
        ==============
        ====== DFLight Conversions =======
        Timed best=1.798 ms, mean=1.798 ± 0.0 ms for self.pandas
        Timed best=0.064 ms, mean=0.064 ± 0.0 ms for light-from-pandas
        Timed best=0.010 ms, mean=0.010 ± 0.0 ms for light-from-dict
        ==============
        ====== BENCHMARK: .LOC[] =======
        Timed best=101.365 ms, mean=101.564 ± 0.2 ms for DF-heavy.iloc
        Timed best=102.038 ms, mean=102.273 ± 0.2 ms for DF-heavy.loc
        Timed best=29.357 ms, mean=29.449 ± 0.1 ms for dict[SERIES].loc
        Timed best=21.701 ms, mean=22.014 ± 0.3 ms for dict[SERIES].iloc
        Timed best=11.469 ms, mean=11.566 ± 0.1 ms for dict[SERIES][]
        Timed best=0.807 ms, mean=0.826 ± 0.0 ms for dict[NDARRAY][]
        Timed best=0.478 ms, mean=0.492 ± 0.0 ms for dict[list][]
        Timed best=0.969 ms, mean=0.994 ± 0.0 ms for DF-Light.iloc/loc
        Timed best=0.760 ms, mean=0.776 ± 0.0 ms for DF-Light._getrow

    """
    def __init__(self, data=None, columns=None):
        if columns is not None:
            if data is None:
                data = ub.odict(zip(columns, [[] for _ in range(len(columns))]))
            else:
                data = ub.odict(zip(columns, data.T))

        self._raw = data
        self._data = None
        self._localizer = LocLight(self)
        self.__normalize__()

    @property
    def iloc(self):
        return self._localizer

    @property
    def values(self):
        data = self._getcols(self.columns)
        return data

    @property
    def loc(self):
        return self._localizer

    def __eq__(self, other):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:pandas)
            >>> self = DataFrameLight._demodata(num=7)
            >>> other = self.pandas()
            >>> assert np.all(self == other)
        """
        self_vals = self.values
        if isinstance(other, DataFrameLight):
            other_vals = other.values
        if pd is not None and isinstance(other, pd.DataFrame):
            other_vals = other.reindex(columns=self.columns).values
        else:
            other_vals = other
        return self_vals == other_vals

    def to_string(self, *args, **kwargs):
        return self.pandas().to_string(*args, **kwargs)

    def to_dict(self, orient='dict', into=dict):
        """
        Convert the data frame into a dictionary.

        Args:
            orient (str): Currently naitively suports orient in
                {'dict', 'list'}, otherwise we fallback to pandas conversion
                and call its to_dict method.

            into (type): type of dictionary to transform into

        Returns:
           dict

        Example:
            >>> from kwarray.dataframe_light import *  # NOQA
            >>> self = DataFrameLight._demodata(num=7)
            >>> print(self.to_dict(orient='dict'))
            >>> print(self.to_dict(orient='list'))
        """
        if orient == 'dict':
            out = into(self.iterrows())
        elif orient == 'list':
            import kwarray
            out = into((k, kwarray.ArrayAPI.tolist(v))
                        for k, v in self._data.items())
        else:
            out = self.pandas().to_dict(orient=orient, into=into)
        return out

    def pandas(self):
        """
        Convert back to pandas if you need the full API

        Example:
            >>> # xdoctest: +REQUIRES(module:pandas)
            >>> df_light = DataFrameLight._demodata(num=7)
            >>> df_heavy = df_light.pandas()
            >>> got = DataFrameLight(df_heavy)
            >>> assert got._data == df_light._data
        """
        if pd is None:
            raise Exception('Pandas is not available')
        return pd.DataFrame(self._data)

    def _pandas(self):
        """ Deprecated, use self.pandas instead """
        return self.pandas()

    @classmethod
    def _demodata(cls, num=7):
        """
        Example:
            >>> self = DataFrameLight._demodata(num=7)
            >>> print('self = {!r}'.format(self))
            >>> other = DataFrameLight._demodata(num=11)
            >>> print('other = {!r}'.format(other))
            >>> both = self.union(other)
            >>> print('both = {!r}'.format(both))
            >>> assert both is not self
            >>> assert other is not self
        """
        demodata = {
            'foo': [0] * num,
            'bar': [i % 3 for i in range(num)],
            'baz': [2.73] * num,
        }
        self = cls(demodata)
        return self

    def __nice__(self):
        return 'keys: {}, len={}'.format(list(self.keys()), len(self))

    def __len__(self):
        if self._data:
            key = next(iter(self.keys()))
            return len(self._data[key])
        else:
            return 0

    def __contains__(self, item):
        return item in self.keys()

    def __normalize__(self):
        """
        Try to convert input data to Dict[List]
        """
        if self._raw is None:
            self._data = {}
        elif isinstance(self._raw, dict):
            self._data = self._raw
            if __debug__:
                lens = []
                for d in self._data.values():
                    if not isinstance(d, (list, np.ndarray)):
                        raise TypeError(type(d))
                    lens.append(len(d))
                assert ub.allsame(lens)
        elif isinstance(self._raw, DataFrameLight):
            self._data = copy.copy(self._raw._data)
        elif pd is not None and isinstance(self._raw, pd.DataFrame):
            self._data = self._raw.to_dict(orient='list')
        else:
            raise TypeError('Unknown _raw type')

    @property
    def columns(self):
        return list(self.keys())

    def sort_values(self, key, inplace=False, ascending=True):
        sortx = np.argsort(self._getcol(key))
        if not ascending:
            sortx = sortx[::-1]
        return self.take(sortx, inplace=inplace)

    def keys(self):
        if self._data:
            for key in self._data.keys():
                yield key

    def _getrow(self, index):
        return {key: self._data[key][index] for key in self._data.keys()}

    def _getcol(self, key):
        return self._data[key]

    def _getcols(self, keys):
        num = len(self)
        col_data = [self._getcol(key) for key in keys]
        data = np.hstack([np.asarray(d).reshape(num, -1) for d in col_data])
        return data

    def get(self, key, default=None):
        """
        Get item for given key. Returns default value if not found.
        """
        return self[key] if key in self else default

    def clear(self):
        """
        Removes all rows inplace
        """
        if self._data:
            for key in self._data.keys():
                self._data[key].clear()

    def __getitem__(self, key):
        """
        Note:
            only handles the case where key is a single column name.

        Example:
            >>> df_light = DataFrameLight._demodata(num=7)
            >>> sub1 = df_light['bar']
            >>> # xdoctest: +REQUIRES(module:pandas)
            >>> df_heavy = df_light.pandas()
            >>> sub2 = df_heavy['bar']
            >>> assert np.all(sub1 == sub2)
        """
        return self._getcol(key)

    def __setitem__(self, key, value):
        """
        Note:
            only handles the case where key is a single column name. and value
            is an array of all the values to set.

        Example:
            >>> df_light = DataFrameLight._demodata(num=7)
            >>> value = [2] * len(df_light)
            >>> df_light['bar'] = value
            >>> # xdoctest: +REQUIRES(module:pandas)
            >>> df_heavy = df_light.pandas()
            >>> df_heavy['bar'] = value
            >>> assert np.all(df_light == df_heavy)
        """
        self._data[key] = value

    def compress(self, flags, inplace=False):
        """
        NOTE: NOT A PART OF THE PANDAS API
        """
        subset = self if inplace else self.__class__()
        for key in self._data.keys():
            subset._data[key] = list(ub.compress(self._data[key], flags))
        return subset

    def take(self, indices, inplace=False):
        """
        Return the elements in the given *positional* indices along an axis.

        Args:
            inplace (bool): NOT PART OF PANDAS API

        Notes:
            assumes axis=0

        Example:
            >>> df_light = DataFrameLight._demodata(num=7)
            >>> indices = [0, 2, 3]
            >>> sub1 = df_light.take(indices)
            >>> # xdoctest: +REQUIRES(module:pandas)
            >>> df_heavy = df_light.pandas()
            >>> sub2 = df_heavy.take(indices)
            >>> assert np.all(sub1 == sub2)
        """
        subset = self if inplace else self.__class__()
        if isinstance(indices, slice):
            for key in self._data.keys():
                subset._data[key] = self._data[key][indices]
        else:
            for key in self._data.keys():
                subset._data[key] = list(ub.take(self._data[key], indices))
        return subset

    def copy(self):
        other = copy.copy(self)
        other._data = other._data.copy()
        other._localizer = LocLight(other)
        return other

    def extend(self, other):
        """
        Extend ``self`` inplace using another dataframe array

        Args:
            other (DataFrameLight | dict[str, Sequence]):
                values to concat to end of this object

        NOTE:
            Not part of the pandas API

        Example:
            >>> self = DataFrameLight(columns=['foo', 'bar'])
            >>> other = {'foo': [0], 'bar': [1]}
            >>> self.extend(other)
            >>> assert len(self) == 1
        """
        try:
            _other_data = other._data
        except AttributeError:
            _other_data = other

        _self_data = self._data
        for key, vals1 in _self_data.items():
            vals2 = _other_data[key]
            try:
                vals1.extend(vals2)
            except AttributeError:
                if isinstance(vals1, np.ndarray):
                    _self_data[key] = np.hstack([vals1, vals2])
                else:
                    raise

    def union(self, *others):
        """
        NOTE:
            Note part of the pandas API
        """
        if isinstance(self, DataFrameLight):
            first = self
            rest = others
        else:
            if len(others) == 0:
                return DataFrameLight()
            first = others[0]
            rest = others[1:]

        both = first.copy()
        if not both.keys:
            for other in rest:
                if other.keys:
                    both.keys = copy.copy(other.keys)
                    break

        for other in rest:
            both.extend(other)
        return both

    @classmethod
    def concat(cls, others):
        return cls.union(*others)

    @classmethod
    def from_pandas(cls, df):
        _raw = {k: v.values for k, v in df.to_dict(orient='series').items()}
        return cls(_raw)

    @classmethod
    def from_dict(cls, records):
        record_iter = iter(records)
        columns = {}
        try:
            r = next(record_iter)
            for key, value in r.items():
                columns[key] = [value]
        except StopIteration:
            pass
        else:
            for r in record_iter:
                for key, value in r.items():
                    columns[key].append(value)
        self = cls(columns)
        return self

    def reset_index(self, drop=False):
        """ noop for compatability, the light version doesnt store an index """
        return self

    def groupby(self, by=None, *args, **kwargs):
        """
        Group rows by the value of a column. Unlike pandas this simply
        returns a zip object. To ensure compatiability call list on the
        result of groupby.

        Args:
            by (str): column name to group by
            *args: if specified, the dataframe is coerced to pandas
            *kwargs: if specified, the dataframe is coerced to pandas

        Example:
            >>> df_light = DataFrameLight._demodata(num=7)
            >>> res1 = list(df_light.groupby('bar'))
            >>> # xdoctest: +REQUIRES(module:pandas)
            >>> df_heavy = df_light.pandas()
            >>> res2 = list(df_heavy.groupby('bar'))
            >>> assert len(res1) == len(res2)
            >>> assert all([np.all(a[1] == b[1]) for a, b in zip(res1, res2)])

        Ignore:
            >>> self = DataFrameLight._demodata(num=1000)
            >>> args = ['cx']
            >>> self['cx'] = (np.random.rand(len(self)) * 10).astype(np.int)
            >>> # As expected, our custom restricted implementation is faster
            >>> # than pandas
            >>> ub.Timerit(100).call(lambda: dict(list(self.pandas().groupby('cx')))).print()
            >>> ub.Timerit(100).call(lambda: dict(self.groupby('cx'))).print()
        """
        if len(args) == 0 and len(kwargs) == 0:
            # In this special case we can be fast
            import kwarray
            unique, groupxs = kwarray.group_indices(self[by])
            groups = [self.take(idxs) for idxs in groupxs]
            return zip(unique, groups)
        else:
            # otherwise we need to use the slow method
            return self.pandas().groupby(by=by)

    def rename(self, mapper=None, columns=None, axis=None, inplace=False):
        """
        Rename the columns (index renaming is not supported)

        Example:
            >>> df_light = DataFrameLight._demodata(num=7)
            >>> mapper = {'foo': 'fi'}
            >>> res1 = df_light.rename(columns=mapper)
            >>> res3 = df_light.rename(mapper, axis=1)
            >>> # xdoctest: +REQUIRES(module:pandas)
            >>> df_heavy = df_light.pandas()
            >>> res2 = df_heavy.rename(columns=mapper)
            >>> res4 = df_heavy.rename(mapper, axis=1)
            >>> assert np.all(res1 == res2)
            >>> assert np.all(res3 == res2)
            >>> assert np.all(res3 == res4)
        """
        if columns is not None:
            if axis is not None:
                raise TypeError("Cannot specify both 'axis' and any of 'index' or 'columns'")
        else:
            if axis != 1:
                raise NotImplementedError('only axis=1 is supported')
            columns = mapper

        if not inplace:
            self = self.copy()
        for old, new in columns.items():
            if old in self._data:
                self._data[new] = self._data.pop(old)
        return self

    def iterrows(self):
        """
        Iterate over rows as (index, Dict) pairs.

        Yields:
            Tuple[int, Dict]: the index and a dictionary representing a row

        Example:
            >>> from kwarray.dataframe_light import *  # NOQA
            >>> self = DataFrameLight._demodata(num=3)
            >>> print(ub.repr2(list(self.iterrows())))
            [
                (0, {'bar': 0, 'baz': 2.73, 'foo': 0}),
                (1, {'bar': 1, 'baz': 2.73, 'foo': 0}),
                (2, {'bar': 2, 'baz': 2.73, 'foo': 0}),
            ]

        Benchmark:
            >>> # xdoc: +REQUIRES(--bench)
            >>> from kwarray.dataframe_light import *  # NOQA
            >>> import ubelt as ub
            >>> df_light = DataFrameLight._demodata(num=1000)
            >>> df_heavy = df_light.pandas()
            >>> ti = ub.Timerit(21, bestof=3, verbose=2, unit='ms')
            >>> ti.reset('light').call(lambda: list(df_light.iterrows()))
            >>> ti.reset('heavy').call(lambda: list(df_heavy.iterrows()))
            >>> # xdoctest: +IGNORE_WANT
            Timed light for: 21 loops, best of 3
                time per loop: best=0.834 ms, mean=0.850 ± 0.0 ms
            Timed heavy for: 21 loops, best of 3
                time per loop: best=45.007 ms, mean=45.633 ± 0.5 ms
        """
        for idx in range(len(self)):
            row = self._getrow(idx)
            yield idx, row


class DataFrameArray(DataFrameLight):
    """
    DataFrameLight assumes the backend is a Dict[list]
    DataFrameArray assumes the backend is a Dict[ndarray]

    Take and compress are much faster, but extend and union are slower
    """

    def __normalize__(self):
        """
        Try to convert input data to Dict[ndarray]
        """
        if self._raw is None:
            self._data = {}
        elif isinstance(self._raw, dict):
            self._data = self._raw
            if __debug__:
                lens = []
                for d in self._data.values():
                    if not isinstance(d, (list, np.ndarray)):
                        raise TypeError(type(d))
                    lens.append(len(d))
                assert ub.allsame(lens), (
                    'lens are not all same {} for columns {}'.format(
                        lens,
                        list(self._data.keys()))
                )
        elif isinstance(self._raw, DataFrameLight):
            self._data = copy.copy(self._raw._data)
        elif pd is not None and isinstance(self._raw, pd.DataFrame):
            self._data = {k: v.values for k, v in self._raw.to_dict(orient='series').items()}
        else:
            raise TypeError('Unknown _raw type')
        # self._data = ub.map_vals(np.asarray, self._data)  # does this break anything?

    def extend(self, other):
        for key in self._data.keys():
            vals1 = self._data[key]
            vals2 = other._data[key]
            self._data[key] = np.hstack([vals1, vals2])

    def compress(self, flags, inplace=False):
        subset = self if inplace else self.__class__()
        for key in self._data.keys():
            subset._data[key] = self._data[key][flags]
        return subset

    def take(self, indices, inplace=False):
        subset = self if inplace else self.__class__()
        for key in self._data.keys():
            subset._data[key] = self._data[key][indices]
        return subset

    # def min(self, axis=None):
    #     return self._extreme(func=np.minimum, axis=axis)

    # def max(self, axis=None):
    #     """
    #     Example:
    #         >>> self = DataFrameArray._demodata(num=7)
    #         >>> func = np.maximum
    #     """
    #     return self._extreme(func=np.maximum, axis=axis)

    # def _extreme(self, func, axis=None):
    #     if axis is None:
    #         raise NotImplementedError
    #     if axis == 0:
    #         raise NotImplementedError
    #     elif axis == 1:
    #         newdata = nh.util.iter_reduce_ufunc(func, (self[key] for key in self.keys()))
    #         newobj = self.__class__(newdata, self._keys)
    #     else:
    #         raise NotImplementedError
