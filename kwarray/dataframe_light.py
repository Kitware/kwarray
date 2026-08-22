from __future__ import annotations

"""
A faster-than-pandas pandas-like interface to column-major data, in the case
where the data only needs to be accessed by index.

For data where more complex ids are needed you must use pandas.

TODO:
    - [ ] Rename to DataFrameLite?
"""

import ubelt as ub
import numpy as np
import copy
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator, Mapping, Sequence

    import pandas as pd

    ColumnData = list[object] | np.ndarray
else:
    try:
        import pandas as pd
    except Exception:
        pd = None


__version__ = '0.0.1'


class LocLight:
    parent: DataFrameLight

    def __init__(self, parent: DataFrameLight) -> None:
        self.parent = parent

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.parent._getrow(index)


class DataFrameLight(ub.NiceRepr):
    r"""
    Implements a subset of the pandas.DataFrame API

    The API is restricted to facilitate speed tradeoffs

    Note:
        Assumes underlying data is Dict[list|ndarray]. If the data is known
        to be a Dict[ndarray] use DataFrameArray instead, which has faster
        implementations for some operations.

    Note:
        THe DataFrameLight has no distinction between ids and indexes (i.e. loc
        and iloc). They are always the same.

    Note:
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
        >>> import timerit
        >>> NUM = 1000
        >>> print('NUM = {!r}'.format(NUM))
        >>> # to_dict conversions
        >>> print('==============')
        >>> print('====== to_dict conversions =====')
        >>> _keys = ['list', 'dict', 'series', 'split', 'records', 'index']
        >>> results = []
        >>> df = DataFrameLight._demodata(num=NUM).pandas()
        >>> ti = timerit.Timerit(verbose=False, unit='ms')
        >>> for key in _keys:
        >>>     result = ti.reset(key).call(lambda: df.to_dict(orient=key))
        >>>     results.append((result.mean(), result.report()))
        >>> key = 'series+numpy'
        >>> result = ti.reset(key).call(lambda: {k: v.values for k, v in df.to_dict(orient='series').items()})
        >>> results.append((result.mean(), result.report()))
        >>> print('\n'.join([t[1] for t in sorted(results)]))
        >>> print('==============')
        >>> print('====== DFLight Conversions =======')
        >>> ti = timerit.Timerit(verbose=True, unit='ms')
        >>> key = 'self.pandas'
        >>> self = DataFrameLight(df)
        >>> ti.reset(key).call(lambda: self.pandas())
        >>> key = 'light-from-pandas'
        >>> ti.reset(key).call(lambda: DataFrameLight(df))
        >>> key = 'light-from-dict'
        >>> ti.reset(key).call(lambda: DataFrameLight(self._data))
        >>> print('==============')
        >>> print('====== BENCHMARK: .LOC[] =======')
        >>> ti = timerit.Timerit(num=20, bestof=4, verbose=True, unit='ms')
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

    _raw: object | None
    _data: dict[str, ColumnData]
    _localizer: LocLight

    def __init__(
        self,
        data: object | None = None,
        columns: Sequence[str] | None = None,
    ) -> None:
        if columns is not None:
            if data is None:
                data = ub.odict(zip(columns, [[] for _ in range(len(columns))]))
            else:
                data = ub.odict(zip(columns, cast(Any, data).T))

        self._raw = data
        self._data = {}
        self._localizer = LocLight(self)
        self.__normalize__()

    @property
    def iloc(self) -> LocLight:
        return self._localizer

    @property
    def values(self) -> np.ndarray:
        data = self._getcols(self.columns)
        return data

    @property
    def loc(self) -> LocLight:
        return self._localizer

    def __eq__(self, other: object) -> np.ndarray:  # ty: ignore[invalid-method-override]
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

    def to_string(self, *args: object, **kwargs: object) -> str:
        """
        Returns:
            str:
        """
        return cast(Any, self.pandas()).to_string(*args, **kwargs)

    def to_dict(self, orient: str = 'dict', into: type = dict) -> dict:
        """
        Convert the data frame into a dictionary.

        Args:
            orient (str): Currently naitively supports orient in
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
            out = into(
                (k, v.tolist() if isinstance(v, np.ndarray) else list(v))
                for k, v in self._data.items()
            )
        else:
            out = cast(Any, self.pandas()).to_dict(orient=orient, into=into)
        return out

    def pandas(self) -> pd.DataFrame:
        """
        Convert back to pandas if you need the full API

        Returns:
            pd.DataFrame:

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

    def _pandas(self) -> pd.DataFrame:
        """Deprecated, use self.pandas instead"""
        return self.pandas()

    @classmethod
    def _demodata(cls, num: int = 7) -> DataFrameLight:
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

    def __nice__(self) -> str:
        return 'keys: {}, len={}'.format(list(self.keys()), len(self))

    def __len__(self) -> int:
        if self._data:
            key = next(iter(self.keys()))
            return len(self._data[key])
        else:
            return 0

    def __contains__(self, item: str) -> bool:
        return item in self.keys()

    def __normalize__(self) -> None:
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
                assert ub.allsame(lens), f'Got unequal lens: {lens}'
        elif isinstance(self._raw, DataFrameLight):
            self._data = copy.copy(self._raw._data)
        elif pd is not None and isinstance(self._raw, pd.DataFrame):
            self._data = self._raw.to_dict(orient='list')
        else:
            raise TypeError('Unknown _raw type')

    @property
    def columns(self) -> list[str]:
        return list(self.keys())

    def sort_values(
        self,
        key: str,
        inplace: bool = False,
        ascending: bool = True,
    ) -> DataFrameLight:
        sortx = np.argsort(cast(Any, self._getcol(key)))
        if not ascending:
            sortx = sortx[::-1]
        return self.take(sortx, inplace=inplace)

    def drop(
        self,
        labels: str | Sequence[int] | Sequence[str],
        *,
        axis: int = 0,
    ) -> DataFrameLight:
        """
        Drop specified labels from rows or columns.

        Example
            >>> from kwarray.dataframe_light import *  # NOQA
            >>> self = DataFrameLight._demodata(num=7)
            >>> labels = 'foo'
            >>> # test drop columns
            >>> new = self.drop(labels, axis=1)
            >>> assert 'foo' not in new.columns
            >>> assert 'bar' in new.columns
            >>> # test drop rows
            >>> new = self.drop([1, 3], axis=0)
            >>> assert len(new) == (len(self) - 2)
            >>> # test drop non-existing
            >>> import pytest
            >>> with pytest.raises(KeyError):
            >>>     new = self.drop('not-exist', axis=1)
            >>> with pytest.raises(KeyError):
            >>>     new = self.drop('not-exist', axis=0)
            >>> with pytest.raises(ValueError):
            >>>     new = self.drop(None, axis=3)
        """
        if axis == 1:
            # Drop columns
            if isinstance(labels, str):
                column_labels = {labels}
            else:
                column_labels = set(cast('Sequence[str]', labels))
            columns = ub.oset(self.columns)
            missing = column_labels - columns
            if missing:
                raise KeyError(f'{missing} not found in axis')
            new_columns = list(columns - column_labels)
            new_data = {c: self._data[c] for c in new_columns}
            new = self.__class__(new_data)
        elif axis == 0:
            # Drop rows
            if isinstance(labels, str):
                raise KeyError('row indexes not found in axis')
            keep = np.ones(len(self), dtype=bool)
            row_labels: Any = labels
            if isinstance(row_labels, tuple):
                row_labels = list(row_labels)
            try:
                keep[row_labels] = False
            except IndexError:
                raise KeyError('row indexes not found in axis')
            new = self.compress(keep)
        else:
            raise ValueError('axis can only be 0 or 1')
        return new

    @property
    def shape(self) -> tuple[int, int]:
        n_rows = len(self)
        n_cols = len(self._data)
        return n_rows, n_cols

    def keys(self) -> Generator[str, None, None]:
        if self._data:
            for key in self._data.keys():
                yield key

    def _getrow(self, index: int) -> dict[str, object]:
        return {key: self._data[key][index] for key in self._data.keys()}

    def _getcol(self, key: str) -> ColumnData:
        return self._data[key]

    def _getcols(self, keys: Sequence[str]) -> np.ndarray:
        num = len(self)
        col_data = [self._getcol(key) for key in keys]
        data = np.hstack([np.asarray(d).reshape(num, -1) for d in col_data])
        return data

    def get(
        self,
        key: str,
        default: object | None = None,
    ) -> ColumnData | object | None:
        """
        Get item for given key. Returns default value if not found.
        """
        return self[key] if key in self else default

    def clear(self) -> None:
        """
        Removes all rows inplace
        """
        if self._data:
            for key, values in self._data.items():
                if isinstance(values, list):
                    values.clear()
                else:
                    self._data[key] = values[:0]

    def __getitem__(self, key: str) -> ColumnData:
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

    def __setitem__(self, key: str, value: ColumnData) -> None:
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

    def compress(
        self,
        flags: Sequence[bool] | np.ndarray,
        inplace: bool = False,
    ) -> DataFrameLight:
        """
        NOTE: NOT A PART OF THE PANDAS API
        """
        subset = self if inplace else self.__class__()
        for key in self._data.keys():
            subset._data[key] = list(ub.compress(self._data[key], flags))
        return subset

    def take(
        self,
        indices: slice | Sequence[int] | np.ndarray,
        inplace: bool = False,
    ) -> DataFrameLight:
        """
        Return the elements in the given *positional* indices along an axis.

        Args:
            inplace (bool): NOT PART OF PANDAS API

        Note:
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
            for key, values in self._data.items():
                if isinstance(values, np.ndarray):
                    subset._data[key] = values[indices]
                else:
                    subset._data[key] = [values[idx] for idx in indices]
        return subset

    def copy(self) -> DataFrameLight:
        other = copy.copy(self)
        other._data = other._data.copy()
        other._localizer = LocLight(other)
        return other

    def append(self, row: Mapping[str, object]) -> None:
        """
        Append a row to this data frame.

        Args:
            other (dict[str, Any]):
                A value for each column to add the end of this object.

        Note:
            Not part of the pandas API.

        Example:
            >>> from kwarray.dataframe_light import *  # NOQA
            >>> self = DataFrameLight(columns=['foo', 'bar'])
            >>> other = {'foo': 0, 'bar': 1}
            >>> self.append(other)
            >>> self.append(other)
            >>> assert len(self) == 2
        """
        _data = self._data
        # Check if the row contains new columns
        new_keys = row.keys() - _data.keys()
        if new_keys:
            num_rows = len(self)
            if num_rows == 0:
                # If this is the first row just initialize the columns
                for key in new_keys:
                    _data[key] = []
            else:
                # If so, backfill the previous rows for the new column
                for key in new_keys:
                    _data[key] = [None] * num_rows

        for k, vals in _data.items():
            new_value = row.get(k, None)
            if isinstance(vals, list):
                vals.append(new_value)
            else:
                _data[k] = np.append(vals, cast(Any, new_value))

    def extend(
        self,
        other: DataFrameLight | Mapping[str, Sequence[object] | np.ndarray],
    ) -> None:
        """
        Extend ``self`` inplace using another dataframe array

        Args:
            other (DataFrameLight | dict[str, Sequence]):
                values to concat to end of this object

        Note:
            Not part of the pandas API.

        Example:
            >>> self = DataFrameLight(columns=['foo', 'bar'])
            >>> other = {'foo': [0], 'bar': [1]}
            >>> self.extend(other)
            >>> assert len(self) == 1
        """
        _other_data = (
            other._data if isinstance(other, DataFrameLight) else other
        )

        _self_data = self._data
        for key, vals1 in _self_data.items():
            vals2 = _other_data[key]
            if isinstance(vals1, list):
                vals1.extend(vals2)
            else:
                _self_data[key] = np.hstack([vals1, np.asarray(vals2)])

    def union(
        self,
        *others: DataFrameLight | Mapping[str, Sequence[object] | np.ndarray],
    ) -> DataFrameLight:
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
    def concat(
        cls,
        others: Sequence[
            DataFrameLight | Mapping[str, Sequence[object] | np.ndarray]
        ],
    ) -> DataFrameLight:
        return cast(Any, cls).union(*others)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> DataFrameLight:
        _raw = {k: v.values for k, v in df.to_dict(orient='series').items()}
        return cls(_raw)

    @classmethod
    def from_dict(
        cls,
        records: Sequence[Mapping[str, object]],
    ) -> DataFrameLight:
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

    def reset_index(self, drop: bool = False) -> DataFrameLight:
        """noop for compatibility, the light version doesnt store an index"""
        return self

    def groupby(
        self,
        by: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> Iterator[tuple[object, DataFrameLight]] | object:
        """
        Group rows by the value of a column. Unlike pandas this simply
        returns a zip object. To ensure compatibility call list on the
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
            >>> timerit.Timerit(100).call(lambda: dict(list(self.pandas().groupby('cx')))).print()
            >>> timerit.Timerit(100).call(lambda: dict(self.groupby('cx'))).print()
        """
        if by is not None and len(args) == 0 and len(kwargs) == 0:
            # In this special case we can be fast
            import kwarray

            unique, groupxs = kwarray.group_indices(self[by])
            groups = [self.take(idxs) for idxs in groupxs]
            return zip(unique, groups)
        else:
            # otherwise we need to use the slow method
            return self.pandas().groupby(by=by)

    def rename(
        self,
        mapper: Mapping[str, str] | None = None,
        columns: Mapping[str, str] | None = None,
        axis: int | None = None,
        inplace: bool = False,
    ) -> DataFrameLight:
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
                raise TypeError(
                    "Cannot specify both 'axis' and any of 'index' or 'columns'"
                )
        else:
            if axis != 1:
                raise NotImplementedError('only axis=1 is supported')
            columns = mapper

        if columns is None:
            raise TypeError('columns or mapper must be specified')
        if not inplace:
            self = self.copy()
        for old, new in columns.items():
            if old in self._data:
                self._data[new] = self._data.pop(old)
        return self

    def iterrows(self) -> Generator[tuple[int, dict[str, object]], None, None]:
        """
        Iterate over rows as (index, Dict) pairs.

        Yields:
            Tuple[int, Dict]: the index and a dictionary representing a row

        Example:
            >>> from kwarray.dataframe_light import *  # NOQA
            >>> self = DataFrameLight._demodata(num=3)
            >>> print(ub.urepr(list(self.iterrows()), sort=1))
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
            >>> ti = timerit.Timerit(21, bestof=3, verbose=2, unit='ms')
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

    _data: dict[str, np.ndarray]

    def __normalize__(self) -> None:
        """
        Try to convert input data to Dict[ndarray]
        """
        if self._raw is None:
            self._data = {}
        elif isinstance(self._raw, dict):
            raw_data: Any = self._raw
            self._data = raw_data
            for key, values in self._data.items():
                if not isinstance(values, np.ndarray):
                    self._data[key] = np.asarray(values)
            if __debug__:
                lens = [len(d) for d in self._data.values()]
                assert ub.allsame(lens), (
                    'lens are not all same {} for columns {}'.format(
                        lens, list(self._data.keys())
                    )
                )
        elif isinstance(self._raw, DataFrameLight):
            self._data = {
                key: np.asarray(values)
                for key, values in self._raw._data.items()
            }
        elif pd is not None and isinstance(self._raw, pd.DataFrame):
            self._data = {
                k: v.values
                for k, v in self._raw.to_dict(orient='series').items()
            }
        else:
            raise TypeError('Unknown _raw type')

    def extend(
        self,
        other: DataFrameLight | Mapping[str, Sequence[object] | np.ndarray],
    ) -> None:
        other_data = other._data if isinstance(other, DataFrameLight) else other
        for key in self._data.keys():
            vals1 = self._data[key]
            vals2 = np.asarray(other_data[key])
            self._data[key] = np.hstack([vals1, vals2])

    def compress(
        self,
        flags: Sequence[bool] | np.ndarray,
        inplace: bool = False,
    ) -> DataFrameArray:
        subset = self if inplace else self.__class__()
        for key in self._data.keys():
            subset._data[key] = self._data[key][flags]
        return subset

    def take(
        self,
        indices: slice | Sequence[int] | np.ndarray,
        inplace: bool = False,
    ) -> DataFrameArray:
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
