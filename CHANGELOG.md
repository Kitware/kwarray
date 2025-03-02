# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.


## Version 0.7.1 - Unreleased

### Changed
* input to `ensure_rng` can now be a string.


## Version 0.7.0 - Released 2024-08-14

### Removed
* Removed support for Python 3.6 and 3.7

### Fixed
* Now correctly respect the shape arguments in `kwarray.distributions.Mixture.sample`
* Binomial was previously broken and now returns correct values from sample.


## Version 0.6.19 - Released 2024-06-19

### Added

* Support for numpy 2.0 on Python 3.9+


## Version 0.6.18 - Released 2024-03-19

### Added

* Add memmap support to ``Stitcher`` in numpy mode.


## Version 0.6.17 - Released 2024-02-01

### Added

* Add `allclose` and `isclose` to ArrayAPI.


## Version 0.6.16 - Released 2023-11-17


## Version 0.6.15 - Released 2023-10-13


## Version 0.6.14 - Released 2023-07-14

### Changed
* Make scipy optional

### Fixed
* Issue in `_combine_mean_stds` when nums is less than 1 and bessel correction is on.
* Added `__params__` to Categorical
* Fixed case where `rng` passed to `Normal.random` was not respected.
* Ensure `kwarray.group_indicies` has consistent output across numpy versions


## Version 0.6.13 - Released 2023-05-31

### Fixed
* Bug in robust normalizers with quantile-clip extrema


## Version 0.6.12 - Released 2023-04-28

### Fixed:
* Problems with last release


## Version 0.6.11 - Released 2023-04-28


### Changed
* Avoiding global torch imports. This may slow down the ArrayAPI, and we should
  revisit a refactor after removing 3.6 support.

## Version 0.6.10 - Released 2023-04-04

### Added
* Added `nan_policy` argument to `kwarray.Sticher`
* Added `array_equal` to ArrayAPI

### Changed
* Allow `None` as `indices` in `kwarray.Sticher.add`
* Change `nan_behavior` to `nan_policy` in `kwarray.RunningStats`
* Updated `ub.repr2` to `ub.urepr`.

### Fixed
* `robust_normalize` now preserves masked arrays.


## Version 0.6.9 - Released 2023-02-02

### Changed
* Enhanced the values accepted by `robust_normalize` params, added strict quantile method.

### Fixed
* Robust normalize did not respect custom params when normalizing across each axis
* Robust normalize did not respect min/max value


## Version 0.6.8 - Released 2023-01-11

### Fixed:
* Removed extraneous print statement in `robust_normalize`

## Version 0.6.7 - Released 2023-01-02

### Changed:
* Nothing, this was an accidental release.

## Version 0.6.6 - Released 2023-01-02

### Fixed:
* Bug in `isect_flags`, which returned a bad shape if there was no intersection.
* Fix issue in normalize when empty arrays are given

### Changed:
* Updated ported robust normalizer code from kwimage
* Move normalize from `util_numpy` to `util_robust`


## Version 0.6.5 - Released 2022-11-07

### Changed
* Updated requirements for Python 3.11


## Version 0.6.4 - Released 2022-09-28

### Changed
* `kwarray.embed_slice` now does error checking
* Add `kwarray.RunningStats.update_many` to update with multiple observations at once.
* Added missing `ArrayAPI.min`, `ArrayAPI.min_argmin` funcs.

### Fixed
* Fix ArrayAPI.max for torch in the case where not both items are ints
* Fixed various ArrayAPI cases, added tests.


## Version 0.6.3 - Released 2022-07-31

### Added
* `equal_with_nan` in `util_numpy`

### Fixed:
* Fixed issue in RunningStats where computation was incorrect when weights were given.
* RunningStats now handles nans.

## Version 0.6.2 - Released 2022-06-15

### Added
* Type stubs


## Version 0.6.1 - Released 2022-06-13

### Added
* Add quantile to `kwarray.stats_dict`


### Fixed
* Corner cases in `kwarray.stats_dict`
* Issue in `group_indices` when given a special view array
* Fixed numpy warnings
* Fixed usage of `torch.complex32`


## Version 0.6.0 - Released 2022-03-04

### Added
* `util_robust` for robust normalization.
* Add `return_index` to `unique_rows`

### Fixed
* Fixes to normalize
* Remove support for Python < 3.6


## Version 0.5.21 - Released 2021-11-05

### Fixed
* Release issues


## Version 0.5.20 - Released 2021-11-05

### Added
* Added weights argument to RunningStats
* argminima now supports axis
* Add `unique_rows`


## Version 0.5.19 - Released 2021-05-13

### Changed
* `SlidingWindow` no longer returns slices that start at negative indexes.
* `SlidingWindow` can now handle None values in the window.


## Version 0.5.18 - Released 2021-05-10


## Version 0.5.17 - Released 2021-05-05

### Added
* Renamed `cast` to `coerce` in distributions.


## Version 0.5.16 - Released 2021-04-26

### Added
* Add `util_slices` with `padded_slice` and `embed_slice`


## Version 0.5.15 - Released 2021-04-25

### Added
* Add `util_slider`.


## Version 0.5.14 - Released 2021-04-22

### Added
* Add `argsort` to ArrayAPI.
* Add `dtype_info`.

### Changed
* Fixed warnings by changing np.int and np.float to int and float


## Version 0.5.13 - Released 2021-01-08

### Added:
* Add `kwarray.normalize` (moved from kwimage)

### Fixed
* Fixed issue in `one_hot_lookup` with ONNX


## Version 0.5.12 - Released 2020-11-27


## Version 0.5.11 - Released 2020-11-20


## Version 0.5.9 - Released 2020-10-27

### Added:
* `DataFrameLight.to_dict`

### Changed
* Torch is now an optional dependency
* Pandas is now an optional dependency

## Version 0.5.8 - 2020-04-15

### Changed
* Fixed publish issues with previous versions

## Version 0.5.7 - 2020-04-14

### Added
* `ArrayAPI.round` now accepts decimals keyword arg
* `algo_setcover`

## Version 0.5.6 - 2020-04-08

### Fixed
* `stats_dict` median now respects axis kwarg

## Version 0.5.5

### Added
* `DataFrameLight.from_pandas`
* `DataFrameLight.iterrows`
* `DataFrameLight.pandas` in favor of `_pandas`

## Version 0.5.4 - Released 2020-02-19 


### Added
* Add `FlatIndexer`

### Changed
* Added better docs to `kwarray.util_groups`.

### Fixed
* `ArrayAPI.numpy` and other "api-methods" now correctly raise a TypeError on
  unknown data types. Previous behavior returned None.


## [Version 0.5.2] - Released 2020-10-27

### Fixed 
* `ensure_rng` now correctly coerce the `random` and `np.random` modules.
* `group_indices` now works correctly on lists of tuples.


## [Version 0.4.0] - Released 2019 - Nov

### Added
* dev folder with developer benchmarks and scripts
* sanitize script for public release

### Changed
* First public release
* Refactored requirements into tests and runtime
* `ensure_rng` can now accept floats as a seed
* Improved speed of `random_product`.

### Fixed
* Fix bug in one-hot-encoding when dim=-1
* `run_developer_setup.sh` uses `setup.py develop` to avoid issues with PEP 517.
* Support for torch `1.2.0`


## Version 0.3.0

### Fixed
* Fixed bug in `ArrayAPI.compress` with multi-dimensional tensors
* Fixed bug in `ArrayAPI.ones_like`, where it returned zeros

### Added
* Add `clear` to `DataFrameLight`
* Added `util_torch` with `one_hot_embedding`
* Add `ArrayAPI.matmul` 
* Add various `ArrayAPI` functions

### Changed
* Speed improvements
* `boolmask` now automatically converts data to an integer ndarray 


## Version 0.2.0

### Added
* Initial port of functionality from KWIL
    - `fast_rand.py`
    - `util_groups.py`
    - `util_numpy.py`
    - `util_random.py`
    - `util_torch.py`
    - `util_averages.py`
    - `distributions.py`
    - `dataframe_light.py`
    - `arrayapi.py`
    - `algo_assignment.py`

## Version 0.5.10 - Released 2020-10-27
