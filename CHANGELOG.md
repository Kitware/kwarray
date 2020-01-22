# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.

## [Version 0.5.3] - Unreleased

### Changed
* Added better docs to `kwarray.util_groups`.


## [Version 0.5.2] - Unreleased

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
