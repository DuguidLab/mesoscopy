# Changelog

Versions follow [Semantic Versioning](https://semver.org) (`<major>.<minor>.<patch>`).

## [Unreleased]

### Added

- Registration GUI with Napari for identifying anatomical landmarks.

### Changed

- Dropped python 3.8 support, now requires 3.12 or above.
- Preprocessing and registration accept NWB file as input and update at end of processing. Stand-alone HDF5 files are still supported.
- Major API refactoring, to break up large `__init__` files and create a more consistent interface.
- Non-command entries to `__init__` files are now marked as private.
- `preprocess` command no longer requires output directory to be specified; this is now an optional flag which defaults to the user's current directory.

### Fixed

- Registration uses DeltaF series instead of raw fluorescence series when reading from NWB.

### Removed

- Removed `landmarks` subcommand; the parent `register` command now does the same thing.
- Removed average image generation from `register` command (previously under now removed `utils` module).
- Removed processing `aba` module, which used to extract average dF/F traces per ABA area and write output as CSV.

## [0.1.0] - 2023-03-24

Line-in-the-sand release, with all the imaging processing code I used for my thesis.

Not necessarily fit for public consumption, but here it is anyway.
