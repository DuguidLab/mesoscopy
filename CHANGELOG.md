# Changelog

Versions follow [Semantic Versioning](https://semver.org) (`<major>.<minor>.<patch>`).

## Unreleased

## [0.7.4]

### Added

- Add automated QA check results to preprocessing reports ([[#80](https://github.com/DuguidLab/mesoscopy/issues/80)])
- Add timedelta series to preprocessing report.

### Fixed

- Pin pynwb version to pre-3.0, which currently breaks installs.

## [0.7.3]

### Added

- Export ∆F/F as movie file via `export deltaf` ([#66](https://github.com/DuguidLab/mesoscopy/issues/66)).
- Add automated QA checks to preprocessing ([[#80](https://github.com/DuguidLab/mesoscopy/issues/80)]).

### Fixed

- Various preprocessing data type issues when switching between nwb & HDF5 files.

## [0.7.2]

### Fixed

- Preprocessing f0 padding at start end now pads the whole image instead of single value

## [0.7.1]

### Added

- Spatial smoothing (Laplacian of Gaussian) to `process` module via `process smooth`.
- Per-pixel z-scoring for ∆F/F signal to `process` via `process zscore`.

## [0.7.0] - 2025-08-15

### Added

- Registration GUI with Napari for identifying anatomical landmarks.
- Conversion CLI command to create NWB files from raw HDF5 and video recordings (via `convert h5` and `convert video`, respectively).
- Mesoscopy NWB file inspection with the new `inspect` command.
- HTML reports for preprocessing and registration steps with new `report` command.

### Changed

- Dropped python 3.8 support, now requires 3.12 or above.
- Preprocessing and registration accept NWB file as input and update at end of processing. Stand-alone HDF5 files are still supported.
- Major API refactoring, to break up large `__init__` files and create a more consistent interface.
- Non-command entries to `__init__` files are now marked as private.
- `preprocess` command no longer requires output directory to be specified; this is now an optional flag which defaults to the user's current directory.
- `register` command now contains a `label` subcommand, which launches the landmark annotation GUI. The `landmarks` subcommand now accepts registration points in CSV format (exported from the landmark registration GUI) in addition to FIJI XML points.
- Removed inline QA plots, save QA metrics to output HDF5 files for later viewing.

### Fixed

- Registration uses DeltaF series instead of raw fluorescence series when reading from NWB.
- Fix padding insertion error when calculating dF/F, remove redundant padding to cumsum vector.

### Removed

- Removed average image generation from `register` command (previously under now removed `utils` module).
- Removed processing `aba` module, which used to extract average dF/F traces per ABA area and write output as CSV.

## [0.1.0] - 2023-03-24

Line-in-the-sand release, with all the imaging processing code I used for my thesis.

Not necessarily fit for public consumption, but here it is anyway.

---

[0.1.0]: https://github.com/DuguidLab/mesoscopy/compare/v0.1.0...v0.7.0
