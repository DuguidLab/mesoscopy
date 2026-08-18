# Changelog

Versions follow [Semantic Versioning](https://semver.org) (`<major>.<minor>.<patch>`).

## Unreleased

## [0.9.1] - 2026-08-18

### Added

- `-n`/`--nuisance-regressors` option to `process regression` for appending external nuisance regressors (e.g. behavioural motion energy) from an NPZ or HDF5 file, interpolated onto the recording's own timestamps and z-scored before being added to the regressor matrix; may be passed multiple times ([#113](https://github.com/DuguidLab/mesoscopy/pull/113)).

## [0.9.0] - 2026-08-13

### Changed

- Build/dev tooling migrated from `hatch` to `uv` + `Make`; all `make` recipes now wrap `uv run` and no manual environment activation is needed ([#109](https://github.com/DuguidLab/mesoscopy/pull/109)).
- `register`'s `--crop-x`/`--crop-y` options replaced with `--output-width`/`--output-height`; registered output now consistently matches the ABA template shape (or a given output shape) instead of a crop region ([#110](https://github.com/DuguidLab/mesoscopy/pull/110)).
- Landmark identification GUI reworked to make point identification more explicit and warn when landmarks are missing ([#110](https://github.com/DuguidLab/mesoscopy/pull/110)).

### Fixed

- Registration coordinates read inconsistently as xy vs. yx in places, causing warped/misaligned output ([#110](https://github.com/DuguidLab/mesoscopy/pull/110)).
- Landmark GUI could silently drop a point if it was deleted, corrupting downstream registration.
- ABA scaling issue affecting registered output size.
- Automagic landmark file discovery and max-intensity-projection discovery for NWB files.
- Landmark pair name matching, plus a new QA check for registration fit quality.
- NWB update step storing an incorrect transform value after registration.
- Docs generation.

### Performance

- Registration output array is now preallocated instead of built as a Python list and stacked.

## [0.8.0] - 2026-08-06

### Added

- `regression` subcommand to `process` CLI for pixel-wise ridge regression against a set of regressors, with results savable as `.npz` or `.h5` ([#10](https://github.com/DuguidLab/mesoscopy/issues/10)).
- `timestamps` subcommand to `export` CLI for exporting frame timestamps as a text file ([#100](https://github.com/DuguidLab/mesoscopy/issues/100)).

## [0.7.6] - 2026-07-15

### Fixed

- Registration leftward smear fixed by reading registration points as xy instead of yx ([#97](https://github.com/DuguidLab/mesoscopy/issues/97)).

## [0.7.5] - 2026-07-15

### Added

- `regions` subcommand to `process` CLI for extracting ΔF/F activity per ABA region ([#28](https://github.com/DuguidLab/mesoscopy/issues/28)).
- Option to return activity across all regions as a dataframe from `extract_all_regions`.

### Changed

- Consolidated `load_deltaf` into `io.py`, eagerly loading the ΔF/F series into RAM for faster downstream access ([#94](https://github.com/DuguidLab/mesoscopy/pull/94)).
- Improved README with installation instructions, usage, upgrade, contributing, and funders sections ([#22](https://github.com/DuguidLab/mesoscopy/issues/22)).

### Fixed

- Warp transformation now applies the affine transform without inversion, fixing incorrect registration output ([#91](https://github.com/DuguidLab/mesoscopy/issues/91)).

### Performance

- `extract_all_regions` sped up with boolean mask pre-calculation.
- `landmarks_affine` parallelised with `ThreadPoolExecutor`.

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
