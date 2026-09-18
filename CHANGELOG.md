# Changelog

Versions follow [Semantic Versioning](https://semver.org) (`<major>.<minor>.<patch>`).

## Unreleased

### Added

- `report` accepts a `*_perievent.csv` written by `process peri-event` and writes `<stem>_report.html`, an interactive peri-event report: per-region traces as mean ± 95% CI or individual trials, filtered by `sdt_type` and optionally split by trial type, a clickable Allen CCF top view for choosing the region, and an all-regions grid. When the `*_metrics.csv` and `*_metrics-session.csv` from `process metrics` are stored together, the report also marks the onset, peak and offset of the mean trace with the per-trial mean ± SD or median and IQR of each time alongside (per-trial markers in the individual-trials view), plots per-trial metrics for the selected region, colours the atlas by a session metric and lists the session table. `-t/--trials` supplies `sdt_type` for peri-event files written without it ([#162](https://github.com/DuguidLab/mesoscopy/issues/162)).
- [Reports](https://docs.mesoscopy.org/how-to/reports/) how-to page covering the preprocessing, registration and peri-event reports ([#162](https://github.com/DuguidLab/mesoscopy/issues/162)).
- `report.perievent` module with `report_payload`, `perievent_cube`, `trial_types`, `atlas_outlines`, `metrics_paths`, `pack_float32` and `unpack_float32` ([#162](https://github.com/DuguidLab/mesoscopy/issues/162)).

## [0.13.1] - 2026-09-18

### Added

- `process peri-event` CSV output now carries the trials CSV columns (`sdt_type`, `outcome`, `response_time`, ...) on every row, joined by `trial_index` after `F`. `process metrics` and `metrics.metrics_tables` carry any per-trial columns of the peri-event table through to `<stem>_metrics.csv`, so `-t/--trials` is only needed for peri-event files written without them and skips columns the peri-event table already carries ([#158](https://github.com/DuguidLab/mesoscopy/issues/158)).

### Fixed

- `process peri-event --with-metrics` crashed with `IndexError` when no trials were kept, after writing a header-only `_perievent.csv`, and `process metrics` crashed the same way on that file. The peri-event command now skips the metrics step with a message when no trials are kept, `process metrics` reports the empty input, and `metrics.metrics_tables` raises `ValueError` on a table with no rows ([#156](https://github.com/DuguidLab/mesoscopy/issues/156)).

## [0.13.0] - 2026-09-18

### Added

- `process metrics` command for extracting per-trial response metrics from the `*_perievent.csv` written by `process peri-event`. Traces are baseline-subtracted per trial (`--baseline START END`, default all pre-event samples), then onset time (`--onset sd`, a `--onset-sd` multiple of the baseline SD, or `--onset peak`, an `--onset-fraction` of the amplitude, held for `--onset-min-samples` samples; or `--onset extrapolate`, a line fitted to the rise between `--extrapolate-range` fractions of the amplitude and extrapolated back to baseline), peak time, signed amplitude, signed area under the curve, decay time (from the peak to `--decay-fraction` of the amplitude), offset time (the return to the onset threshold after the peak) and duration are taken over the response window (`--response START END`, default all post-event samples). `--smooth N` detects the peak, onset, decay and offset on an N-sample moving average. `process peri-event --with-metrics` writes the same tables in one step, taking the same options and joining the trials file. Writes `<stem>_metrics.csv` per trial per region, optionally joined with a `-t/--trials` CSV, and `<stem>_metrics-session.csv` per region with the across-trial mean, SD and CV of each metric plus the mean pairwise trial-trace correlation ([#152](https://github.com/DuguidLab/mesoscopy/issues/152)).
- `process.metrics` module with `baseline_stats`, `smooth`, `peak`, `auc`, `onset_time`, `extrapolated_onset`, `offset_time`, `decay_time`, `trace_correlation`, `trial_metrics`, `session_metrics`, `metrics_tables` and `join_trials`, and a [Response metrics](https://docs.mesoscopy.org/how-to/metrics/) how-to page ([#152](https://github.com/DuguidLab/mesoscopy/issues/152)).

### Fixed

- `process peri-event` grid times carried float residue, e.g. `8.9e-16` at the event and `3.0000000000000036` at `--post 3`, so windows bounded at those times could miss the end samples. `perievent.window_grid` now builds each time as `(k - pre * fs) / fs`, which puts the event on exactly `0.0` and every grid time on its nearest double ([#152](https://github.com/DuguidLab/mesoscopy/issues/152)).

### Performance

- The root CLI imports a subcommand's module only when that subcommand is invoked, and each stage's command module imports its heavy dependencies (pynwb, dask, zarr, pandas, scipy, scikit-learn, pims, napari, plotly) inside the command that uses them. `mesoscopy --help` and every `<stage> --help` now start in about 0.1 s instead of 1-2 s, and HDF5-only commands no longer load pynwb ([#149](https://github.com/DuguidLab/mesoscopy/issues/149)).

## [0.12.0] - 2026-09-14

### Added

- `process peri-event` command for extracting per-trial windows around a behavioural event (`--event cue_onset|trial_start|response|reward`) from a behaviour-aligned HDF5 recording or `_regions.csv`, using the `*_trials.csv` written by `visiomode-analysis session` (or equivalent behaviour CSV). Windows are resampled onto a `--pre`/`--post`/`--fs` grid by interpolation or nearest sample, optionally baseline-subtracted with `--baseline START END`, and trials lacking the event or running past the recording are dropped. HDF5 input gives `<recording stem>_event-<name>_perievent.h5` with `/traces`, `/time`, `/trial_index` and `/event_time`; CSV input gives a long-format CSV with `trial_index`, `event_time`, `time`, `region` and `F` columns ([#133](https://github.com/DuguidLab/mesoscopy/issues/133)).
- `process.perievent` module with `event_times`, `window_grid`, `extract` and `apply_baseline` ([#133](https://github.com/DuguidLab/mesoscopy/issues/133)).
- `register landmarks` registers to the Allen CCF template at any scale. `-s/--scale` sets the template scale relative to its native 140x142 pixels, and defaults to the scale at which template pixels match the recording's, so recordings keep their own resolution. The template name, shape and scale are written as attributes of the registered HDF5 file and noted in the NWB `CCFRegisteredSeries` comments ([#90](https://github.com/DuguidLab/mesoscopy/issues/90)).
- `resources.get_atlas` and `resources.get_default_landmarks` take a target shape and resample the atlas (nearest neighbour) and scale the landmarks to it, with `resources.template_shape`, `resources.template_scale`, `resources.scale_landmarks` and `resources.resize_labels` helpers; `register.transform.auto_template_scale` picks the scale matching a recording's pixel size ([#90](https://github.com/DuguidLab/mesoscopy/issues/90)).

### Changed

- `process regions` resamples the atlas to the frame shape of the registered recording instead of requiring 140x142 frames ([#90](https://github.com/DuguidLab/mesoscopy/issues/90)).

### Fixed

- `register label` and the `inspect` viewers failed to launch with `AttributeError: No napari attribute view_image` on napari >= 0.6, which removed `napari.view_image`. The viewers are now created with `napari.Viewer().add_image`. The napari lower bound is now `>=0.6`, and Dependabot keeps `uv.lock` current so CI runs on new upstream releases.
- `preprocess --no-qa` crashed when writing the preprocessed file, as the skipped QA results were passed to the HDF5 writer as `None`.

### Deprecated

- `--output-width`/`--output-height` on `register landmarks`, in favour of `--scale`. They now scale the template to the requested frame size, with a missing dimension filled in at the atlas aspect ratio ([#90](https://github.com/DuguidLab/mesoscopy/issues/90)).

## [0.11.0] - 2026-09-14

### Added

- `align` command for writing behaviour-aligned frame timestamps to a preprocessed or registered HDF5 recording, as seconds from a behaviour session start given either as an ISO 8601 `--session-start` or read from a visiomode `--behaviour-json` file. Writes a `/timestamps_aligned` dataset carrying `session_start_time`, `behaviour_session` and `offset_s` attributes, and refuses offsets beyond `--max-offset` ([#128](https://github.com/DuguidLab/mesoscopy/issues/128)).
- `align.align_recording` and `align.read_session_start`, plus `io.read_timestamps_aligned` and `io.write_timestamps_aligned` ([#128](https://github.com/DuguidLab/mesoscopy/issues/128)).
- `process regression` checks the recording's `/timestamps_aligned` against the regressor file's `session_start_time` and `timestamps`: it warns and applies `trial_idx` positionally if either side is unaligned, fails on a session start or length mismatch, warns if aligned timestamps differ by more than 5 ms, and copies `session_start_time` and `behaviour_session` into the regression output ([#128](https://github.com/DuguidLab/mesoscopy/issues/128)).
- `process smooth` and `process zscore` copy `/timestamps_aligned` and its attributes from the input recording to the output file when present ([#128](https://github.com/DuguidLab/mesoscopy/issues/128)).
- `process regions` adds a `time_aligned` column (seconds from behaviour session start) after `timestamp` when the recording has `/timestamps_aligned` ([#128](https://github.com/DuguidLab/mesoscopy/issues/128)).

### Changed

- `io.read_regressors` returns a fourth element, the regressor file's behaviour alignment (`session_start_time`, `behaviour_session`, `timestamps`) or `None` ([#128](https://github.com/DuguidLab/mesoscopy/issues/128)).

### Fixed

- Incorrect `--help` text for `-o`/`--out_dir` on `process zscore`, `process regions`, `report` and `register label`, and for `-s`/`--sigma` on `process smooth`, which all described the wrong output.
- Typical workflow docs referred to the nonexistent `convert` and `process area-responses` commands instead of `convert h5` and `process regions`.
- `process smooth` ignored `-s`/`--sigma` and always smoothed with the default kernel width.

## [0.10.0] - 2026-09-12

### Added

- `-m`/`--mask` option to `process regions` for extracting ΔF/F activity from custom region masks (NPY, NPZ or TIFF) drawn in registered frame coordinates; boolean masks give one region, integer-labelled masks one region per label, and masks are used as drawn rather than mirrored across hemispheres. Supplying a mask replaces the ABA extraction unless `--include-aba` is given, and may be passed multiple times ([#108](https://github.com/DuguidLab/mesoscopy/issues/108)).
- `extract_mask_activity` and `extract_all_masks` in `process.region` for extracting mean ΔF/F from custom masks, ignoring NaN pixels, plus `io.read_mask` for loading masks from NPY, NPZ or TIFF files ([#108](https://github.com/DuguidLab/mesoscopy/issues/108)).

### Fixed

- ABA region extraction in `process.region` was not NaN-aware. In `extract_all_regions` a single NaN pixel nulled every region's trace for that frame, including regions not containing it, because the masks are applied by matmul and `NaN * 0` is `NaN`. `extract_region_activity` returned a fully masked array if any pixel in the region was NaN, and returned a `MaskedArray` rather than the `ndarray` it is annotated to return. Both now ignore NaN pixels, matching `extract_mask_activity`, and a region is only NaN in a frame if it has no non-NaN pixel there ([#115](https://github.com/DuguidLab/mesoscopy/issues/115)).
- `extract_region_activity` raised an opaque `IndexError` for an unrecognised region acronym instead of the documented `ValueError`. The error now names the acronym and suggests close matches ([#117](https://github.com/DuguidLab/mesoscopy/issues/117)).
- `ridge_regression_fast` emitted spurious BLAS floating-point warnings on every call ([#120](https://github.com/DuguidLab/mesoscopy/issues/120)).

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
