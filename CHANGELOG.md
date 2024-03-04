# Changelog

Versions follow [Semantic Versioning](https://semver.org) (`<major>.<minor>.<patch>`).

## [Unreleased]

### Changed

- Preprocessing and registration accept NWB file as input and update at end of processing. Stand-alone HDF5 files are still supported.

## [0.1.0] - 2023-03-24

Line-in-the-sand release, with all the imaging processing code I used for my thesis.

Not necessarily fit for public consumption, but here it is anyway.

### Added

- This changelog.
- Add hatch build tool.
- Preprocessing pipeline to extract delta F and perform haemodynamic correction.
- Registration to the Allen Mouse Brain Atlas.
- Extract average activity per ABA region.

[unreleased]: https://github.com/DuguidLab/mesoscopy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DuguidLab/mesoscopy/releases/tag/v0.1.0
