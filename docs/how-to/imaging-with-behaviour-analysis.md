# Imaging with behaviour

## Pixel-wise regression

`mesoscopy process regression` fits a ridge regression of behavioural regressors to every pixel of a ∆F/F
recording. Run it on a z-scored recording from `process zscore`.

```bash
mesoscopy process regression /path/to/example-recording_zscored.h5 /path/to/example-recording_regressors.npz
```

The regressor file is an NPZ or HDF5 file holding the `regressors` matrix, their `labels` and the `trial_idx` of the
frames they cover.

Pass `-n/--nuisance-regressors` to add regressors recorded on another clock, such as motion energy from a behaviour
camera. The nuisance file is an NPZ or HDF5 file with a `timestamps` array. Every other array in it becomes one
regressor, resampled onto the recording's timestamps, z-scored and added to the regressor matrix. Repeat `-n` to add
regressors from several files.

The command writes `example-recording_zscored_regression.npz` with the coefficients, R² and mean squared error of
every pixel. Pass `--h5` for an HDF5 file instead.
