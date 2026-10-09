# Typical workflow

## Convert recording file to NWB format

```bash
mesoscopy convert h5 /path/to/example-recording.h5
```

!!! note
    See also our guide to [converting video files to HDF5](how-to/convert-video-to-h5.md) if you record videos in AVI or MP4 format.

## Inspect raw data

```bash
mesoscopy inspect /path/to/example-recording.nwb
```

## Preprocess to correct for haemodynamics and extract ∆F signal

```bash
mesoscopy preprocess /path/to/example-recording.nwb
```

## Register to the Allen Brain Atlas

First mark the anatomical landmarks on the recording. This opens the napari landmark GUI, seeded with a point per
landmark. Drag each one onto its anatomical location, then press **Save and Close**.

```bash
mesoscopy register label /path/to/example-recording.nwb
```

This writes `example-recording_landmarks.csv` to the output directory (`-o`, the current directory by default),
holding each landmark's `(x, y)` position in the pixel space of the ∆F/F series.

Then warp the recording onto the atlas. The landmarks file is found automatically when it sits next to the
recording or in the output directory. Pass `-r/--recording-points` to point at it yourself.

```bash
mesoscopy register landmarks /path/to/example-recording.nwb
```

The registered frames are written in Allen CCF template space. By default the template is scaled so that its
pixels match the size of the recording's, so a recording keeps its own resolution rather than being resampled to
the atlas's native 140x142 pixels. Pass `-s/--scale` to pick the template scale yourself, where `--scale 1` gives
the native atlas size. `mesoscopy process regions` resamples the atlas to whatever frame size the registration
produced.

!!! note
    `-t/--template-points` supplies the *template* landmarks being registered onto, not your recording's
    landmarks. Leave it unset to use the Allen CCF landmarks that ship with mesoscopy.

Registration reports how far each landmark ends up from its template position, and warns when the fit is poor.

```
Estimating transform from 9 landmarks...
Landmark fit: RMSE 2.28 px, worst is 'rFP' at 3.94 px (in template pixels).
```

To check the alignment by eye, generate the QA report for the registered HDF5 file written by the step above. Its
path is echoed as `Saved registered frames at ...`.

```bash
mesoscopy report /path/to/example-recording_registered.h5
```

## Extract region activity

```bash
mesoscopy process regions /path/to/example-recording.nwb
```

## Extract peri-event window activity

Given a recording aligned with `mesoscopy align` and the `*_trials.csv` from `visiomode-analysis session`, extract
trial-by-trial activity aligned to behaviour.

```bash
mesoscopy process peri-event /path/to/example-recording_regions.csv /path/to/example-recording_trials.csv --event cue_onset --pre 1 --post 3
```

An HDF5 recording such as `_zscored.h5` gives an HDF5 file of `(n_trials, n_samples, height, width)` traces. A
`_regions.csv` gives a long-format CSV with the trials columns (`sdt_type`, `outcome`, ...) joined onto every row.
Add `--baseline START END` to subtract a per-trial baseline.

## Extract response metrics

Given the peri-event CSV from the step above, extract the response metrics of every trial and region.

```bash
mesoscopy process metrics /path/to/example-recording_regions_event-cueonset_perievent.csv
```

This writes `example-recording_regions_event-cueonset_metrics.csv` with the onset time, peak time, amplitude,
area under the curve, decay time, offset time and duration of every trial in every region, with the trials
columns carried over from the peri-event file. `example-recording_regions_event-cueonset_metrics-session.csv`
holds the mean, SD and CV of each metric across trials and the mean pairwise correlation between trial traces, per
region and trial group. Go/no-go sessions are grouped by trial type, stimulus and lever push as well as all
trials, and their session table also has trial-to-trial reliability metrics over the cue-to-lever epoch. See
[trial groups](how-to/metrics.md#trial-groups) and [reliability](how-to/metrics.md#reliability).
`example-recording_regions_event-cueonset_metrics-boot.csv` holds the same metrics taken from the trial-mean trace
of each region and trial group, with bootstrap intervals, and
`example-recording_regions_event-cueonset_traces-boot.csv` holds the mean traces. See
[mean-trace metrics](how-to/metrics.md#mean-trace-metrics).

Traces are baseline-subtracted with the mean over `--baseline START END` (default all pre-event samples), and
metrics are taken over `--response START END` (default all post-event samples). Onset is the first
`--onset-min-samples` consecutive samples above `--onset-sd` baseline standard deviations, or above
`--onset-fraction` of the peak amplitude with `--onset peak`. `--onset extrapolate` instead fits a line to the
rise between `--extrapolate-range LOW HIGH` fractions of the amplitude and takes where it crosses baseline. Offset
is the first `--onset-min-samples` consecutive samples back at or below the onset threshold after the peak, or
below `LOW` of the amplitude for `--onset extrapolate`, and duration is offset minus onset. Decay is the time from
the peak until the trace falls to `--decay-fraction` of the amplitude. Each is empty when it never happens.

Noisy traces can put the peak on a single-sample spike, which shortens the decay and shifts the offset. `--smooth N`
detects the peak, onset, decay and offset on an `N`-sample moving average while keeping the raw trace for the
baseline SD and the area under the curve.

`process peri-event ... --with-metrics` writes the same tables in one step and takes the same options. See
[Response metrics](how-to/metrics.md) for the definitions.

## Measure connectivity between regions

Given the same peri-event CSV, measure how pairs of regions covary while the animal responds to the cue.

```bash
mesoscopy process connectivity /path/to/example-recording_regions_event-cueonset_perievent.csv
```

This writes `example-recording_regions_event-cueonset_connectivity.csv` with one row per pair of regions and trial
group. See [Connectivity](how-to/connectivity.md) for the metrics.

## Decode the stimulus from region activity

Given the same peri-event CSV, test whether each region, or all regions together, can tell which stimulus the animal
saw on a trial.

```bash
mesoscopy process decode /path/to/example-recording_regions_event-cueonset_perievent.csv
```

This writes `example-recording_regions_event-cueonset_label-stim_decoding.csv` with one row per decoder, region and
trial group, and the population weights next to it. `--label response` decodes the lever push instead. See
[Decoding](how-to/decoding.md) for the scores.

## Next steps
