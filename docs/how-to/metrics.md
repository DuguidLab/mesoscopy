# Response metrics

`mesoscopy process metrics` turns the per-trial windows written by `process peri-event` into a table of response
metrics per trial per region, and a per-session summary of how those metrics vary across trials.

## Input

A long-format `_perievent.csv`, as written by `process peri-event` from a `_regions.csv`:

```bash
mesoscopy process regions /path/to/recording_smoothed.h5
mesoscopy process peri-event /path/to/recording_smoothed_regions.csv /path/to/recording_trials.csv --event cue_onset --pre 1 --post 3
mesoscopy process metrics /path/to/recording_smoothed_regions_event-cueonset_perievent.csv
```

Or in one step:

```bash
mesoscopy process peri-event /path/to/recording_smoothed_regions.csv /path/to/recording_trials.csv --with-metrics
```

`--with-metrics` accepts every option of `process metrics`. The peri-event `--baseline`, when given, is also the
metrics baseline window. Metrics are not computed for HDF5 peri-event windows.

## Output

`<stem>_metrics.csv` has one row per trial per region:

| Column | Meaning |
| --- | --- |
| `trial_index`, `event_time`, `region` | Carried over from the peri-event file. |
| `baseline_mean`, `baseline_sd` | Mean and SD of the raw trace over the baseline window. |
| `onset_time` | Time of the response onset, seconds from the event. |
| `peak_time`, `amplitude` | Time and height of the signed maximum over the response window. |
| `auc` | Signed trapezoidal area under the baseline-subtracted trace over the response window. |
| `decay_time` | Time from the peak until the trace falls to a fraction of the amplitude. |
| `offset_time` | Time at which the response returns to the onset threshold after the peak. |
| `duration` | `offset_time` minus `onset_time`. |

The trials table columns (`sdt_type`, `outcome`, `response_time`, ...) follow, carried over from the peri-event file.
For a peri-event file without them, `--trials` joins a trials CSV by row index.

`<stem>_metrics-session.csv` has one row per region with `n_trials`, `trace_correlation` (the mean pairwise
Pearson correlation between trial traces over the response window) and, for each metric above, `<metric>_mean`,
`<metric>_sd` and `<metric>_cv` across trials, ignoring trials where the metric is undefined. `onset_n`, `decay_n`
and `offset_n` count the trials where each was found.

Any metric is empty when it is undefined for that trial, for example a decay when the trace never falls back to
the fraction, or an onset when the trace never crosses the threshold.

## Metrics

Every trace is first baseline-subtracted with its own mean over the baseline window, `--baseline START END`
(default: all pre-event samples, end exclusive). Metrics are then taken over the response window,
`--response START END` (default: all post-event samples, end inclusive).

**Onset** is found in one of three ways, chosen with `--onset`:

- `sd` (default): the first `--onset-min-samples` consecutive samples above `--onset-sd` times the baseline SD.
  Simple and per trial, but sensitive to how well 25 or so baseline samples estimate the noise.
- `peak`: the same, with the threshold at `--onset-fraction` of the peak amplitude. Independent of baseline noise,
  but inherits whatever the peak does.
- `extrapolate`: a line is fitted to the last rise before the peak, between `--extrapolate-range LOW HIGH`
  fractions of the amplitude, and the onset is where that line crosses baseline. The standard latency estimate
  in electrophysiology; robust on clean responses, but on noisy traces the fitted rise can be short and the onset
  can fall before the event.

**Offset** is the first `--onset-min-samples` consecutive samples at or below the onset threshold after the
peak, or below `LOW` of the amplitude for `--onset extrapolate`. It needs the peak to be above the threshold and
the trace to come back down within the response window.

**Decay** is the time from the peak until the trace first falls to `--decay-fraction` of the amplitude. It needs
a positive amplitude.

**Smoothing.** On noisy traces the peak can land on a single-sample spike, which shortens the decay and shifts
the offset. `--smooth N` finds the peak, onset, decay and offset on an `N`-sample centred moving average (`N`
odd). The baseline SD and the AUC are always taken from the raw trace, so smoothing does not shrink the noise
estimate that the `sd` onset threshold depends on.

## Choosing settings

- Z-scored recordings make the `sd` onset threshold directly comparable across regions.
- If onsets look early, raise `--onset-sd` or `--onset-min-samples`; if many trials have no onset, lower them.
- If decays look implausibly short, add `--smooth 5` and check the peak times.
- Session CV is unstable for any metric whose mean sits near zero, which the extrapolated onset often does.
  Read the SD in that case.

## As a library

```python
import pandas as pd
from mesoscopy.process import metrics

perievent = pd.read_csv("recording_smoothed_regions_event-cueonset_perievent.csv")
per_trial, per_session = metrics.metrics_tables(perievent, smoothing=5)
```

`metrics.trial_metrics` works on a `(n_trials, n_samples)` array for one region, and the per-metric functions
(`peak`, `auc`, `onset_time`, `extrapolated_onset`, `offset_time`, `decay_time`, `trace_correlation`) are
available on their own.
