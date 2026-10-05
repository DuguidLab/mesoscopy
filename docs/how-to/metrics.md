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

`<stem>_metrics-session.csv` has one row per region and [trial group](#trial-groups), with `region`, `group`,
`n_trials`, `trace_correlation` (the mean pairwise Pearson correlation between trial traces over the response
window) and, for each metric above, `<metric>_mean`, `<metric>_sd` and `<metric>_cv` across trials, ignoring
trials where the metric is undefined. `onset_n`, `decay_n` and `offset_n` count the trials where each was found.

For go/no-go sessions aligned to the cue, trial start or lever push, the session table also has the
[reliability](#reliability) columns.

Any metric is empty when it is undefined for that trial, for example a decay when the trace never falls back to
the fraction, or an onset when the trace never crosses the threshold.

## Trial groups

For go/no-go sessions, the session metrics are taken over all trials and over groups of trials, one row each:

| `group` | Trials |
| --- | --- |
| `all` | Every trial. |
| `sdt-hit`, `sdt-miss`, `sdt-false_alarm`, `sdt-correct_rejection` | One trial type. |
| `stim-go`, `stim-nogo` | By the stimulus shown: hits and misses, or false alarms and correct rejections. |
| `resp-push`, `resp-nopush` | By whether the lever was pushed: hits and false alarms, or misses and correct rejections. |

The groups come from the `sdt_type` trials column, carried over by `process peri-event` or joined with `--trials`.
Without it, or for sessions other than go/no-go, the session table has the `all` rows only, with a warning.

Every group gets a row. A group with fewer than `--min-trials` (default 10) trials keeps its trial counts
(`n_trials`, `onset_n`, `decay_n`, `offset_n` and `reliability_n`) and is otherwise empty. The `all` row's
across-trial summaries are always taken; its reliability columns follow the same minimum.

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

## Reliability

How consistent a region's response is from trial to trial, over the part of each trial before the lever push, so
that movement does not inflate it. Taken for go/no-go sessions only, and needs the `cue_onset`, `response_time` and
`sdt_type` trials columns (carried over by `process peri-event`, or joined with `--trials`). Without them, or for
reward-aligned windows, the columns are skipped with a warning.

**Epoch.** The event the windows are aligned to is read from which trials column matches `event_time`.

- Cue- or trial-start-aligned: the response window, ending at each trial's response. Misses and correct rejections
  end at the median response time of the trials with a response. `--no-mask-response` uses the whole response
  window.
- Lever-aligned (`--event response`): from the cue to the lever push. The baseline is `--cue-baseline START END`
  (default `-1 0`) before each trial's cue, so extract these windows with a `--pre` long enough to reach it (e.g.
  `--pre 5`); trials whose baseline falls outside the window have no response fraction or variance quench
  baseline, and a warning gives their count.

Trials with a response time below `--min-rt` (default 0.2 s) are left out of the reliability metrics.

**Metrics.**

| Column | Meaning |
| --- | --- |
| `epoch_correlation` | Mean zero-lag Pearson correlation between every pair of trials, over the samples both have. How alike single trials are. |
| `response_fraction` | Fraction of trials whose mean epoch value exceeds their baseline mean by `--response-sd` (default 2) baseline SDs. |
| `variance_quench` | Across-trial variance over the epoch divided by that over the baseline. Below 1 when variability drops after the event. |
| `signal_fraction` | Fraction of single-trial variance explained by the trial-mean trace. |
| `reliability_n` | Trials used. |

**Groups.** Each metric is taken for every [trial group](#trial-groups), on that group's row of the session table.

**Trial counts.** None of the metrics depends on the number of trials, but all are noisier with fewer, so read
them alongside `reliability_n` when comparing groups.

**Time warping.** Epochs end at different times. By default, trials are compared sample by sample on the shared
time axis. `--time-warp` instead resamples each epoch onto a common grid from its start to its end, comparing the
shape of the response regardless of response time.

## As a library

```python
import pandas as pd
from mesoscopy.process import metrics

perievent = pd.read_csv("recording_smoothed_regions_event-cueonset_perievent.csv")
per_trial, per_session = metrics.metrics_tables(perievent, smoothing=5)
```

`metrics.trial_metrics` works on a `(n_trials, n_samples)` array for one region, and the per-metric functions
(`peak`, `auc`, `onset_time`, `extrapolated_onset`, `offset_time`, `decay_time`, `trace_correlation`) are
available on their own. `metrics.trial_groups` splits a trials table into the trial groups.
`metrics.reliability_metrics` takes the reliability columns for one region, per trial group, with options in a
`metrics.ReliabilityOptions`; `epoch_correlation`, `response_fraction`, `variance_quench` and `signal_fraction` work
on masked `(n_trials, n_samples)` arrays from `reliability_epochs` and `epoch_traces`.
