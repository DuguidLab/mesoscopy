# Response metrics

`mesoscopy process metrics` turns the per-trial windows written by `process peri-event` into response metrics for
every trial and region, a summary of how those metrics vary across trials, and the same metrics taken from the
trial-mean trace with bootstrap intervals.

## Input

It takes a long-format `_perievent.csv`, as written by `process peri-event` from a `_regions.csv`.

```bash
mesoscopy process regions /path/to/recording_smoothed.h5
mesoscopy process peri-event /path/to/recording_smoothed_regions.csv /path/to/recording_trials.csv --event cue_onset --pre 1 --post 3
mesoscopy process metrics /path/to/recording_smoothed_regions_event-cueonset_perievent.csv
```

`process peri-event --with-metrics` writes the same tables in one step and accepts every option of
`process metrics`.

```bash
mesoscopy process peri-event /path/to/recording_smoothed_regions.csv /path/to/recording_trials.csv --with-metrics
```

A peri-event `--baseline` also serves as the metrics baseline window. HDF5 peri-event windows get no metrics.

## Output

`<stem>_metrics.csv` has one row per trial per region.

| Column | Meaning |
| --- | --- |
| `trial_index`, `event_time`, `region` | Carried over from the peri-event file. |
| `baseline_mean`, `baseline_sd` | The mean and SD of the raw trace over the baseline window. |
| `onset_time` | The time of the response onset, in seconds from the event. |
| `peak_time`, `amplitude` | The time and height of the signed maximum over the response window. |
| `auc` | The signed trapezoidal area under the baseline-subtracted trace over the response window. |
| `decay_time` | The time from the peak until the trace falls to a fraction of the amplitude. |
| `offset_time` | The time at which the response returns to the onset threshold after the peak. |
| `duration` | `offset_time` minus `onset_time`. |

The trials columns such as `sdt_type`, `outcome` and `response_time` follow, carried over from the peri-event
file. For a peri-event file without them, `--trials` joins a trials CSV by row index.

`<stem>_metrics-session.csv` has one row per region and [trial group](#trial-groups). It holds `region`, `group`,
`n_trials` and `trace_correlation`, the mean pairwise Pearson correlation between trial traces over the response
window, then `<metric>_mean`, `<metric>_sd` and `<metric>_cv` across trials for each metric above, over the trials
where the metric was found. `onset_n`, `decay_n` and `offset_n` count those trials.

For go/no-go sessions aligned to the cue, trial start or lever push, the session table also has the
[reliability](#reliability) columns.

`<stem>_metrics-boot.csv` and `<stem>_traces-boot.csv` hold the [mean-trace metrics](#mean-trace-metrics) and the
mean traces they are taken from.

A metric is empty when it is undefined for that trial, such as a decay when the trace never falls back to the
fraction or an onset when the trace never crosses the threshold.

## Trial groups

For go/no-go sessions, the session metrics are taken over all trials and over groups of trials, one row each.

| `group` | Trials |
| --- | --- |
| `all` | Every trial. |
| `sdt-hit`, `sdt-miss`, `sdt-false_alarm`, `sdt-correct_rejection` | One trial type. |
| `stim-go`, `stim-nogo` | Hits and misses, or false alarms and correct rejections, by the stimulus shown. |
| `resp-push`, `resp-nopush` | Hits and false alarms, or misses and correct rejections, by whether the lever was pushed. |

The groups come from the `sdt_type` trials column, carried over by `process peri-event` or joined with `--trials`.
Without it the session table has the `all` rows only.

Every group gets a row. A group with fewer than `--min-trials` (default 10) trials keeps its trial counts
(`n_trials`, `onset_n`, `decay_n`, `offset_n` and `reliability_n`) and is otherwise empty. The `all` row's
across-trial summaries are always taken. Its reliability columns follow the same minimum, counted on
`reliability_n`, the trials left after `--min-rt`.

## Metrics

Every trace is first baseline-subtracted with its own mean over the baseline window, `--baseline START END`
(default all pre-event samples, end exclusive). The metrics are then taken over the response window,
`--response START END` (default all post-event samples, end inclusive).

**Onset** is found in one of three ways, chosen with `--onset`.

- `sd` (default) takes the first `--onset-min-samples` consecutive samples above `--onset-sd` times the baseline
  SD. It is simple and per trial, and it depends on how well 25 or so baseline samples estimate the noise.
- `peak` does the same with the threshold at `--onset-fraction` of the peak amplitude. It is independent of
  baseline noise and inherits whatever the peak does.
- `extrapolate` fits a line to the last rise before the peak, between `--extrapolate-range LOW HIGH` fractions of
  the amplitude, and takes the onset where that line crosses baseline. This is the standard latency estimate in
  electrophysiology. It is robust on clean responses, and on noisy traces the fitted rise can be short and put the
  onset before the event.

**Offset** is the first `--onset-min-samples` consecutive samples at or below the onset threshold after the peak,
or below `LOW` of the amplitude for `--onset extrapolate`. It needs the peak above the threshold and the trace
back down within the response window.

**Decay** is the time from the peak until the trace first falls to `--decay-fraction` of the amplitude. It needs
a positive amplitude.

**Smoothing.** On noisy traces the peak can land on a single-sample spike, which shortens the decay and shifts
the offset. `--smooth N` finds the peak, onset, decay and offset on an `N`-sample centred moving average, with `N`
odd. The baseline SD and the AUC always come from the raw trace, so smoothing does not shrink the noise estimate
behind the `sd` onset threshold.

## Choosing settings

- Z-scored recordings make the `sd` onset threshold comparable across regions.
- If onsets look early, raise `--onset-sd` or `--onset-min-samples`. If many trials have no onset, lower them.
- If decays look implausibly short, add `--smooth 5` and check the peak times.
- The session CV is unstable for any metric whose mean sits near zero, which the extrapolated onset often does.
  Read the SD instead.

## Mean-trace metrics

Single-trial onsets, peaks and amplitudes are noisy. The mean-trace metrics take the same metrics from the mean of
each trial group's traces instead, with a bootstrap interval for each.

Each trace is baseline-subtracted as above and the traces are averaged, ignoring missing samples. The metrics are
taken on this mean trace with the same options as the per-trial metrics. The trials are then resampled with
replacement `--bootstrap` times (default 10000), the metrics are taken on the mean of each resample, and the
interval spans the central `--ci` percent (default 95) of the resampled values. Each trial group is resampled as one
pool, and every region uses the same resamples. `--seed` (default 42) fixes the resamples, so a rerun gives the same
intervals. `--bootstrap 0` skips the mean-trace metrics.

`<stem>_metrics-boot.csv` has one row per region and trial group with `region`, `group`, `n_trials` and, for each
metric, `boot_<metric>` for the metric of the mean trace, `boot_<metric>_ci_low` and `boot_<metric>_ci_high` for
its interval, and `boot_<metric>_n` for the resamples where the metric was found. A group with fewer than
`--min-trials` trials keeps only `n_trials`.

`<stem>_traces-boot.csv` has the mean trace of each region and trial group, one row per sample, with `region`,
`group`, `n_trials`, `time`, `mean`, and the pointwise interval `ci_low` and `ci_high`. Groups with fewer than
`--min-trials` trials are left out.

Keep four things in mind when reading them.

- With `--onset sd`, the threshold comes from the mean trace's own baseline SD, which is smaller than a single
  trial's, so mean-trace onsets come earlier than single-trial ones.
- A mean-trace onset at the start of the response window means the trace was already above the threshold there.
  `--onset extrapolate` can place the onset before the window.
- The mean trace's amplitude is lower than the mean single-trial amplitude, because single-trial peaks ride on
  noise and responses that vary in timing flatten when averaged.
- An interval is taken over the resamples where the metric was found, so read it alongside `boot_<metric>_n`. An
  offset found in a few hundred of 10000 resamples has an unreliable interval.

## Reliability

The reliability metrics measure how consistent a region's response is from trial to trial, over the part of each
trial before the lever push, so that movement does not inflate them. They are taken for go/no-go sessions and need
the `cue_onset`, `response_time` and `sdt_type` trials columns, carried over by `process peri-event` or joined with
`--trials`. Without them, or for reward-aligned windows, the columns are skipped with a warning.

**Epoch.** The event the windows are aligned to is read from which trials column matches `event_time`. Cue- and
trial-start-aligned epochs run over the response window and end at each trial's response. Misses and correct
rejections end at the median response time of the trials with a response, and `--no-mask-response` uses the whole
response window instead. Lever-aligned epochs (`--event response`) run from the cue to the lever push, with the
baseline `--cue-baseline START END` (default `-1 0`) before each trial's cue, so extract these windows with a
`--pre` long enough to reach it, such as `--pre 5`. Trials with a response time below `--min-rt` (default 0.2 s)
are left out.

**Metrics.**

| Column | Meaning |
| --- | --- |
| `epoch_correlation` | The mean zero-lag Pearson correlation between every pair of trials, over the samples both have. How alike single trials are. |
| `response_fraction` | The fraction of trials whose mean epoch value exceeds their baseline mean by `--response-sd` (default 2) baseline SDs. |
| `variance_quench` | The across-trial variance over the epoch divided by that over the baseline. It is below 1 when variability drops after the event. |
| `signal_fraction` | The fraction of single-trial variance explained by the trial-mean trace. |
| `reliability_n` | The trials used. |

**Groups.** Each metric is taken for every [trial group](#trial-groups), on that group's row of the session table.
None of the metrics depends on the number of trials, but all are noisier with fewer, so read them alongside
`reliability_n` when comparing groups.

**Time warping.** Epochs end at different times. By default, trials are compared sample by sample on the shared
time axis. `--time-warp` resamples each epoch onto a common grid from its start to its end instead, which compares
the shape of the response regardless of response time.

## As a library

```python
import pandas as pd
from mesoscopy.process import metrics

perievent = pd.read_csv("recording_smoothed_regions_event-cueonset_perievent.csv")
tables = metrics.metrics_tables(perievent, smoothing=5)
tables.per_trial, tables.per_session, tables.boot, tables.boot_traces
```

`metrics.trial_metrics` works on a `(n_trials, n_samples)` array for one region, and the per-metric functions
`peak`, `auc`, `onset_time`, `extrapolated_onset`, `offset_time`, `decay_time` and `trace_correlation` work on their
own. `metrics.trial_groups` splits a trials table into the trial groups. `metrics.bootstrap_metrics` takes the
mean-trace metrics of one region's traces, with resamples from `metrics.resample_counts`, and `metrics_tables`
takes its options in a `metrics.BootstrapOptions`. `metrics.reliability_metrics` takes the reliability columns for
one region, per trial group, with options in a `metrics.ReliabilityOptions`. `epoch_correlation`,
`response_fraction`, `variance_quench` and `signal_fraction` work on masked `(n_trials, n_samples)` arrays from
`reliability_epochs` and `epoch_traces`.
