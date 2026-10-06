# Connectivity

`mesoscopy process connectivity` measures how pairs of regions covary over the cue-to-response epoch of the
per-trial windows written by `process peri-event`, for all trials and per trial group.

## Input

A long-format `_perievent.csv`, as written by `process peri-event` from a `_regions.csv`, with the trials
columns carried over or joined with `--trials` (see [Response metrics](metrics.md#input)):

```bash
mesoscopy process connectivity /path/to/recording_smoothed_regions_event-cueonset_perievent.csv
```

## Output

`<stem>_connectivity.csv` has one row per pair of regions and [trial group](metrics.md#trial-groups). Each pair
appears once, in the order the regions appear in the peri-event file.

| Column | Meaning |
| --- | --- |
| `region_a`, `region_b` | The pair. |
| `group` | Trial group, as in the session metrics. |
| `r` | Pearson correlation between the two regions over the epoch samples of every trial in the group, pooled. |
| `r_residual` | The same after subtracting the group's mean response at each sample from every trial: the noise correlation. |
| `r_trials_avg` | Mean of the correlations taken within each trial, averaged as Fisher z. Trials with fewer than three epoch samples are left out. |
| `partial_r` | Partial correlation given every other region. |
| `lag`, `r_lag` | Lag of the largest absolute cross-correlation within `--max-lag` (default 0.5 s), in seconds, and the correlation there. Positive when `region_b` follows `region_a`. |
| `n_trials`, `n_samples` | Trials used, and epoch samples pooled over them. |

A group with fewer than `--min-trials` (default 10) trials keeps `n_trials` and is otherwise empty; the `all`
rows are always taken. Sessions without the `sdt_type` column, or other than go/no-go, get the `all` rows only,
with a warning.

## Epoch

The epoch is the [reliability](metrics.md#reliability) epoch: from the cue to each trial's response, within the
response window (`--response START END`, default all post-event samples). Misses and correct rejections end at
the median response time of the trials with a response, and lever-aligned windows run from the cue to the push.
`--no-mask-response` uses the whole response window. Trials with a response time below `--min-rt` (default
0.2 s) are left out.

The epoch needs the `cue_onset` and `response_time` trials columns and the event the windows are aligned to.
Without them, or for reward-aligned windows, the whole response window is used for every trial, with a warning.

## Reading the metrics

- `r` mixes two things: that both regions respond to the cue, and that they fluctuate together from moment to
  moment. `r_residual` keeps only the second, by removing each group's mean response first. On z-scored
  recordings, where single-trial fluctuations are large next to the mean response, the two are close.
- `r_trials_avg` is taken within trials, so differences in overall level between trials do not count. It runs
  higher than `r` when trials sit at different levels.
- `partial_r` removes what a pair shares with every other region, including any global signal. A pair with a
  strong `r` and a `partial_r` near zero is coupled through the rest of the cortex. It needs more samples than
  regions, so small groups give unstable values.
- `lag` is in steps of the sample interval. A lag at the edge of `--max-lag` means the peak was not within range.

## As a library

```python
import pandas as pd
from mesoscopy.process import connectivity

perievent = pd.read_csv("recording_smoothed_regions_event-cueonset_perievent.csv")
table = connectivity.connectivity_table(perievent)
```

`connectivity.pair_metrics` returns the matrices of one trial group from a masked `(n_trials, n_samples,
n_regions)` array, built by `connectivity.epoch_cube` from the per-trial bounds of `connectivity.epoch_bounds`.
`correlation_matrix`, `residual_epochs`, `trial_correlation`, `partial_correlation`, `lagged_correlation` and
`peak_lag` are available on their own.
