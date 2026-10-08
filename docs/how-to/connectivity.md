# Connectivity

`mesoscopy process connectivity` measures how pairs of regions covary while the animal responds to the cue. It
works on the per-trial windows written by `process peri-event`, the same `_perievent.csv` that `process metrics`
takes.

```bash
mesoscopy process connectivity /path/to/recording_smoothed_regions_event-cueonset_perievent.csv
```

A peri-event file written from a `_regions.csv` already carries the trials columns. If yours does not, pass the
trials CSV with `--trials`.

## Output

The command writes `<stem>_connectivity.csv` with one row per pair of regions and
[trial group](metrics.md#trial-groups). Each pair appears once, in the order the regions appear in the peri-event
file.

| Column | Meaning |
| --- | --- |
| `region_a`, `region_b` | The pair. |
| `group` | The trial group, as in the session metrics. |
| `r` | The Pearson correlation between the two regions over every epoch sample of the group's trials. |
| `r_residual` | The same correlation after the group's mean response is subtracted from every trial. This is the noise correlation. |
| `r_trials_avg` | The correlation within each trial, averaged across trials as Fisher z. |
| `mi`, `mi_residual` | The mutual information between the two regions in bits, over the same samples as `r` and `r_residual`. |
| `partial_r` | The partial correlation between the two regions, given every other region. |
| `lag`, `r_lag` | The lag of the peak cross-correlation within `--max-lag` (default 0.5 s), in seconds, and the correlation at that lag. A positive lag means `region_b` follows `region_a`. |
| `te_ab`, `te_ba` | With `--with-te`, the transfer entropy from `region_a` to `region_b` and back, in bits, over the same samples as `r_residual`. |
| `te_ab_z`, `te_ba_z` | Their z-scores against `--te-surrogates` (default 200) surrogates; absent with 0. |
| `n_trials`, `n_samples` | The trials used and the samples pooled across them. |

Groups with fewer than `--min-trials` trials (default 10) keep their `n_trials` and are otherwise empty. The `all`
rows are always filled. A session without the `sdt_type` column gets the `all` rows only. Metrics the command does
not estimate, the mutual information with `--no-mi` and the transfer entropy without `--with-te`, have no column.

Keep the table next to the peri-event file and `mesoscopy report` on that file shows it as a heatmap, one metric
and trial group at a time. See [Reports](reports.md#peri-event-report).

## Epoch

Each trial contributes the samples between the cue and its response. Misses and correct rejections end at the
median response time of the trials that did respond, and lever-aligned windows run from the cue to the push.
`--response START END` bounds the epoch (default all post-event samples), and trials with a response time below
`--min-rt` (default 0.2 s) are left out. This is the same epoch the [reliability](metrics.md#reliability) metrics
use. Pass `--no-mask-response` to use the whole response window instead.

The epoch needs the `cue_onset` and `response_time` trials columns. Without them the command falls back to the
whole response window and warns.

## Whole recording

A `_regions.csv` from `process regions` gives the same table over the whole recording.

```bash
mesoscopy process connectivity /path/to/recording_smoothed_regions.csv
```

It writes `<stem>_connectivity.csv` next to the epoch table, so `recording_smoothed_regions.csv` gives
`recording_smoothed_regions_connectivity.csv` and its peri-event file gives
`recording_smoothed_regions_event-cueonset_connectivity.csv`. The rows are the `all` group only. `r_residual`,
`r_trials_avg` and `mi_residual` have no column, since there are no trials to take them over, `n_trials` is
empty, and `n_samples` is the number of frames. With `--with-te` the transfer entropy is taken over the raw
traces, with circular shifts of the source as surrogates. The trial options are ignored. This table is the
baseline the epoch tables depart from.

## Reading the table

`r` is high whenever both regions respond to the cue, whether or not they fluctuate together from moment to
moment. `r_residual` strips out each group's mean response first, so it keeps only the moment-to-moment
covariation. On z-scored recordings the mean response is a small part of the epoch variance, and the two come out
close.

`r_trials_avg` is taken within trials, so trials sitting at different levels do not pull it down.

`mi` captures dependence of any shape, where `r` captures linear dependence only. A Gaussian pair has
`-0.5 * log2(1 - r²)` bits, so a `mi` well above that marks a non-linear relation. Independent regions land near
zero, on either side of it. The estimator is the Kraskov-Stögbauer-Grassberger nearest-neighbour method with
`--mi-neighbours` (default 3) neighbours. It is the slowest part of the command. Pass `--no-mi` to skip it and
leave the columns out.

`partial_r` removes what a pair shares with every other region, including the global signal. A pair with a strong
`r` and a `partial_r` near zero is coupled through the rest of the cortex.

`lag` moves in steps of the sample interval. A lag sitting at `--max-lag` means the peak lies beyond it.

`te_ab` is the information `region_a` adds to predicting `region_b` beyond what `region_b`'s own past gives, and
`te_ba` the reverse, so the two together give the direction of a coupling. The columns only appear when the
command runs with `--with-te`. The estimator is the Gaussian one, half the Granger log-ratio, conditioned on
`--te-history` (default 1) past samples of the target and taken at the source lag within `--max-lag` that gives
the most. On the epoch tables it is taken over the residuals, like `r_residual`. The values are small, since the
target's own past already explains most of its next sample. A value of 0.03 bits means the source explains 4% of
what is left, and the whole-trace values of a recording run from 0.01 to 0.1 bits. Read them against other pairs,
groups and sessions rather than against the `mi` scale.

`te_ab_z` says how far the value sits above surrogates with the same source but no time alignment to the target,
trial shuffles on the epoch tables and circular shifts on the whole trace, so it is the significance of the
direction. The epoch values are noisy and are to be read with it. On the whole trace every pair comes out
significant, so rank by `te_ab` there. The estimate is seeded by `--seed` (default 42). Pass `--te-surrogates 0`
to skip the surrogates and leave the z-score columns out.

## As a library

```python
import pandas as pd
from mesoscopy.process import connectivity

perievent = pd.read_csv("recording_smoothed_regions_event-cueonset_perievent.csv")
table = connectivity.connectivity_table(perievent)
```

`connectivity.trace_table` takes the regions table of a `_regions.csv` instead.

`connectivity.pair_metrics` returns the matrices of one trial group from a masked `(n_trials, n_samples,
n_regions)` array, built by `connectivity.epoch_cube` from the per-trial bounds of `connectivity.epoch_bounds`.
`connectivity.pooled_metrics` returns the metrics that need no trials, which is what `trace_table` takes over the
traces of `connectivity.trace_samples`.
`correlation_matrix`, `residual_epochs`, `trial_correlation`, `mutual_information`, `partial_correlation`,
`lagged_correlation`, `peak_lag`, `transfer_entropy` and `transfer_entropy_circular` work on their own, and
`epoch_transfer_entropy` and `trace_transfer_entropy` add the surrogate z-scores.
