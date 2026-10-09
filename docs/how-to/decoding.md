# Decoding

`mesoscopy process decode` tests whether the activity of a region, or of all regions together, tells which stimulus
the animal saw on a trial, or whether it pushed the lever. It works on the per-trial windows written by
`process peri-event`, the same `_perievent.csv` that `process metrics` and `process connectivity` take, and needs
the `sdt_type` trials column of a go/no-go session.

```bash
mesoscopy process decode /path/to/recording_smoothed_regions_event-cueonset_perievent.csv
```

A peri-event file written from a `_regions.csv` already carries the trials columns. If yours does not, pass the
trials CSV with `--trials`.

## Output

The command writes `<stem>_label-stim_decoding.csv` with one row per decoder, region and
[trial group](metrics.md#trial-groups). The decoders are a logistic regression and a linear discriminant, each
fitted to one region at a time and to all regions together, which appear as the region `all`.

| Column | Meaning |
| --- | --- |
| `decoder` | `logistic` or `lda`. |
| `label` | What is decoded, `stim` or `response`. |
| `region` | The region, or `all` for the population decoder. |
| `group` | The trial group the decoder is fitted within. |
| `accuracy`, `accuracy_sd` | The fraction of held-out trials classified correctly, as the mean and SD over the folds. |
| `balanced_accuracy`, `balanced_accuracy_sd` | The mean of the two per-class accuracies, so a majority class cannot inflate it. |
| `d_prime` | The sensitivity index from the hit and false alarm rates of the decoder. |
| `auroc` | The area under the ROC curve of the decision values over all trials. |
| `f2` | The F-beta score with beta 2, weighting recall over precision. |
| `tp`, `fn`, `fp`, `tn` | The confusion counts over all trials, averaged over repeats. |
| `shuffle_accuracy`, `shuffle_accuracy_sd` | The accuracy of the same decoder on shuffled labels, as the mean and SD over the shuffles. |
| `shuffle_balanced_accuracy`, `shuffle_balanced_accuracy_sd` | The balanced accuracy on shuffled labels. |
| `shuffle_balanced_accuracy_95` | The 95th percentile of the shuffled balanced accuracy, the level a real score has to beat. |
| `p_value` | The fraction of shuffles whose balanced accuracy reaches the observed one, with one added to both counts. |
| `n_trials`, `n_class_a`, `n_class_b` | The trials used, and how many of them are in the positive and the negative class. |

The positive class is the go stimulus or the lever push. A positive `d_prime` means the decoder picks it out and a
positive population weight means higher activity favours it. The weights of the population decoder go to
`<stem>_label-stim_decoding-weights.csv`, one row per decoder, group and region, with the mean and SD over the fits.

Groups where either class has fewer than `--min-trials` trials (default 10) keep their counts and are otherwise
empty. Without `--shuffles` the shuffle columns have no column.

## Labels and groups

`--label stim`, the default, decodes the go against the no-go stimulus. `--label response` decodes the lever push
against no push. Both come from `sdt_type`.

Each decoder is fitted over all trials and within the groups that hold the other variable constant. For the
stimulus these are `all`, `resp-push` (hits against false alarms) and `resp-nopush` (misses against correct
rejections), so a decoder that still separates the stimuli among pushed trials is not reading the movement. For
the response they are `all`, `stim-go` (hits against misses) and `stim-nogo` (false alarms against correct
rejections).

## Epoch

The feature of each trial is each region's mean ∆F/F over the epoch. The epoch runs from the start of the response
window to the median response time of the trials that responded, so every trial has the same epoch. `--response
START END` bounds it (default all post-event samples), `--response-pad` extends it past the response by a fixed time
(default 0 s), and trials with a response time below `--min-rt` (default 0.2 s) are left out. Pass
`--no-mask-response` to use the whole response window instead.

`--per-trial-end` ends each trial's epoch at its own response, as the [connectivity](connectivity.md#epoch) epoch
does. Trials that did not respond then end at the median while the others spread around it, which lets the response
label be decoded from the epoch length alone, so leave it off unless you need the per-trial epoch.

## Validation

Every decoder is scored by stratified k-fold cross-validation with `--folds` folds (default 5), repeated `--repeats`
times (default 10) with a fresh split each time. Features are standardised within each training fold. The logistic
regression weights the classes to balance them, and the linear discriminant uses equal priors with automatic
shrinkage. Every region is scored on the same splits, so the regions compare on the same held-out trials.

## Chance level

The chance level comes from fitting the same decoders to `--shuffles` (default 200) permutations of the labels, each
under one k-fold. Raw accuracy on shuffled labels sits at the share of the majority class, balanced accuracy near
0.5, so the two together show whether a decoder beats chance or only the class imbalance. `p_value` is one-sided on
balanced accuracy and cannot fall below one over the number of shuffles plus one. Pass `--shuffles 0` to skip the
shuffles and leave the columns out.

## Rolling

Pass `--with-rolling` to follow decoding through the trial. The command then also writes
`<stem>_label-stim_decoding-rolling.csv`, with the same scores taken in windows that slide over the whole
peri-event window.

```bash
mesoscopy process decode /path/to/recording_smoothed_regions_event-cueonset_perievent.csv --with-rolling
```

Each window's feature is the region mean over its samples, with no epoch. `--rolling-window` (default 0.5 s) sets
the window length and `--rolling-step` (default 0.1 s) the time between windows, both rounded to whole samples. At
25 Hz the defaults give windows of 0.48 s every 0.08 s. The table has one row per decoder, region, group and
window, and `time` is the window centre relative to the event.

The chance level is taken once per decoder, region and group on the `--baseline` window (default every sample
before the event), since shuffled labels give the same chance level in every window. The shuffle columns therefore
repeat along a time course and `p_value` is read against them in every window. The rolling table is the slowest
part of the command, since the population decoder is fitted in every window.

## Reading the table

`balanced_accuracy` is the headline score. A region whose score sits above `shuffle_balanced_accuracy_95` carries
information about the label. `auroc` ranks the decision values rather than thresholding them, so it is unmoved by the
class imbalance and by the choice of threshold.

The population decoder usually beats every single region, since regions carry partly independent information. Its
weights show which regions it leans on, in units of a standard deviation of each region's feature.

A region that decodes the stimulus before the cue in the rolling table reads the structure of the session rather
than the stimulus. Correction trials repeat the stimulus after an error, so the coming stimulus can be predicted
from the last trial.

## As a library

```python
import pandas as pd
from mesoscopy.process import decoding

perievent = pd.read_csv("recording_smoothed_regions_event-cueonset_perievent.csv")
tables = decoding.decoding_table(perievent, label="stim")
tables.table.query("decoder == 'lda' and group == 'all'").nlargest(5, "balanced_accuracy")
```

`decoding.decoding_table` returns a `DecodingTables` with the scores in `table` and the population weights in
`weights`. `decoding.rolling_table` returns the rolling table. `decoding.region_decisions` fits the one-feature
decoders of every region, fold and shuffle at once from a `(n_trials, n_features)` array and the fold of each trial,
and `decoding.fit_scores` fits one scikit-learn decoder per fold.
