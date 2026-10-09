#  Copyright (c) 2026 Constantinos Eleftheriou <Constantinos.Eleftheriou@ed.ac.uk>.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this
#   software and associated documentation files (the "Software"), to deal in the
#   Software without restriction, including without limitation the rights to use, copy,
#   modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#   and to permit persons to whom the Software is furnished to do so, subject to the
#  following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies
#  or substantial portions of the Software
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
#  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
#  IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
#  IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

"""Trial-level decoding of  stimulus or  response from region peri-event traces.

Each region, and all regions together, is used to classify the trials of a go/no-go session with a logistic
regression and a linear discriminant decoder under repeated stratified cross-validation.
"""

from __future__ import annotations

import dataclasses
import typing
import warnings

import numpy as np
import numpy.typing as npt

from mesoscopy.process import connectivity as pc
from mesoscopy.process import metrics as pm

if typing.TYPE_CHECKING:
    import pandas as pd

DECODERS = ("logistic", "lda")
LABELS: dict[str, tuple[tuple[str, str], tuple[str, ...]]] = {
    "stim": (("stim-go", "stim-nogo"), ("all", "resp-push", "resp-nopush")),
    "response": (("resp-push", "resp-nopush"), ("all", "stim-go", "stim-nogo")),
}

# Region value of the population decoder's rows.
POPULATION = "all"

# Scores of one decoder, in output column order.
SCORES = (
    "accuracy",
    "accuracy_sd",
    "balanced_accuracy",
    "balanced_accuracy_sd",
    "d_prime",
    "auroc",
    "f2",
    "tp",
    "fn",
    "fp",
    "tn",
)

# Columns of the decoding table, in order.
TABLE_COLUMNS = ("decoder", "label", "region", "group", *SCORES, "n_trials", "n_class_a", "n_class_b")

# Columns of the weights table, in order.
WEIGHT_COLUMNS = ("decoder", "label", "group", "region", "weight", "weight_sd")


@dataclasses.dataclass(frozen=True)
class DecodingTables:
    """Tables from `decoding_table`.

    Attributes:
        table (pd.DataFrame): One row per decoder, region and trial group, written to `<stem>_decoding.csv`.
        weights (pd.DataFrame): Population decoder weights, one row per decoder, trial group and region, written to
            `<stem>_decoding-weights.csv`.
    """

    table: pd.DataFrame
    weights: pd.DataFrame


def trial_labels(trial_info: pd.DataFrame, label: str) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """Class of each trial for a label.

    Args:
        trial_info (pd.DataFrame): One row per trial with an `sdt_type` column.
        label (str): One of `LABELS`.

    Returns:
        tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]: Class per trial, 1 for the positive class (the go
        stimulus or the lever push) and 0 for the negative, and which trials belong to either class; each shape
        `(n_trials,)`.

    Raises:
        ValueError: If `label` is not one of `LABELS`.
    """
    if label not in LABELS:
        msg = f"Unknown label {label!r}; expected one of {', '.join(LABELS)}."
        raise ValueError(msg)
    (positive, negative), _ = LABELS[label]
    groups = pm.trial_groups(trial_info)
    return groups[positive].astype(np.int64), groups[positive] | groups[negative]


def epoch_features(
    perievent: pd.DataFrame,
    trial_index: npt.NDArray,
    time: npt.NDArray[np.float64],
    regions: list[str],
    start: npt.NDArray[np.float64],
    end: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Mean of each region's trace over each trial's epoch.

    Args:
        perievent (pd.DataFrame): Long-format peri-event table with `trial_index`, `time`, `region` and `F`.
        trial_index (npt.NDArray): Trials in row order, shape `(n_trials,)`.
        time (npt.NDArray[np.float64]): Sample times relative to the event, shape `(n_samples,)`.
        regions (list[str]): Regions in column order.
        start (npt.NDArray[np.float64]): Per-trial epoch start, shape `(n_trials,)`.
        end (npt.NDArray[np.float64]): Per-trial epoch end, exclusive, shape `(n_trials,)`.

    Returns:
        npt.NDArray[np.float64]: Features of shape `(n_trials, n_regions)`, NaN for a trial without a sample in
        its epoch.
    """
    cube = pc.epoch_cube(perievent, trial_index, time, regions, start, end)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(cube, axis=1)


def make_decoder(decoder: str) -> typing.Any:
    """Unfitted decoder pipeline, standardising the features first.

    Args:
        decoder (str): `logistic` for an L2 logistic regression with balanced class weights, or `lda` for a linear
            discriminant with Ledoit-Wolf shrinkage and equal priors.

    Returns:
        typing.Any: A scikit-learn pipeline.

    Raises:
        ValueError: If `decoder` is not one of `DECODERS`.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if decoder == "logistic":
        model: typing.Any = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    elif decoder == "lda":
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])
    else:
        msg = f"Unknown decoder {decoder!r}; expected one of {', '.join(DECODERS)}."
        raise ValueError(msg)
    return make_pipeline(StandardScaler(), model)


def cross_validation_splits(
    y: npt.NDArray[np.int64], folds: int, repeats: int, seed: int | None
) -> list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
    """Train and test indices of a repeated stratified k-fold.

    Args:
        y (npt.NDArray[np.int64]): Class per trial, shape `(n_trials,)`.
        folds (int): Folds per repeat.
        repeats (int): Repeats, each with a fresh shuffle.
        seed (int | None): Seed for the shuffles.

    Returns:
        list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]: `folds * repeats` pairs of train and test
        indices, repeat by repeat.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold

    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    return [(train, test) for train, test in splitter.split(np.zeros((len(y), 1)), y)]


def fit_scores(
    features: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    splits: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]],
    decoder: str,
    folds: int,
) -> dict[str, typing.Any]:
    """Cross-validated scores and weights of one decoder.

    Each repeat scores every trial once, so the confusion counts are summed over the folds of a repeat and
    averaged over repeats, and the AUROC is taken over each repeat's pooled decision values.

    Args:
        features (npt.NDArray[np.float64]): Features of shape `(n_trials, n_features)`.
        y (npt.NDArray[np.int64]): Class per trial, shape `(n_trials,)`.
        splits (list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]): Train and test indices from
            `cross_validation_splits`.
        decoder (str): One of `DECODERS`.
        folds (int): Folds per repeat, to group the splits by repeat.

    Returns:
        dict[str, typing.Any]: `accuracy` and `balanced_accuracy` per fold, shape `(n_splits,)`; `auroc` per
        repeat, shape `(n_repeats,)`; `tp`, `fn`, `fp` and `tn` averaged over repeats; `weights` of shape
        `(n_splits, n_features)`, one row per fit.
    """
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.metrics import roc_auc_score

    n_splits = len(splits)
    accuracy = np.empty(n_splits)
    balanced = np.empty(n_splits)
    weights = np.empty((n_splits, features.shape[1]))
    predicted = np.empty(len(y), dtype=np.int64)
    decision = np.empty(len(y))
    auroc = []
    confusion = np.zeros(4)
    for i, (train, test) in enumerate(splits):
        model = make_decoder(decoder).fit(features[train], y[train])
        predicted[test] = model.predict(features[test])
        decision[test] = model.decision_function(features[test])
        weights[i] = model[-1].coef_[0]
        accuracy[i] = float(np.mean(predicted[test] == y[test]))
        balanced[i] = balanced_accuracy_score(y[test], predicted[test])
        if (i + 1) % folds == 0:
            auroc.append(roc_auc_score(y, decision))
            confusion += confusion_counts(y, predicted)
    n_repeats = n_splits // folds
    tp, fn, fp, tn = confusion / n_repeats
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "auroc": np.asarray(auroc),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "weights": weights,
    }


def confusion_counts(y: npt.NDArray, predicted: npt.NDArray) -> npt.NDArray[np.float64]:
    """True positives, false negatives, false positives and true negatives.

    Args:
        y (npt.NDArray): Class per trial, 1 positive and 0 negative.
        predicted (npt.NDArray): Predicted class per trial.

    Returns:
        npt.NDArray[np.float64]: The four counts, in that order.
    """
    y = np.asarray(y) == 1
    predicted = np.asarray(predicted) == 1
    return np.array(
        [
            (y & predicted).sum(),
            (y & ~predicted).sum(),
            (~y & predicted).sum(),
            (~y & ~predicted).sum(),
        ],
        dtype=np.float64,
    )


def d_prime(tp: float, fn: float, fp: float, tn: float) -> float:
    """Sensitivity index from confusion counts.

    Hit and false alarm rates of exactly 0 or 1 are pulled in to `1 / (2N)` and `1 - 1 / (2N)`, with `N` the
    trials of that class (Macmillan & Kaplan, 1985), so the index stays finite.

    Args:
        tp (float): True positives.
        fn (float): False negatives.
        fp (float): False positives.
        tn (float): True negatives.

    Returns:
        float: `z(hit rate) - z(false alarm rate)`, NaN without trials of either class.
    """
    from scipy.stats import norm

    n_positive = tp + fn
    n_negative = fp + tn
    if n_positive <= 0 or n_negative <= 0:
        return float("nan")
    hit_rate = np.clip(tp / n_positive, 1 / (2 * n_positive), 1 - 1 / (2 * n_positive))
    fa_rate = np.clip(fp / n_negative, 1 / (2 * n_negative), 1 - 1 / (2 * n_negative))
    return float(norm.ppf(hit_rate) - norm.ppf(fa_rate))


def f_beta(tp: float, fn: float, fp: float, beta: float = 2.0) -> float:
    """F-beta score from confusion counts.

    Args:
        tp (float): True positives.
        fn (float): False negatives.
        fp (float): False positives.
        beta (float, optional): Weight of recall over precision. Defaults to 2.

    Returns:
        float: The score, NaN when nothing is positive or predicted positive.
    """
    denominator = (1 + beta**2) * tp + beta**2 * fn + fp
    return float((1 + beta**2) * tp / denominator) if denominator > 0 else float("nan")


def score_summary(scores: dict[str, typing.Any]) -> dict[str, float]:
    """The `SCORES` of one decoder from its `fit_scores`.

    Args:
        scores (dict[str, typing.Any]): Per-fold scores and confusion counts from `fit_scores`.

    Returns:
        dict[str, float]: One value per `SCORES` name.
    """
    tp, fn, fp, tn = (float(scores[name]) for name in ("tp", "fn", "fp", "tn"))
    return {
        "accuracy": float(np.mean(scores["accuracy"])),
        "accuracy_sd": float(np.std(scores["accuracy"], ddof=1)) if len(scores["accuracy"]) > 1 else float("nan"),
        "balanced_accuracy": float(np.mean(scores["balanced_accuracy"])),
        "balanced_accuracy_sd": float(np.std(scores["balanced_accuracy"], ddof=1))
        if len(scores["balanced_accuracy"]) > 1
        else float("nan"),
        "d_prime": d_prime(tp, fn, fp, tn),
        "auroc": float(np.mean(scores["auroc"])),
        "f2": f_beta(tp, fn, fp),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }


def usable_trials(features: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Trials with a value in every region that has one anywhere.

    Args:
        features (npt.NDArray[np.float64]): Features of shape `(n_trials, n_regions)`.

    Returns:
        tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]: Which trials to use, shape `(n_trials,)`, and which
        regions have a value, shape `(n_regions,)`.
    """
    present = ~np.isnan(features).all(axis=0)
    complete = present.any() & ~np.isnan(features[:, present]).any(axis=1)
    return complete, present


def group_decoding(
    features: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    regions: list[str],
    present: npt.NDArray[np.bool_],
    folds: int,
    repeats: int,
    seed: int | None,
) -> dict[str, dict[str, typing.Any]]:
    """Scores of every decoder on each region and on all regions, for one trial group.

    All fits share the same splits, so the regions are compared on the same test trials.

    Args:
        features (npt.NDArray[np.float64]): Features of the group's trials, shape `(n_trials, n_regions)`.
        y (npt.NDArray[np.int64]): Class per trial, shape `(n_trials,)`.
        regions (list[str]): Regions in column order.
        present (npt.NDArray[np.bool_]): Regions with a value, shape `(n_regions,)`; the others are left out.
        folds (int): Folds per repeat.
        repeats (int): Repeats of the k-fold.
        seed (int | None): Seed for the shuffles.

    Returns:
        dict[str, dict[str, typing.Any]]: Per decoder, the `score_summary` of each region keyed by region, of all
        regions keyed by `POPULATION`, and `weights`, the population weights' mean and SD over the fits per region,
        each shape `(n_regions,)`.
    """
    splits = cross_validation_splits(y, folds, repeats, seed)
    columns = np.flatnonzero(present)
    result: dict[str, dict[str, typing.Any]] = {}
    for decoder in DECODERS:
        scores: dict[str, typing.Any] = {}
        for column in columns:
            scores[regions[column]] = score_summary(fit_scores(features[:, [column]], y, splits, decoder, folds))
        population = fit_scores(features[:, columns], y, splits, decoder, folds)
        scores[POPULATION] = score_summary(population)
        mean = np.full(len(regions), np.nan)
        sd = np.full(len(regions), np.nan)
        mean[columns] = population["weights"].mean(axis=0)
        sd[columns] = population["weights"].std(axis=0, ddof=1) if len(splits) > 1 else np.nan
        scores["weights"] = (mean, sd)
        result[decoder] = scores
    return result


def decoding_table(
    perievent: pd.DataFrame,
    label: str = "stim",
    response: tuple[float, float] | None = None,
    trials: pd.DataFrame | None = None,
    event: str | None = None,
    mask_response: bool = True,
    min_rt: float = 0.2,
    response_pad: float = 0.0,
    min_trials: int = 10,
    folds: int = 5,
    repeats: int = 10,
    seed: int | None = 42,
) -> DecodingTables:
    """Decode the stimulus or the response of each trial from region activity, per trial group.

    The feature of each trial is each region's mean over its epoch from `connectivity.epoch_bounds`. Each region,
    and all regions together, is scored with every `DECODERS` decoder under a repeated stratified k-fold, over all
    trials and within the groups that hold the other variable constant.

    Args:
        perievent (pd.DataFrame): Peri-event table with `trial_index`, `event_time`, `time`, `region` and `F`
            columns, as written by `process peri-event` from a `_regions.csv`, plus the `sdt_type` trials column;
            checked by `metrics.perievent_trials`.
        label (str, optional): `stim` for the go against the no-go stimulus, or `response` for the lever push
            against no push. Defaults to `stim`.
        response (tuple[float, float] | None, optional): Response window `[start, end]`. Defaults to all
            post-event samples.
        trials (pd.DataFrame | None, optional): Trials table to join by row index, via `metrics.join_trials`, for
            peri-event tables without trials columns. Defaults to None.
        event (str | None, optional): Event the windows are aligned to, one of `perievent.EVENTS`. Defaults to
            the event matching `event_time`, via `metrics.infer_event`.
        mask_response (bool, optional): End cue-aligned epochs at each trial's response. Defaults to True.
        min_rt (float, optional): Trials with a response time below this, in seconds, are dropped. Defaults to
            0.2.
        response_pad (float, optional): Seconds added to the end of each trial's epoch. Defaults to 0.
        min_trials (int, optional): Trials each class needs in a group; groups with fewer get empty scores.
            Defaults to 10.
        folds (int, optional): Folds per repeat. Defaults to 5.
        repeats (int, optional): Repeats of the k-fold, each with a fresh shuffle. Defaults to 10.
        seed (int | None, optional): Seed for the shuffles. Defaults to 42.

    Returns:
        DecodingTables: `table` with one row per decoder, region (plus `all` for the population decoder) and
        group, with the `TABLE_COLUMNS`, and `weights` with one row per decoder, group and region. A positive
        weight means higher activity favours the positive class.

    Raises:
        ValueError: If `label` is unknown, `folds` is below 2, `repeats` is below 1, `min_trials` is below
            `folds`, or the trials cannot be grouped by `sdt_type`.

    Example:
        >>> perievent = pd.read_csv("ses-01_regions_event-cueonset_perievent.csv")
        >>> tables = decoding_table(perievent)
        >>> tables.table.query("decoder == 'lda' and group == 'all'").nlargest(5, "balanced_accuracy")
    """
    if label not in LABELS:
        msg = f"Unknown label {label!r}; expected one of {', '.join(LABELS)}."
        raise ValueError(msg)
    if folds < 2:  # noqa: PLR2004
        msg = f"folds must be at least 2, got {folds}."
        raise ValueError(msg)
    if repeats < 1:
        msg = f"repeats must be at least 1, got {repeats}."
        raise ValueError(msg)
    if min_trials < folds:
        msg = f"min_trials must be at least folds ({folds}), got {min_trials}."
        raise ValueError(msg)

    time, trial_info = pm.perievent_trials(perievent)
    info = trial_info
    if trials is not None:
        extra_columns = [column for column in trial_info.columns if column not in {"trial_index", "event_time"}]
        info = pm.join_trials(trial_info, trials.drop(columns=extra_columns, errors="ignore"))
    group_skip = pm.group_skip_reason(info)
    if group_skip:
        msg = f"Cannot label the trials: {group_skip}"
        raise ValueError(msg)
    response = response if response is not None else (0.0, float(time[-1]))
    event = event if event is not None else pm.infer_event(info)
    start, end, keep = pc.epoch_bounds(info, event, response, mask_response, min_rt, response_pad)

    regions = list(perievent["region"].unique())
    features = epoch_features(perievent, info["trial_index"].to_numpy(), time, regions, start, end)
    complete, present = usable_trials(features)
    y, labelled = trial_labels(info, label)
    usable = keep & complete & labelled

    _, group_names = LABELS[label]
    groups = pm.trial_groups(info)
    per_group: dict[str, dict[str, typing.Any]] = {}
    for name in group_names:
        used = groups[name] & usable
        counts = {"n_trials": int(used.sum()), "n_class_a": int(y[used].sum()), "n_class_b": int((1 - y[used]).sum())}
        if min(counts["n_class_a"], counts["n_class_b"]) < min_trials:
            per_group[name] = counts
            continue
        per_group[name] = counts | group_decoding(features[used], y[used], regions, present, folds, repeats, seed)

    return DecodingTables(_rows(label, regions, per_group), _weight_rows(label, regions, per_group))


def _rows(label: str, regions: list[str], per_group: dict[str, dict[str, typing.Any]]) -> pd.DataFrame:
    """Table of `TABLE_COLUMNS` from each group's `group_decoding` scores.

    Returns:
        pd.DataFrame: One row per decoder, region and group, in that order, with NaN scores for the groups and
        regions without any.
    """
    import pandas as pd

    rows = []
    for decoder in DECODERS:
        for region in [*regions, POPULATION]:
            for name, values in per_group.items():
                scores = values.get(decoder, {}).get(region, {})
                row: dict[str, typing.Any] = {"decoder": decoder, "label": label, "region": region, "group": name}
                row |= {score: scores.get(score, np.nan) for score in SCORES}
                rows.append(row | {key: values[key] for key in ("n_trials", "n_class_a", "n_class_b")})
    return pd.DataFrame(rows, columns=list(TABLE_COLUMNS))


def _weight_rows(label: str, regions: list[str], per_group: dict[str, dict[str, typing.Any]]) -> pd.DataFrame:
    """Table of `WEIGHT_COLUMNS` from each group's population weights.

    Returns:
        pd.DataFrame: One row per decoder, group and region, in that order, with NaN weights for the groups
        without a fit.
    """
    import pandas as pd

    rows: list[dict[str, typing.Any]] = []
    for decoder in DECODERS:
        for name, values in per_group.items():
            mean, sd = values.get(decoder, {}).get("weights", (np.full(len(regions), np.nan),) * 2)
            for i, region in enumerate(regions):
                row = {"decoder": decoder, "label": label, "group": name, "region": region}
                rows.append(row | {"weight": mean[i], "weight_sd": sd[i]})
    return pd.DataFrame(rows, columns=list(WEIGHT_COLUMNS))
