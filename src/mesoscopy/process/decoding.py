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

"""Trial-level decoding of stimulus or response from region peri-event traces.

Each region, and all regions together, is used to classify the trials of a go/no-go session with a logistic
regression and a linear discriminant decoder under repeated stratified cross-validation, against a label-shuffle
null, over the cue-to-response epoch or in windows sliding over the peri-event window. The one-feature decoders are
fitted in numpy over every region, window, fold and shuffle at once; the population decoder goes through
scikit-learn.
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

# Label-shuffle null of one decoder, in output column order; left out without shuffles.
NULL_SCORES = (
    "shuffle_accuracy",
    "shuffle_accuracy_sd",
    "shuffle_balanced_accuracy",
    "shuffle_balanced_accuracy_sd",
    "shuffle_balanced_accuracy_95",
    "p_value",
)

# Columns of the decoding table, in order.
TABLE_COLUMNS = ("decoder", "label", "region", "group", *SCORES, *NULL_SCORES, "n_trials", "n_class_a", "n_class_b")

# Columns of the rolling table, in order.
ROLLING_COLUMNS = (
    "decoder",
    "label",
    "region",
    "group",
    "time",
    *SCORES,
    *NULL_SCORES,
    "n_trials",
    "n_class_a",
    "n_class_b",
)

# Elements of the per-fit standardised feature tensor processed at once by the vectorised decoders.
_CHUNK_ELEMENTS = 4_000_000

# Columns of the weights table, in order.
WEIGHT_COLUMNS = ("decoder", "label", "group", "region", "weight", "weight_sd")


@dataclasses.dataclass(frozen=True)
class DecodingTables:
    """Tables from `decoding_table`.

    Attributes:
        table (pd.DataFrame): One row per decoder, region and trial group, written to
            `<stem>_label-<label>_decoding.csv`.
        weights (pd.DataFrame): Population decoder weights, one row per decoder, trial group and region, written to
            `<stem>_label-<label>_decoding-weights.csv`.
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


def fold_ids(
    splits: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]], folds: int, n_trials: int
) -> npt.NDArray[np.int64]:
    """Test fold of each trial per repeat.

    Args:
        splits (list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]): Train and test indices from
            `cross_validation_splits`.
        folds (int): Folds per repeat.
        n_trials (int): Trials.

    Returns:
        npt.NDArray[np.int64]: Fold per trial, shape `(n_repeats, n_trials)`.
    """
    ids = np.empty((len(splits) // folds, n_trials), dtype=np.int64)
    for i, (_, test) in enumerate(splits):
        ids[i // folds, test] = i % folds
    return ids


def _standardised(x: npt.NDArray[np.float64], train: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Features standardised by each fit's training mean and SD, as `StandardScaler` does.

    Returns:
        npt.NDArray[np.float64]: Shape `(n_problems, n_folds, n_trials, n_features)`.
    """
    n = train.sum(axis=-1)[:, :, None]
    mean = np.einsum("pfn,nr->pfr", train, x) / n
    centred = x[None, None] - mean[:, :, None, :]
    var = np.einsum("pfn,pfnr->pfr", train, centred**2) / n
    # StandardScaler's rule for a feature that is constant up to rounding.
    eps = np.finfo(np.float64).eps
    sd = np.sqrt(var)
    sd[var <= n * eps * var + (n * mean * eps) ** 2] = 1.0
    return centred / sd[:, :, None, :]


def _test_fold(values: npt.NDArray[np.float64], fold_of: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """Each trial's value from the fit it was held out of.

    Returns:
        npt.NDArray[np.float64]: Shape `(n_problems, n_trials, n_features)` from `(n_problems, n_folds, n_trials,
        n_features)`.
    """
    index = fold_of[:, None, :, None]
    return np.take_along_axis(values, np.broadcast_to(index, (*index.shape[:3], values.shape[-1])), axis=1)[:, 0]


def lda_decisions(
    xs: npt.NDArray[np.float64], positive: npt.NDArray[np.float64], train: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Decision values of one-feature linear discriminants with equal priors, one per problem, fold and feature.

    Matches scikit-learn's `lsqr` solver, with the pooled variance being the mean of the two class variances,
    and shrinkage leaves a single variance unchanged.

    Args:
        xs (npt.NDArray[np.float64]): Standardised features from `_standardised`, shape `(n_problems, n_folds,
            n_trials, n_features)`.
        positive (npt.NDArray[np.float64]): 1 for positive trials, shape `(n_problems, n_trials)`.
        train (npt.NDArray[np.float64]): 1 for training trials, shape `(n_problems, n_folds, n_trials)`.

    Returns:
        npt.NDArray[np.float64]: Decision values of every trial under every fit, same shape as `xs`; positive
        favours the positive class.
    """
    weights = (train * positive[:, None, :], train * (1 - positive)[:, None, :])
    means, variances = [], []
    for weight in weights:
        n = weight.sum(axis=-1)[:, :, None]
        mean = np.einsum("pfn,pfnr->pfr", weight, xs) / n
        means.append(mean)
        variances.append(np.einsum("pfn,pfnr->pfr", weight, (xs - mean[:, :, None, :]) ** 2) / n)
    pooled = 0.5 * (variances[0] + variances[1])
    coef = np.divide(means[0] - means[1], pooled, out=np.zeros_like(pooled), where=pooled > 0)
    intercept = -0.5 * (means[0] + means[1]) * coef
    return xs * coef[:, :, None, :] + intercept[:, :, None, :]


def logistic_decisions(
    xs: npt.NDArray[np.float64],
    positive: npt.NDArray[np.float64],
    train: npt.NDArray[np.float64],
    c: float = 1.0,
    tolerance: float = 1e-8,
    max_iter: int = 100,
) -> npt.NDArray[np.float64]:
    """Decision values of one-feature balanced L2 logistic regressions, one per problem, fold and feature.

    Newton's method with backtracking on `0.5 * w**2 / c + sum(s_i * log(1 + exp(-y_i * (w * x_i + b))))`, the
    objective scikit-learn's `lbfgs` solver minimises, with `s_i` the balanced class weights.

    Args:
        xs (npt.NDArray[np.float64]): Standardised features from `_standardised`, shape `(n_problems, n_folds,
            n_trials, n_features)`.
        positive (npt.NDArray[np.float64]): 1 for positive trials, shape `(n_problems, n_trials)`.
        train (npt.NDArray[np.float64]): 1 for training trials, shape `(n_problems, n_folds, n_trials)`.
        c (float, optional): Inverse L2 strength. Defaults to 1.
        tolerance (float, optional): Largest parameter step at convergence. Defaults to 1e-8.
        max_iter (int, optional): Newton iterations. Defaults to 100.

    Returns:
        npt.NDArray[np.float64]: Decision values of every trial under every fit, same shape as `xs`.
    """
    from scipy.special import expit

    n_train = train.sum(axis=-1, keepdims=True)
    n_positive = (train * positive[:, None, :]).sum(axis=-1, keepdims=True)
    class_weight = np.where(
        positive[:, None, :] > 0, n_train / (2 * n_positive), n_train / (2 * (n_train - n_positive))
    )
    weight = (train * class_weight)[..., None]
    sign = (2 * positive - 1)[:, None, :, None]
    target = positive[:, None, :, None]
    shape = (*xs.shape[:2], xs.shape[3])
    w = np.zeros(shape)
    b = np.zeros(shape)

    def objective(w: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        z = xs * w[:, :, None, :] + b[:, :, None, :]
        return 0.5 * w**2 / c + (weight * np.logaddexp(0, -sign * z)).sum(axis=2)

    value = objective(w, b)
    for _ in range(max_iter):
        z = xs * w[:, :, None, :] + b[:, :, None, :]
        p = expit(z)
        residual = weight * (p - target)
        curvature = weight * p * (1 - p)
        g_w = w / c + (residual * xs).sum(axis=2)
        g_b = residual.sum(axis=2)
        h_ww = 1 / c + (curvature * xs**2).sum(axis=2)
        h_wb = (curvature * xs).sum(axis=2)
        h_bb = curvature.sum(axis=2)
        det = np.maximum(h_ww * h_bb - h_wb**2, np.finfo(float).tiny)
        step_w = (h_bb * g_w - h_wb * g_b) / det
        step_b = (h_ww * g_b - h_wb * g_w) / det
        scale = np.ones(shape)
        for _ in range(20):
            candidate = objective(w - scale * step_w, b - scale * step_b)
            # Allow for rounding in the objective, or the search stalls at the optimum.
            worse = candidate > value * (1 + 1e-12) + 1e-12
            if not worse.any():
                break
            scale[worse] *= 0.5
        w -= scale * step_w
        b -= scale * step_b
        value = objective(w, b) if worse.any() else candidate
        if max(np.abs(scale * step_w).max(), np.abs(scale * step_b).max()) < tolerance:
            break
    return xs * w[:, :, None, :] + b[:, :, None, :]


def region_decisions(
    features: npt.NDArray[np.float64], y: npt.NDArray, fold_of: npt.NDArray[np.int64], decoder: str
) -> npt.NDArray[np.float64]:
    """Held-out decision values of one-feature decoders, for every problem and feature at once.

    A problem is one labelling of the trials with one k-fold assignment; the fits of a fold train on the other
    folds and score the held-out trials.

    Args:
        features (npt.NDArray[np.float64]): Features of shape `(n_trials, n_features)`, one decoder per column.
        y (npt.NDArray): Class per trial per problem, shape `(n_problems, n_trials)`.
        fold_of (npt.NDArray[np.int64]): Test fold per trial per problem, shape `(n_problems, n_trials)`.
        decoder (str): One of `DECODERS`.

    Returns:
        npt.NDArray[np.float64]: Decision values of shape `(n_problems, n_trials, n_features)`.

    Raises:
        ValueError: If `decoder` is not one of `DECODERS`.
    """
    if decoder not in DECODERS:
        msg = f"Unknown decoder {decoder!r}; expected one of {', '.join(DECODERS)}."
        raise ValueError(msg)
    folds = int(fold_of.max()) + 1
    n_trials, n_features = features.shape
    chunk = max(1, _CHUNK_ELEMENTS // (folds * n_trials * n_features))
    decisions = np.empty((len(y), n_trials, n_features))
    for first in range(0, len(y), chunk):
        last = min(first + chunk, len(y))
        fold = fold_of[first:last]
        positive = np.asarray(y[first:last], dtype=np.float64)
        train = (fold[:, None, :] != np.arange(folds)[None, :, None]).astype(np.float64)
        xs = _standardised(features, train)
        fitted = lda_decisions(xs, positive, train) if decoder == "lda" else logistic_decisions(xs, positive, train)
        decisions[first:last] = _test_fold(fitted, fold)
    return decisions


def problem_scores(
    decisions: npt.NDArray[np.float64], y: npt.NDArray, fold_of: npt.NDArray[np.int64]
) -> dict[str, npt.NDArray[np.float64]]:
    """Scores of held-out decision values per problem, fold and feature.

    Args:
        decisions (npt.NDArray[np.float64]): Held-out decision values from `region_decisions`, shape
            `(n_problems, n_trials, n_features)`.
        y (npt.NDArray): Class per trial per problem, shape `(n_problems, n_trials)`.
        fold_of (npt.NDArray[np.int64]): Test fold per trial per problem, shape `(n_problems, n_trials)`.

    Returns:
        dict[str, npt.NDArray[np.float64]]: `accuracy` and `balanced_accuracy` of shape `(n_problems, n_folds,
        n_features)`, and `auroc`, `tp`, `fn`, `fp` and `tn` of shape `(n_problems, n_features)`, the counts over
        each problem's trials.
    """
    from scipy.stats import rankdata

    folds = int(fold_of.max()) + 1
    positive = np.asarray(y, dtype=np.float64)
    negative = 1 - positive
    predicted = (decisions > 0).astype(np.float64)
    correct = 1 - np.abs(predicted - positive[:, :, None])
    in_fold = (fold_of[:, :, None] == np.arange(folds)).astype(np.float64)
    size = in_fold.sum(axis=1)
    n_positive = np.einsum("pn,pnf->pf", positive, in_fold)
    accuracy = np.einsum("pnr,pnf->pfr", correct, in_fold) / size[:, :, None]
    tpr = np.einsum("pnr,pn,pnf->pfr", correct, positive, in_fold) / n_positive[:, :, None]
    tnr = np.einsum("pnr,pn,pnf->pfr", correct, negative, in_fold) / (size - n_positive)[:, :, None]
    tp = np.einsum("pnr,pn->pr", predicted, positive)
    fp = np.einsum("pnr,pn->pr", predicted, negative)
    n1 = positive.sum(axis=1)[:, None]
    n0 = negative.sum(axis=1)[:, None]
    rank_sum = np.einsum("pnr,pn->pr", rankdata(decisions, axis=1), positive)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "auroc": (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0),
        "tp": tp,
        "fn": n1 - tp,
        "fp": fp,
        "tn": n0 - fp,
    }


def null_summary(
    observed: float, null_accuracy: npt.NDArray[np.float64], null_balanced: npt.NDArray[np.float64]
) -> dict[str, float]:
    """The `NULL_SCORES` of one decoder from its label-shuffle scores.

    Args:
        observed (float): Balanced accuracy of the true labels.
        null_accuracy (npt.NDArray[np.float64]): Accuracy per shuffle, shape `(n_shuffles,)`.
        null_balanced (npt.NDArray[np.float64]): Balanced accuracy per shuffle, shape `(n_shuffles,)`.

    Returns:
        dict[str, float]: The mean and SD of both, the 95th percentile of the balanced accuracy, and the one-sided
        `p_value`, the fraction of shuffles scoring at least `observed` with one added to both counts, NaN for a NaN
        `observed`.
    """
    sd = (lambda values: float(np.std(values, ddof=1))) if len(null_balanced) > 1 else (lambda _: float("nan"))
    p_value = (1 + np.sum(null_balanced >= observed)) / (len(null_balanced) + 1) if np.isfinite(observed) else np.nan
    return {
        "shuffle_accuracy": float(np.mean(null_accuracy)),
        "shuffle_accuracy_sd": sd(null_accuracy),
        "shuffle_balanced_accuracy": float(np.mean(null_balanced)),
        "shuffle_balanced_accuracy_sd": sd(null_balanced),
        "shuffle_balanced_accuracy_95": float(np.percentile(null_balanced, 95)),
        "p_value": float(p_value),
    }


def shuffled_labels(
    y: npt.NDArray[np.int64], folds: int, shuffles: int, rng: np.random.Generator
) -> tuple[npt.NDArray[np.int64], list[list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]]]:
    """Label shuffles with a stratified k-fold each.

    Args:
        y (npt.NDArray[np.int64]): Class per trial, shape `(n_trials,)`.
        folds (int): Folds per shuffle.
        shuffles (int): Shuffles.
        rng (np.random.Generator): Random generator for the permutations and the splits.

    Returns:
        tuple[npt.NDArray[np.int64], list[list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]]]: Shuffled
        classes, shape `(n_shuffles, n_trials)`, and the `cross_validation_splits` of each shuffle.
    """
    labels = np.stack([rng.permutation(y) for _ in range(shuffles)]) if shuffles else np.empty((0, len(y)), np.int64)
    splits = [cross_validation_splits(row, folds, 1, int(rng.integers(2**31 - 1))) for row in labels]
    return labels, splits


def _column_summary(observed: dict[str, npt.NDArray[np.float64]], column: int) -> dict[str, float]:
    """The `SCORES` of one feature column of `problem_scores`.

    Returns:
        dict[str, float]: `score_summary` over the column's folds and problems.
    """
    return score_summary(
        {
            "accuracy": observed["accuracy"][:, :, column].ravel(),
            "balanced_accuracy": observed["balanced_accuracy"][:, :, column].ravel(),
            "auroc": observed["auroc"][:, column],
            **{name: observed[name][:, column].mean() for name in ("tp", "fn", "fp", "tn")},
        }
    )


def group_decoding(
    features: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    regions: list[str],
    present: npt.NDArray[np.bool_],
    folds: int,
    repeats: int,
    shuffles: int,
    seed: int | None,
) -> dict[str, dict[str, typing.Any]]:
    """Scores of every decoder on each region and on all regions, for one trial group.

    All fits share the same splits, and all shuffles the same permutations, so the regions are compared on the same
    test trials. The one-feature decoders go through `region_decisions`, the population decoder through
    `fit_scores`.

    Args:
        features (npt.NDArray[np.float64]): Features of the group's trials, shape `(n_trials, n_regions)`.
        y (npt.NDArray[np.int64]): Class per trial, shape `(n_trials,)`.
        regions (list[str]): Regions in column order.
        present (npt.NDArray[np.bool_]): Regions with a value, shape `(n_regions,)`; the others are left out.
        folds (int): Folds per repeat.
        repeats (int): Repeats of the k-fold.
        shuffles (int): Label shuffles for the `NULL_SCORES`; 0 leaves them out.
        seed (int | None): Seed for the splits and the shuffles.

    Returns:
        dict[str, dict[str, typing.Any]]: Per decoder, the `score_summary` (and `null_summary`, with shuffles) of
        each region keyed by region and of all regions keyed by `POPULATION`, and `weights`, the population
        weights' mean and SD over the fits per region, each shape `(n_regions,)`.
    """
    rng = np.random.default_rng(seed)
    splits = cross_validation_splits(y, folds, repeats, seed)
    observed_folds = fold_ids(splits, folds, len(y))
    observed_labels = np.broadcast_to(y, (repeats, len(y)))
    null_labels, null_splits = shuffled_labels(y, folds, shuffles, rng)
    null_folds = np.array([fold_ids(s, folds, len(y))[0] for s in null_splits], dtype=np.int64)

    columns = np.flatnonzero(present)
    x = features[:, columns]
    result: dict[str, dict[str, typing.Any]] = {}
    for decoder in DECODERS:
        scores: dict[str, typing.Any] = {}
        observed = problem_scores(
            region_decisions(x, observed_labels, observed_folds, decoder), observed_labels, observed_folds
        )
        null = None
        if shuffles:
            null = problem_scores(region_decisions(x, null_labels, null_folds, decoder), null_labels, null_folds)
        for i, column in enumerate(columns):
            summary = _column_summary(observed, i)
            if null is not None:
                summary |= null_summary(
                    summary["balanced_accuracy"],
                    null["accuracy"][:, :, i].mean(axis=1),
                    null["balanced_accuracy"][:, :, i].mean(axis=1),
                )
            scores[regions[column]] = summary

        population = fit_scores(x, y, splits, decoder, folds)
        summary = score_summary(population)
        if shuffles:
            null_scores = [
                fit_scores(x, row, s, decoder, folds) for row, s in zip(null_labels, null_splits, strict=True)
            ]
            summary |= null_summary(
                summary["balanced_accuracy"],
                np.array([n["accuracy"].mean() for n in null_scores]),
                np.array([n["balanced_accuracy"].mean() for n in null_scores]),
            )
        scores[POPULATION] = summary
        mean = np.full(len(regions), np.nan)
        sd = np.full(len(regions), np.nan)
        mean[columns] = population["weights"].mean(axis=0)
        sd[columns] = population["weights"].std(axis=0, ddof=1) if len(splits) > 1 else np.nan
        scores["weights"] = (mean, sd)
        result[decoder] = scores
    return result


def labelled_trials(
    perievent: pd.DataFrame, trials: pd.DataFrame | None = None
) -> tuple[npt.NDArray[np.float64], pd.DataFrame]:
    """Sample times and per-trial information of a peri-event table whose trials can be labelled.

    Args:
        perievent (pd.DataFrame): Peri-event table, as for `decoding_table`.
        trials (pd.DataFrame | None, optional): Trials table to join by row index, via `metrics.join_trials`, for
            peri-event tables without trials columns. Defaults to None.

    Returns:
        tuple[npt.NDArray[np.float64], pd.DataFrame]: Sorted sample times and one row per trial, from
        `metrics.perievent_trials`, with the trials columns joined.

    Raises:
        ValueError: If the trials cannot be grouped by `sdt_type`, via `metrics.group_skip_reason`.
    """
    time, info = pm.perievent_trials(perievent)
    if trials is not None:
        extra_columns = [column for column in info.columns if column not in {"trial_index", "event_time"}]
        info = pm.join_trials(info, trials.drop(columns=extra_columns, errors="ignore"))
    group_skip = pm.group_skip_reason(info)
    if group_skip:
        msg = f"Cannot label the trials: {group_skip}"
        raise ValueError(msg)
    return time, info


def median_end(
    trial_info: pd.DataFrame, end: npt.NDArray[np.float64], keep: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """Every trial's epoch end set to the median end of the kept trials that responded.

    A common end gives every trial the same epoch, so a class whose trials all end at the median response time, as
    misses and correct rejections do under `connectivity.epoch_bounds`, cannot be told apart by epoch length.

    Args:
        trial_info (pd.DataFrame): One row per trial, with `response_time` in seconds from the cue when known.
        end (npt.NDArray[np.float64]): Per-trial epoch end relative to the event, shape `(n_trials,)`.
        keep (npt.NDArray[np.bool_]): Which trials are kept, shape `(n_trials,)`.

    Returns:
        npt.NDArray[np.float64]: The common end for every trial, shape `(n_trials,)`; `end` unchanged without a
        `response_time` column or a kept responding trial.
    """
    import pandas as pd

    if "response_time" not in trial_info.columns:
        return end
    response_time = pd.to_numeric(trial_info["response_time"], errors="coerce").to_numpy(dtype=np.float64)
    responded = (response_time >= 0) & keep
    return np.full_like(end, np.median(end[responded])) if responded.any() else end


def decoding_table(
    perievent: pd.DataFrame,
    label: str = "stim",
    response: tuple[float, float] | None = None,
    trials: pd.DataFrame | None = None,
    event: str | None = None,
    mask_response: bool = True,
    per_trial_end: bool = False,
    min_rt: float = 0.2,
    response_pad: float = 0.0,
    min_trials: int = 10,
    folds: int = 5,
    repeats: int = 10,
    shuffles: int = 200,
    seed: int | None = 42,
) -> DecodingTables:
    """Decode the stimulus or the response of each trial from region activity, per trial group.

    The feature of each trial is each region's mean over the epoch, which runs from the response window start to
    the median response time of the trials that responded, via `connectivity.epoch_bounds` and `median_end`, so
    every trial has the same epoch. Each region, and all regions together, is scored with every `DECODERS` decoder
    under a repeated stratified k-fold, over all trials and within the groups that hold the other variable
    constant. The chance level comes from fitting the same decoders to shuffled labels, each shuffle under one
    k-fold.

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
        mask_response (bool, optional): End cue-aligned epochs at the response rather than the response window
            end. Defaults to True.
        per_trial_end (bool, optional): End each trial's epoch at its own response instead of the median, as
            `connectivity.epoch_bounds` does. Defaults to False.
        min_rt (float, optional): Trials with a response time below this, in seconds, are dropped. Defaults to
            0.2.
        response_pad (float, optional): Seconds added to the end of each trial's epoch. Defaults to 0.
        min_trials (int, optional): Trials each class needs in a group; groups with fewer get empty scores.
            Defaults to 10.
        folds (int, optional): Folds per repeat. Defaults to 5.
        repeats (int, optional): Repeats of the k-fold, each with a fresh split. Defaults to 10.
        shuffles (int, optional): Label shuffles for the `NULL_SCORES`; 0 leaves the columns out. Defaults to 200.
        seed (int | None, optional): Seed for the splits and the shuffles. Defaults to 42. The options are checked
            by `_check_options` and the trials by `labelled_trials`, which raise `ValueError`.

    Returns:
        DecodingTables: `table` with one row per decoder, region (plus `all` for the population decoder) and
        group, with the `TABLE_COLUMNS`, and `weights` with one row per decoder, group and region. A positive
        weight means higher activity favours the positive class.

    Example:
        >>> perievent = pd.read_csv("ses-01_regions_event-cueonset_perievent.csv")
        >>> tables = decoding_table(perievent)
        >>> tables.table.query("decoder == 'lda' and group == 'all'").nlargest(5, "balanced_accuracy")
    """
    _check_options(label, folds, repeats, shuffles, min_trials)
    time, info = labelled_trials(perievent, trials)
    response = response if response is not None else (0.0, float(time[-1]))
    event = event if event is not None else pm.infer_event(info)
    start, end, keep = pc.epoch_bounds(info, event, response, mask_response, min_rt, response_pad)
    if mask_response and not per_trial_end:
        end = median_end(info, end, keep)

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
        per_group[name] = counts | group_decoding(
            features[used], y[used], regions, present, folds, repeats, shuffles, seed
        )

    return DecodingTables(_rows(label, regions, per_group, shuffles > 0), _weight_rows(label, regions, per_group))


def window_features(
    cube: npt.NDArray[np.float64], starts: npt.NDArray[np.int64], window: int
) -> npt.NDArray[np.float64]:
    """Mean of each region's trace over each window.

    Args:
        cube (npt.NDArray[np.float64]): Traces of shape `(n_trials, n_samples, n_regions)`.
        starts (npt.NDArray[np.int64]): First sample of each window, from `connectivity.window_starts`.
        window (int): Samples per window.

    Returns:
        npt.NDArray[np.float64]: Features of shape `(n_trials, n_windows, n_regions)`, NaN for a window without a
        value.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.stack([np.nanmean(cube[:, start : start + window], axis=1) for start in starts], axis=1)


def _null_scores(
    baseline: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    folds: int,
    shuffles: int,
    rng: np.random.Generator,
    decoder: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Accuracy and balanced accuracy per shuffle of each region and of all regions, on the baseline features.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Each of shape `(n_shuffles, n_regions + 1)`, the
        population last.
    """
    labels, splits = shuffled_labels(y, folds, shuffles, rng)
    fold_of = np.array([fold_ids(s, folds, len(y))[0] for s in splits], dtype=np.int64)
    scores = problem_scores(region_decisions(baseline, labels, fold_of, decoder), labels, fold_of)
    population = [fit_scores(baseline, row, s, decoder, folds) for row, s in zip(labels, splits, strict=True)]
    accuracy = np.column_stack([scores["accuracy"].mean(axis=1), [p["accuracy"].mean() for p in population]])
    balanced = scores["balanced_accuracy"].mean(axis=1)
    return accuracy, np.column_stack([balanced, [p["balanced_accuracy"].mean() for p in population]])


def rolling_decoding(
    features: npt.NDArray[np.float64],
    baseline: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    present: npt.NDArray[np.bool_],
    folds: int,
    repeats: int,
    shuffles: int,
    seed: int | None,
) -> dict[str, list[list[dict[str, float]]]]:
    """Scores of every decoder on each region and on all regions in every window, for one trial group.

    The one-feature decoders of every window and region are fitted at once through `region_decisions` and the
    population decoder is fitted per window through `fit_scores`. The null is taken once on `baseline` and applied
    to every window.

    Args:
        features (npt.NDArray[np.float64]): Window features of the group's trials, shape `(n_trials, n_windows,
            n_regions)`.
        baseline (npt.NDArray[np.float64]): Baseline features of the same trials, shape `(n_trials, n_regions)`.
        y (npt.NDArray[np.int64]): Class per trial, shape `(n_trials,)`.
        present (npt.NDArray[np.bool_]): Regions with a value, shape `(n_regions,)`; the others are left out.
        folds (int): Folds per repeat.
        repeats (int): Repeats of the k-fold.
        shuffles (int): Label shuffles for the `NULL_SCORES`; 0 leaves them out.
        seed (int | None): Seed for the splits and the shuffles.

    Returns:
        dict[str, list[list[dict[str, float]]]]: Per decoder, one list per region in column order, then the
        population, each with one `score_summary` (and `null_summary`, with shuffles) per window; a region without
        a value gets empty dicts.
    """
    rng = np.random.default_rng(seed)
    splits = cross_validation_splits(y, folds, repeats, seed)
    fold_of = fold_ids(splits, folds, len(y))
    labels = np.broadcast_to(y, (repeats, len(y)))
    columns = np.flatnonzero(present)
    n_trials, n_windows, n_regions = features.shape
    x = features[:, :, columns].reshape(n_trials, n_windows * len(columns))
    result: dict[str, list[list[dict[str, float]]]] = {}
    for decoder in DECODERS:
        observed = problem_scores(region_decisions(x, labels, fold_of, decoder), labels, fold_of)
        null = _null_scores(baseline[:, columns], y, folds, shuffles, rng, decoder) if shuffles else None
        per_region: list[list[dict[str, float]]] = [[{} for _ in range(n_windows)] for _ in range(n_regions + 1)]
        for i, column in enumerate(columns):
            for w in range(n_windows):
                summary = _column_summary(observed, w * len(columns) + i)
                if null is not None:
                    summary |= null_summary(summary["balanced_accuracy"], null[0][:, i], null[1][:, i])
                per_region[column][w] = summary
        for w in range(n_windows):
            summary = score_summary(fit_scores(features[:, w, columns], y, splits, decoder, folds))
            if null is not None:
                summary |= null_summary(summary["balanced_accuracy"], null[0][:, -1], null[1][:, -1])
            per_region[n_regions][w] = summary
        result[decoder] = per_region
    return result


def rolling_table(
    perievent: pd.DataFrame,
    label: str = "stim",
    trials: pd.DataFrame | None = None,
    window: float = 0.5,
    step: float = 0.1,
    baseline: tuple[float, float] | None = None,
    min_rt: float = 0.2,
    min_trials: int = 10,
    folds: int = 5,
    repeats: int = 10,
    shuffles: int = 200,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Decode the stimulus or the response in windows sliding over the peri-event window, per trial group.

    The feature of each trial in a window is each region's mean over the window's samples. The windows cover the
    whole peri-event window and `window` and `step` are rounded to whole samples. The chance level is taken once
    per decoder, region and group on the `baseline` features.

    Args:
        perievent (pd.DataFrame): Peri-event table, as for `decoding_table`.
        label (str, optional): `stim` or `response`, as for `decoding_table`. Defaults to `stim`.
        trials (pd.DataFrame | None, optional): Trials table to join by row index. Defaults to None.
        window (float, optional): Window length, in seconds. Defaults to 0.5.
        step (float, optional): Time between the starts of consecutive windows, in seconds. Defaults to 0.1.
        baseline (tuple[float, float] | None, optional): Window `[start, end)` relative to the event whose mean
            gives the null features. Defaults to every sample before the event.
        min_rt (float, optional): Trials with a response time below this, in seconds, are dropped. Defaults to
            0.2.
        min_trials (int, optional): Trials each class needs in a group. Defaults to 10.
        folds (int, optional): Folds per repeat. Defaults to 5.
        repeats (int, optional): Repeats of the k-fold. Defaults to 10.
        shuffles (int, optional): Label shuffles for the `NULL_SCORES`; 0 leaves the columns out. Defaults to 200.
        seed (int | None, optional): Seed for the splits and the shuffles. Defaults to 42.

    Returns:
        pd.DataFrame: One row per decoder, region (plus `all` for the population decoder), group and window, with
        the `ROLLING_COLUMNS`; `time` is the window centre relative to the event.

    Raises:
        ValueError: As for `decoding_table`, or if the baseline window has no sample, or `window` is shorter than a
            sample or longer than the traces.

    Example:
        >>> perievent = pd.read_csv("ses-01_regions_event-cueonset_perievent.csv")
        >>> rolling = rolling_table(perievent)
        >>> rolling.query("decoder == 'lda' and group == 'all'").pivot(index="time", columns="region", values="auroc")
    """
    _check_options(label, folds, repeats, shuffles, min_trials)
    time, info = labelled_trials(perievent, trials)
    keep = pm.kept_trials(info, min_rt)

    regions = list(perievent["region"].unique())
    unbounded = np.full(len(info), np.inf)
    cube = pc.epoch_cube(perievent, info["trial_index"].to_numpy(), time, regions, -unbounded, unbounded)
    interval = float(np.median(np.diff(time))) if len(time) > 1 else 1.0
    window_samples, step_samples = round(window / interval), round(step / interval)
    starts = pc.window_starts(len(time), window_samples, step_samples)
    centres = (time[starts] + time[starts + window_samples - 1]) / 2
    baseline = baseline if baseline is not None else (float(time[0]), 0.0)
    in_baseline = (time >= baseline[0]) & (time < baseline[1])
    if not in_baseline.any():
        msg = f"No sample within the baseline window [{baseline[0]}, {baseline[1]}) for the null."
        raise ValueError(msg)
    features = window_features(cube, starts, window_samples)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        baseline_features = np.nanmean(cube[:, in_baseline], axis=1)
    present = ~np.isnan(features).all(axis=(0, 1)) & ~np.isnan(baseline_features).all(axis=0)
    complete = present.any() & ~np.isnan(features[:, :, present]).any(axis=(1, 2))
    complete &= ~np.isnan(baseline_features[:, present]).any(axis=1)
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
        per_group[name] = counts | rolling_decoding(
            features[used], baseline_features[used], y[used], present, folds, repeats, shuffles, seed
        )
    return _rolling_rows(label, regions, centres, per_group, shuffles > 0)


def _check_options(label: str, folds: int, repeats: int, shuffles: int, min_trials: int) -> None:
    """Check the options shared by `decoding_table` and `rolling_table`.

    Raises:
        ValueError: If `label` is unknown, `folds` is below 2, `repeats` is below 1, `shuffles` is negative, or
            `min_trials` is below `folds`.
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
    if shuffles < 0:
        msg = f"shuffles must be at least 0, got {shuffles}."
        raise ValueError(msg)
    if min_trials < folds:
        msg = f"min_trials must be at least folds ({folds}), got {min_trials}."
        raise ValueError(msg)


def _rolling_rows(
    label: str,
    regions: list[str],
    centres: npt.NDArray[np.float64],
    per_group: dict[str, dict[str, typing.Any]],
    with_null: bool,
) -> pd.DataFrame:
    """Table of `ROLLING_COLUMNS` from each group's `rolling_decoding` scores.

    Returns:
        pd.DataFrame: One row per decoder, region, group and window, in that order, with NaN scores for the groups
        and regions without any, and the `NULL_SCORES` only with `with_null`.
    """
    import pandas as pd

    columns = [column for column in ROLLING_COLUMNS if with_null or column not in NULL_SCORES]
    scores = [*SCORES, *NULL_SCORES]
    blocks = []
    for decoder in DECODERS:
        for index, region in enumerate([*regions, POPULATION]):
            for name, values in per_group.items():
                windows = values[decoder][index] if decoder in values else [{} for _ in centres]
                block = {"decoder": decoder, "label": label, "region": region, "group": name, "time": centres}
                block |= {score: [summary.get(score, np.nan) for summary in windows] for score in scores}
                blocks.append(
                    pd.DataFrame(block | {key: values[key] for key in ("n_trials", "n_class_a", "n_class_b")})
                )
    return pd.concat(blocks, ignore_index=True)[columns]


def _rows(label: str, regions: list[str], per_group: dict[str, dict[str, typing.Any]], with_null: bool) -> pd.DataFrame:
    """Table of `TABLE_COLUMNS` from each group's `group_decoding` scores.

    Returns:
        pd.DataFrame: One row per decoder, region and group, in that order, with NaN scores for the groups and
        regions without any, and the `NULL_SCORES` only with `with_null`.
    """
    import pandas as pd

    columns = [column for column in TABLE_COLUMNS if with_null or column not in NULL_SCORES]
    rows = []
    for decoder in DECODERS:
        for region in [*regions, POPULATION]:
            for name, values in per_group.items():
                scores = values.get(decoder, {}).get(region, {})
                row: dict[str, typing.Any] = {"decoder": decoder, "label": label, "region": region, "group": name}
                row |= {score: scores.get(score, np.nan) for score in [*SCORES, *NULL_SCORES]}
                rows.append(row | {key: values[key] for key in ("n_trials", "n_class_a", "n_class_b")})
    return pd.DataFrame(rows, columns=columns)


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
