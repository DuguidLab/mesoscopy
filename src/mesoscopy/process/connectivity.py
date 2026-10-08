#  Copyright (c) 2022 Constantinos Eleftheriou <Constantinos.Eleftheriou@ed.ac.uk>.
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

"""Pairwise connectivity between regions over the cue-to-response epoch of peri-event traces, or whole traces.

The transfer entropy is the Gaussian estimator, equal to half the Granger log-ratio.
"""

from __future__ import annotations

import typing
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt

from mesoscopy.process import metrics as pm

if typing.TYPE_CHECKING:
    import pandas as pd

# Transfer entropy columns: source `region_a` to target `region_b` and back, with their surrogate z-scores.
TE_METRICS = ("te_ab", "te_ba", "te_ab_z", "te_ba_z")

# Connectivity metrics per region pair, in output column order.
PAIR_METRICS = ("r", "r_residual", "r_trials_avg", "mi", "mi_residual", "partial_r", "lag", "r_lag", *TE_METRICS)

# Columns of the connectivity table, in order; the metrics not estimated are left out.
TABLE_COLUMNS = ("region_a", "region_b", "group", *PAIR_METRICS, "n_trials", "n_samples")

# Trials columns the cue-to-response epoch needs.
EPOCH_COLUMNS = frozenset({"cue_onset", "response_time"})

# Events the cue-to-response epoch is defined for.
EPOCH_EVENTS = frozenset({"cue_onset", "trial_start", "response"})


def pooled_samples(epochs: npt.NDArray) -> npt.NDArray[np.float64]:
    """Samples where every region has a value, pooled over trials.

    Regions without a value anywhere are left out of the check and stay NaN.

    Args:
        epochs (npt.NDArray): Masked traces of shape `(n_trials, n_samples, n_regions)`, NaN outside each trial's
            epoch.

    Returns:
        npt.NDArray[np.float64]: Pooled samples, shape `(n_pooled, n_regions)`.
    """
    values = np.asarray(epochs, dtype=np.float64).reshape(-1, np.shape(epochs)[-1])
    present = ~np.isnan(values).all(axis=0)
    complete = ~np.isnan(values[:, present]).any(axis=1)
    return values[complete]


def correlation_matrix(samples: npt.NDArray) -> npt.NDArray[np.float64]:
    """Pearson correlation between every pair of columns.

    Args:
        samples (npt.NDArray): Samples of shape `(n_samples, n_regions)`.

    Returns:
        npt.NDArray[np.float64]: Correlation matrix, shape `(n_regions, n_regions)`. NaN for constant or NaN
        columns, and everywhere with fewer than two samples.
    """
    values = np.asarray(samples, dtype=np.float64)
    n_regions = values.shape[1]
    if values.shape[0] < 2:  # noqa: PLR2004
        return np.full((n_regions, n_regions), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.atleast_2d(np.corrcoef(values, rowvar=False))


def residual_epochs(epochs: npt.NDArray) -> npt.NDArray[np.float64]:
    """Epochs minus the mean over trials at each sample, per region.

    Args:
        epochs (npt.NDArray): Masked traces of shape `(n_trials, n_samples, n_regions)`.

    Returns:
        npt.NDArray[np.float64]: Residuals, same shape. NaN where the epoch is.
    """
    values = np.asarray(epochs, dtype=np.float64)
    with warnings.catch_warnings():
        # Samples no trial has give NaN, which is the answer.
        warnings.simplefilter("ignore", RuntimeWarning)
        return values - np.nanmean(values, axis=0, keepdims=True)


def trial_correlation(epochs: npt.NDArray, min_samples: int = 3) -> npt.NDArray[np.float64]:
    """Mean over trials of each pair's within-trial Pearson correlation, averaged as Fisher z.

    Args:
        epochs (npt.NDArray): Masked traces of shape `(n_trials, n_samples, n_regions)`.
        min_samples (int, optional): Samples a trial needs to count. Defaults to 3.

    Returns:
        npt.NDArray[np.float64]: Mean correlation, shape `(n_regions, n_regions)`. NaN where no trial has one;
        trials with a correlation of exactly +-1 are skipped.
    """
    values = np.asarray(epochs, dtype=np.float64)
    n_trials, _, n_regions = values.shape
    z = np.full((n_trials, n_regions, n_regions), np.nan)
    for i in range(n_trials):
        samples = pooled_samples(values[i : i + 1])
        if samples.shape[0] < min_samples:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            z[i] = np.arctanh(correlation_matrix(samples))
    z[~np.isfinite(z)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.tanh(np.nanmean(z, axis=0))


def partial_correlation(samples: npt.NDArray) -> npt.NDArray[np.float64]:
    """Partial correlation between every pair of columns given all the others.

    Args:
        samples (npt.NDArray): Samples of shape `(n_samples, n_regions)`.

    Returns:
        npt.NDArray[np.float64]: Partial correlation matrix, shape `(n_regions, n_regions)`, from the
        pseudo-inverse of the covariance. NaN for constant or NaN columns, and everywhere with fewer than two
        samples or columns.
    """
    values = np.asarray(samples, dtype=np.float64)
    n_regions = values.shape[1]
    partial = np.full((n_regions, n_regions), np.nan)
    present = np.flatnonzero(~np.isnan(values).any(axis=0))
    if values.shape[0] < 2 or present.size < 2:  # noqa: PLR2004
        return partial
    precision = np.linalg.pinv(np.cov(values[:, present], rowvar=False))
    precision = (precision + precision.T) / 2
    scale = np.sqrt(np.diag(precision))
    with np.errstate(divide="ignore", invalid="ignore"):
        block = -precision / np.outer(scale, scale)
    np.fill_diagonal(block, 1.0)
    partial[np.ix_(present, present)] = block
    return partial


def mutual_information(x: npt.NDArray, y: npt.NDArray, k: int = 3) -> float:
    """Mutual information between two continuous variables, in bits, by the Kraskov-Stögbauer-Grassberger estimator.

    Each variable is scaled to unit SD first, as the estimator depends on the relative scale of the two. Samples at
    exactly the distance of the `k`-th neighbour count as outside it.

    Args:
        x (npt.NDArray): Samples, shape `(n_samples,)`.
        y (npt.NDArray): Samples, shape `(n_samples,)`.
        k (int, optional): Neighbours. Defaults to 3.

    Returns:
        float: The estimate, which can fall slightly below zero for independent variables. NaN with `k` or fewer
        samples, or when either variable is constant or has NaNs.
    """
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    n = x.size
    if n <= k or np.isnan(x).any() or np.isnan(y).any():
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        x /= x.std()
        y /= y.std()
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return float("nan")
    points = np.column_stack([x, y])
    eps = cKDTree(points).query(points, k=k + 1, p=np.inf)[0][:, -1]
    total = 0.0
    for values in (x, y):
        ordered = np.sort(values)
        inside = np.searchsorted(ordered, values + eps, "left") - np.searchsorted(ordered, values - eps, "right") - 1
        total += np.mean(digamma(np.maximum(inside, 0) + 1))
    return float((digamma(k) + digamma(n) - total) / np.log(2))


def mutual_information_matrix(samples: npt.NDArray, k: int = 3) -> npt.NDArray[np.float64]:
    """`mutual_information` between every pair of columns, over threads.

    Args:
        samples (npt.NDArray): Samples of shape `(n_samples, n_regions)`.
        k (int, optional): Neighbours. Defaults to 3.

    Returns:
        npt.NDArray[np.float64]: Symmetric matrix of shape `(n_regions, n_regions)`, NaN on the diagonal.
    """
    values = np.asarray(samples, dtype=np.float64)
    n_regions = values.shape[1]
    matrix = np.full((n_regions, n_regions), np.nan)
    pairs = [(i, j) for i in range(n_regions) for j in range(i + 1, n_regions)]

    def estimate(pair: tuple[int, int]) -> float:
        return mutual_information(values[:, pair[0]], values[:, pair[1]], k)

    with ThreadPoolExecutor() as pool:
        for (i, j), value in zip(pairs, pool.map(estimate, pairs), strict=True):
            matrix[i, j] = matrix[j, i] = value
    return matrix


def lagged_correlation(epochs: npt.NDArray, max_lag: int) -> npt.NDArray[np.float64]:
    """Pearson correlation between each region at `t` and every region at `t + lag`, within trials.

    Args:
        epochs (npt.NDArray): Masked traces of shape `(n_trials, n_samples, n_regions)`.
        max_lag (int): Largest lag, in samples.

    Returns:
        npt.NDArray[np.float64]: Correlations of shape `(2 * max_lag + 1, n_regions, n_regions)`, for lags
        `-max_lag` to `max_lag`; entry `[l, i, j]` correlates region `i` at `t` with region `j` at `t + lag`,
        over the pooled samples where both are within the epoch. NaN where fewer than two samples remain.
    """
    values = np.asarray(epochs, dtype=np.float64)
    n_samples, n_regions = values.shape[1:]
    lagged = np.full((2 * max_lag + 1, n_regions, n_regions), np.nan)
    for index, lag in enumerate(range(-max_lag, max_lag + 1)):
        shift = abs(lag)
        if shift >= n_samples:
            continue
        early, late = values[:, : n_samples - shift], values[:, shift:]
        pairs = pooled_samples(np.concatenate([early, late] if lag >= 0 else [late, early], axis=-1))
        lagged[index] = correlation_matrix(pairs)[:n_regions, n_regions:]
    return lagged


def peak_lag(lagged: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Lag of the largest absolute correlation per pair, and the correlation there.

    Args:
        lagged (npt.NDArray[np.float64]): Correlations per lag from `lagged_correlation`, shape
            `(n_lags, n_regions, n_regions)`.

    Returns:
        tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]: Index of the peak lag, the first on ties, and the
        signed correlation at it, each shape `(n_regions, n_regions)`. The correlation is NaN, and the index 0,
        where every lag is NaN.
    """
    magnitude = np.where(np.isnan(lagged), -np.inf, np.abs(lagged))
    index = np.argmax(magnitude, axis=0)
    peak = np.take_along_axis(lagged, index[None], axis=0)[0]
    return index, peak


def embedded_samples(
    epochs: npt.NDArray, history: int, lag: int, source: npt.NDArray | None = None
) -> npt.NDArray[np.float64]:
    """Rows of every region now, its past samples and every region `lag` samples back, pooled within trials.

    Args:
        epochs (npt.NDArray): Masked traces of shape `(n_trials, n_samples, n_regions)`.
        history (int): Past samples of each region to embed.
        lag (int): Samples the source columns look back.
        source (npt.NDArray | None, optional): Traces the source columns are taken from, same shape as `epochs`.
            Defaults to `epochs`.

    Returns:
        npt.NDArray[np.float64]: Samples of shape `(n_rows, n_regions * (history + 2))`, the columns being each
        region at `t`, then each region at `t - 1` through `t - history`, then each source region at `t - lag`.
        Rows with a NaN are dropped.
    """
    values = np.asarray(epochs, dtype=np.float64)
    source = values if source is None else np.asarray(source, dtype=np.float64)
    n_samples = values.shape[1]
    offset = max(history, lag)
    if offset >= n_samples:
        return np.empty((0, values.shape[-1] * (history + 2)))
    columns = [values[:, offset:]]
    columns += [values[:, offset - past : n_samples - past] for past in range(1, history + 1)]
    columns.append(source[:, offset - lag : n_samples - lag])
    rows = np.concatenate(columns, axis=-1).reshape(-1, values.shape[-1] * (history + 2))
    return rows[np.isfinite(rows).all(axis=1)]


def transfer_entropy_from_covariance(covariance: npt.NDArray, history: int) -> npt.NDArray[np.float64]:
    """Gaussian transfer entropy of every directed pair from the covariance of `embedded_samples` columns.

    Args:
        covariance (npt.NDArray): Covariance of shape `(n_columns, n_columns)` over the columns of
            `embedded_samples`.
        history (int): Past samples embedded per region.

    Returns:
        npt.NDArray[np.float64]: Transfer entropy in bits, shape `(n_regions, n_regions)`; entry `[i, j]` is from
        source `i` to target `j`, minus half the log of one minus the squared partial correlation between the
        target now and the source back, given the target's past. NaN on the diagonal and where undefined; a
        source that determines the target exactly gives about 26 bits.
    """
    n_regions = covariance.shape[0] // (history + 2)
    targets = np.arange(n_regions)
    sources = np.arange(n_regions * (history + 1), n_regions * (history + 2))
    past = np.stack([n_regions * step + targets for step in range(1, history + 1)], axis=1)  # (n_regions, history)
    kept = np.concatenate([targets[:, None], np.broadcast_to(sources, (n_regions, n_regions))], axis=1)
    past_past = covariance[past[:, :, None], past[:, None, :]]
    kept_past = covariance[kept[:, :, None], past[:, None, :]]
    kept_kept = covariance[kept[:, :, None], kept[:, None, :]]
    with np.errstate(invalid="ignore", divide="ignore"):
        conditional = kept_kept - kept_past @ np.linalg.pinv(past_past) @ kept_past.transpose(0, 2, 1)
        scale = np.sqrt(np.diagonal(conditional, axis1=1, axis2=2))
        rho = conditional[:, 0, 1:] / (scale[:, :1] * scale[:, 1:])  # [target, source]
        te = -0.5 * np.log2(1.0 - np.clip(rho**2, 0.0, 1.0 - np.finfo(float).eps)).T
    np.fill_diagonal(te, np.nan)
    return te


def transfer_entropy(
    epochs: npt.NDArray, history: int, max_lag: int, source: npt.NDArray | None = None
) -> npt.NDArray[np.float64]:
    """Transfer entropy of every directed pair, the largest over source lags `1` to `max_lag`.

    Args:
        epochs (npt.NDArray): Masked traces of shape `(n_trials, n_samples, n_regions)`.
        history (int): Past samples of the target to condition on.
        max_lag (int): Largest source lag searched, in samples.
        source (npt.NDArray | None, optional): Traces the sources are taken from, for surrogates. Defaults to
            `epochs`.

    Returns:
        npt.NDArray[np.float64]: Transfer entropy in bits, shape `(n_regions, n_regions)`, as
        `transfer_entropy_from_covariance`. NaN where fewer than `history + 3` rows remain at every lag.
    """
    n_regions = np.shape(epochs)[-1]
    best = np.full((n_regions, n_regions), np.nan)
    for lag in range(1, max_lag + 1):
        rows = embedded_samples(epochs, history, lag, source)
        if len(rows) < history + 3:
            continue
        te = transfer_entropy_from_covariance(np.cov(rows, rowvar=False), history)
        best = np.where(np.isnan(best), te, np.fmax(best, te))
    return best


def circular_covariances(traces: npt.NDArray, lags: typing.Iterable[int]) -> dict[int, npt.NDArray[np.float64]]:
    """Circular cross-covariance between every pair of whole traces at the given lags, via FFT.

    Args:
        traces (npt.NDArray): Traces of shape `(n_samples, n_regions)` without NaN.
        lags (typing.Iterable[int]): Lags in samples, any sign.

    Returns:
        dict[int, npt.NDArray[np.float64]]: Per lag `d`, the matrix whose entry `[j, i]` is the mean over `t` of
        region `j` at `t` times region `i` at `t - d`, wrapping around the trace ends.
    """
    values = np.asarray(traces, dtype=np.float64) - np.mean(traces, axis=0)
    n_samples, n_regions = values.shape
    wanted = sorted(set(lags))
    index = np.array(wanted) % n_samples
    spectrum = np.fft.rfft(values, axis=0)
    out = np.empty((len(wanted), n_regions, n_regions))
    for i in range(n_regions):
        full = np.fft.irfft(spectrum * np.conj(spectrum[:, i : i + 1]), n=n_samples, axis=0) / n_samples
        out[:, :, i] = full[index]
    return {lag: out[q] for q, lag in enumerate(wanted)}


def transfer_entropy_circular(
    traces: npt.NDArray, history: int, max_lag: int, shifts: npt.NDArray
) -> npt.NDArray[np.float64]:
    """Transfer entropy of whole traces, with every source rolled by each shift, from circular covariances.

    Args:
        traces (npt.NDArray): Traces of shape `(n_samples, n_regions)` without NaN.
        history (int): Past samples of the target to condition on.
        max_lag (int): Largest source lag searched, in samples.
        shifts (npt.NDArray): Samples to roll the sources by, shape `(n_shifts,)`; 0 is the unshifted estimate.

    Returns:
        npt.NDArray[np.float64]: Transfer entropy in bits, shape `(n_shifts, n_regions, n_regions)`, as
        `transfer_entropy`, differing from it only by the wrap-around at the trace ends.
    """
    n_regions = np.shape(traces)[-1]
    shifts = np.asarray(shifts, dtype=int)
    steps = range(history + 1)
    lags = {lag + shift - step for lag in range(1, max_lag + 1) for shift in shifts for step in steps}
    at = circular_covariances(traces, lags | {b - a for a in steps for b in steps})
    size = n_regions * (history + 2)
    blocks = [slice(n_regions * step, n_regions * (step + 1)) for step in range(history + 2)]
    out = np.full((len(shifts), n_regions, n_regions), np.nan)
    for q, shift in enumerate(shifts):
        for lag in range(1, max_lag + 1):
            covariance = np.empty((size, size))
            for a in steps:
                for b in steps:
                    covariance[blocks[a], blocks[b]] = at[b - a]
                covariance[blocks[a], blocks[-1]] = at[lag + shift - a]
                covariance[blocks[-1], blocks[a]] = at[lag + shift - a].T
            covariance[blocks[-1], blocks[-1]] = at[0]
            te = transfer_entropy_from_covariance(covariance, history)
            out[q] = np.where(np.isnan(out[q]), te, np.fmax(out[q], te))
    return out


def surrogate_z(value: npt.NDArray, surrogates: npt.NDArray) -> npt.NDArray[np.float64]:
    """Z-score of a matrix against its surrogates.

    Args:
        value (npt.NDArray): Matrix of shape `(n_regions, n_regions)`.
        surrogates (npt.NDArray): Surrogate matrices of shape `(n_surrogates, n_regions, n_regions)`.

    Returns:
        npt.NDArray[np.float64]: `(value - mean) / sd` over the surrogates, NaN without at least two of them or
        where their spread is zero.
    """
    if len(surrogates) < 2:  # noqa: PLR2004
        return np.full(np.shape(value), np.nan)
    spread = np.std(surrogates, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(spread > 0, (value - np.mean(surrogates, axis=0)) / spread, np.nan)


def epoch_transfer_entropy(
    epochs: npt.NDArray, history: int, max_lag: int, n_surrogates: int, rng: np.random.Generator
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Transfer entropy of epochs and its z-score against trial-shuffled sources.

    Args:
        epochs (npt.NDArray): Masked traces of shape `(n_trials, n_samples, n_regions)`.
        history (int): Past samples of the target to condition on.
        max_lag (int): Largest source lag searched, in samples.
        n_surrogates (int): Surrogates, each pairing the targets with the sources of a permutation of the trials.
        rng (np.random.Generator): Random generator for the permutations.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Transfer entropy and its `surrogate_z`, each
        shape `(n_regions, n_regions)`.
    """
    te = transfer_entropy(epochs, history, max_lag)
    n_trials = np.shape(epochs)[0]
    surrogates = (
        np.stack(
            [
                transfer_entropy(epochs, history, max_lag, np.asarray(epochs)[rng.permutation(n_trials)])
                for _ in range(n_surrogates)
            ]
        )
        if n_surrogates
        else np.empty((0, *te.shape))
    )
    return te, surrogate_z(te, surrogates)


def trace_transfer_entropy(
    traces: npt.NDArray, history: int, max_lag: int, n_surrogates: int, rng: np.random.Generator
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Transfer entropy of whole traces and its z-score against circularly shifted sources.

    Args:
        traces (npt.NDArray): Traces of shape `(n_samples, n_regions)` without NaN.
        history (int): Past samples of the target to condition on.
        max_lag (int): Largest source lag searched, in samples.
        n_surrogates (int): Surrogates, each rolling the sources by a random shift of at least
            `max_lag + history` samples from either end.
        rng (np.random.Generator): Random generator for the shifts.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Transfer entropy via `transfer_entropy_circular`
        and its `surrogate_z`, each shape `(n_regions, n_regions)`. The z-score is NaN when the trace is too short
        for a shift.
    """
    n_samples, n_regions = np.shape(traces)
    margin = max_lag + history
    if n_samples < margin + 3:
        return np.full((n_regions, n_regions), np.nan), np.full((n_regions, n_regions), np.nan)
    shifts = rng.integers(margin, n_samples - margin, size=n_surrogates) if n_samples > 2 * margin else np.empty(0)
    estimates = transfer_entropy_circular(traces, history, max_lag, np.array([0, *shifts], dtype=int))
    return estimates[0], surrogate_z(estimates[0], estimates[1:])


def directed_metrics(te: npt.NDArray, z: npt.NDArray | None = None) -> dict[str, npt.NDArray[np.float64]]:
    """The `TE_METRICS` matrices, indexed `[a, b]` like the symmetric ones, from a directed `[source, target]` pair.

    Args:
        te (npt.NDArray): Transfer entropy, shape `(n_regions, n_regions)`.
        z (npt.NDArray | None, optional): Its z-score, same shape. Defaults to None, leaving the z-scores out.

    Returns:
        dict[str, npt.NDArray[np.float64]]: `te_ab` and `te_ba`, plus `te_ab_z` and `te_ba_z` given `z`.
    """
    metrics = {"te_ab": np.asarray(te), "te_ba": np.asarray(te).T}
    if z is not None:
        metrics |= {"te_ab_z": np.asarray(z), "te_ba_z": np.asarray(z).T}
    return metrics


def epoch_bounds(
    trial_info: pd.DataFrame,
    event: str | None,
    response: tuple[float, float],
    mask_response: bool = True,
    min_rt: float = 0.2,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Per-trial epoch `[start, end)` relative to the event, from `metrics.reliability_epochs`.

    Falls back to the response window for every trial, with a warning, when the trials columns the epoch needs
    are missing, the aligned event is unknown, or the windows are reward-aligned.

    Args:
        trial_info (pd.DataFrame): One row per trial with `event_time` and, for the cue-to-response epoch,
            `cue_onset` and `response_time` columns.
        event (str | None): Event the windows are aligned to, or None when unknown.
        response (tuple[float, float]): Response window `[start, end]`, in seconds.
        mask_response (bool, optional): End cue-aligned epochs at the response. Defaults to True.
        min_rt (float, optional): Trials with a response time below this are dropped. Defaults to 0.2.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]: Epoch start and end
        relative to the event, and which trials are kept; each shape `(n_trials,)`.
    """
    reason = _epoch_skip_reason(trial_info, event)
    if reason is None:
        start, end, _, keep = pm.reliability_epochs(trial_info, str(event), response, mask_response, min_rt)
        return start, end, keep
    warnings.warn(f"Using the response window as the epoch: {reason}", stacklevel=2)
    n_trials = len(trial_info)
    end = np.nextafter(float(response[1]), np.inf)
    return np.full(n_trials, float(response[0])), np.full(n_trials, end), np.ones(n_trials, dtype=bool)


def _epoch_skip_reason(trial_info: pd.DataFrame, event: str | None) -> str | None:
    """Why the cue-to-response epoch cannot be taken.

    Returns:
        str | None: The reason, or None when it can be.
    """
    missing = EPOCH_COLUMNS - set(trial_info.columns)
    if missing:
        return f"no {', '.join(sorted(missing))} column(s); pass the trials CSV."
    if event is None:
        return "event_time matches no trials column, so the aligned event is unknown."
    if event not in EPOCH_EVENTS:
        return f"not defined for {event}-aligned windows."
    return None


def epoch_cube(
    perievent: pd.DataFrame,
    trial_index: npt.NDArray,
    time: npt.NDArray[np.float64],
    regions: list[str],
    start: npt.NDArray[np.float64],
    end: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Every region's traces, masked to each trial's epoch, as one array.

    Args:
        perievent (pd.DataFrame): Long-format peri-event table with `trial_index`, `time`, `region` and `F`.
        trial_index (npt.NDArray): Trials in row order, shape `(n_trials,)`.
        time (npt.NDArray[np.float64]): Sample times relative to the event, shape `(n_samples,)`.
        regions (list[str]): Regions in column order.
        start (npt.NDArray[np.float64]): Per-trial epoch start, shape `(n_trials,)`.
        end (npt.NDArray[np.float64]): Per-trial epoch end, exclusive, shape `(n_trials,)`.

    Returns:
        npt.NDArray[np.float64]: Masked traces of shape `(n_trials, n_samples, n_regions)`, via
        `metrics.epoch_traces`.
    """
    cube = np.full((len(trial_index), len(time), len(regions)), np.nan)
    for column, region in enumerate(regions):
        traces = (
            perievent[perievent["region"] == region]
            .pivot(index="trial_index", columns="time", values="F")
            .reindex(index=trial_index, columns=time)
            .to_numpy(dtype=np.float64)
        )
        cube[:, :, column] = pm.epoch_traces(traces, time, start, end)
    return cube


def pooled_metrics(
    epochs: npt.NDArray,
    lag_samples: int,
    mi_neighbours: int = 3,
    mutual_info: bool = True,
) -> dict[str, typing.Any]:
    """Connectivity matrices that need no trial structure, over the pooled samples.

    Args:
        epochs (npt.NDArray): Masked traces of shape `(n_trials, n_samples, n_regions)`.
        lag_samples (int): Largest lag searched for the peak cross-correlation, in samples.
        mi_neighbours (int, optional): Neighbours for `mutual_information`. Defaults to 3.
        mutual_info (bool, optional): Estimate `mi`; False leaves it out. Defaults to True.

    Returns:
        dict[str, typing.Any]: `(n_regions, n_regions)` matrices `r`, `partial_r`, `lag` (in samples), `r_lag`
        and, with `mutual_info`, `mi`, plus `n_samples`, the pooled samples used.
    """
    pooled = pooled_samples(epochs)
    index, r_lag = peak_lag(lagged_correlation(epochs, lag_samples))
    metrics = {
        "r": correlation_matrix(pooled),
        "partial_r": partial_correlation(pooled),
        "lag": np.where(np.isnan(r_lag), np.nan, index - lag_samples),
        "r_lag": r_lag,
        "n_samples": pooled.shape[0],
    }
    if mutual_info:
        metrics["mi"] = mutual_information_matrix(pooled, mi_neighbours)
    return metrics


def pair_metrics(
    epochs: npt.NDArray,
    lag_samples: int,
    min_trial_samples: int = 3,
    mi_neighbours: int = 3,
    mutual_info: bool = True,
    te_history: int = 1,
    te_surrogates: int = 200,
    transfer_entropy: bool = False,
    rng: np.random.Generator | None = None,
) -> dict[str, typing.Any]:
    """Connectivity matrices of one trial group.

    Args:
        epochs (npt.NDArray): Masked traces of shape `(n_trials, n_samples, n_regions)`.
        lag_samples (int): Largest lag searched for the peak cross-correlation and transfer entropy, in samples.
        min_trial_samples (int, optional): Samples a trial needs for `trial_correlation`. Defaults to 3.
        mi_neighbours (int, optional): Neighbours for `mutual_information`. Defaults to 3.
        mutual_info (bool, optional): Estimate `mi` and `mi_residual`; False leaves them out. Defaults to True.
        te_history (int, optional): Target past samples for `epoch_transfer_entropy`. Defaults to 1.
        te_surrogates (int, optional): Trial-shuffle surrogates for the transfer entropy z-scores; 0 leaves them
            out. Defaults to 200.
        transfer_entropy (bool, optional): Estimate the `TE_METRICS` on the residual epochs; False leaves them
            out. Defaults to False.
        rng (np.random.Generator | None, optional): Random generator for the surrogates. Defaults to a fresh one.

    Returns:
        dict[str, typing.Any]: One `(n_regions, n_regions)` matrix per `PAIR_METRICS` name estimated, with `lag`
        in samples, plus `n_samples`, the pooled samples used.
    """
    residual = residual_epochs(epochs)
    residuals = pooled_samples(residual)
    metrics = pooled_metrics(epochs, lag_samples, mi_neighbours, mutual_info) | {
        "r_residual": correlation_matrix(residuals),
        "r_trials_avg": trial_correlation(epochs, min_trial_samples),
    }
    if mutual_info:
        metrics["mi_residual"] = mutual_information_matrix(residuals, mi_neighbours)
    if not transfer_entropy:
        return metrics
    rng = rng if rng is not None else np.random.default_rng()
    te, z = epoch_transfer_entropy(residual, te_history, lag_samples, te_surrogates, rng)
    return metrics | directed_metrics(te, z if te_surrogates else None)


def trace_samples(regions: pd.DataFrame) -> tuple[npt.NDArray[np.float64], list[str], float]:
    """Whole traces of a long-format regions table, one column per region in order of appearance.

    Args:
        regions (pd.DataFrame): Regions table with `region`, `F` and `timestamp` or `time_aligned` columns, as
            written by `process regions`.

    Returns:
        tuple[npt.NDArray[np.float64], list[str], float]: Traces of shape `(n_frames, n_regions)` in time order,
        the region names, and the median frame interval in seconds.

    Raises:
        ValueError: If a column is missing.
    """
    from mesoscopy.process import regression as regr

    missing = {"region", "F"} - set(regions.columns)
    if "timestamp" not in regions.columns and "time_aligned" not in regions.columns:
        missing.add("timestamp")
    if missing:
        msg = f"Missing columns: {sorted(missing)}."
        raise ValueError(msg)
    names = list(regions["region"].unique())
    if "time_aligned" in regions.columns:
        wide = regions.pivot(index="time_aligned", columns="region", values="F")
        seconds = wide.index.to_numpy(dtype=np.float64)
    else:
        wide = regions.pivot(index="timestamp", columns="region", values="F")
        seconds = regr.elapsed_seconds(wide.index.to_numpy(dtype=str))
    order = np.argsort(seconds, kind="stable")
    step = float(np.median(np.diff(seconds[order]))) if len(seconds) > 1 else 1.0
    return wide[names].to_numpy(dtype=np.float64)[order], names, step


def trace_table(
    regions: pd.DataFrame,
    max_lag: float = 0.5,
    mi_neighbours: int = 3,
    mutual_info: bool = True,
    te_history: int = 1,
    te_surrogates: int = 200,
    transfer_entropy: bool = False,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Pairwise connectivity between regions over their whole traces, from a long-format regions table.

    Args:
        regions (pd.DataFrame): Regions table, as for `trace_samples`.
        max_lag (float, optional): Largest lag searched for the peak cross-correlation and transfer entropy, in
            seconds. Defaults to 0.5.
        mi_neighbours (int, optional): Neighbours for `mutual_information`. Defaults to 3.
        mutual_info (bool, optional): Estimate `mi`; False leaves it out. Defaults to True.
        te_history (int, optional): Target past samples for `trace_transfer_entropy`. Defaults to 1.
        te_surrogates (int, optional): Circular-shift surrogates for the transfer entropy z-scores; 0 leaves them
            out. Defaults to 200.
        transfer_entropy (bool, optional): Estimate the `TE_METRICS`; False leaves them out. Defaults to False.
        seed (int | None, optional): Seed for the surrogates. Defaults to 42.

    Returns:
        pd.DataFrame: One `all` row per unordered region pair with the `TABLE_COLUMNS` estimated, as for
        `connectivity_table`. `r_residual`, `r_trials_avg` and `mi_residual` have no meaning without trials and
        are left out, `n_trials` is NaN, and `n_samples` is the frames where every region is finite. The transfer
        entropy is taken over those frames with the wrap-around of `transfer_entropy_circular`.

    Example:
        >>> regions = pd.read_csv("ses-01_regions.csv")
        >>> table = trace_table(regions)
        >>> table.nlargest(5, "partial_r")
    """
    traces, names, step = trace_samples(regions)
    lag_samples = round(max_lag / step)
    metrics = pooled_metrics(traces[None], lag_samples, mi_neighbours, mutual_info)
    metrics["lag"] *= step
    if transfer_entropy:
        rng = np.random.default_rng(seed)
        te, z = trace_transfer_entropy(pooled_samples(traces[None]), te_history, lag_samples, te_surrogates, rng)
        metrics |= directed_metrics(te, z if te_surrogates else None)
    return _rows(names, {"all": {"n_trials": np.nan, **metrics}})


def connectivity_table(
    perievent: pd.DataFrame,
    response: tuple[float, float] | None = None,
    trials: pd.DataFrame | None = None,
    event: str | None = None,
    mask_response: bool = True,
    min_rt: float = 0.2,
    min_trials: int = 10,
    max_lag: float = 0.5,
    min_trial_samples: int = 3,
    mi_neighbours: int = 3,
    mutual_info: bool = True,
    te_history: int = 1,
    te_surrogates: int = 200,
    transfer_entropy: bool = False,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Pairwise connectivity between regions, per trial group, from a long-format peri-event table.

    Each metric is taken over the samples of each trial's epoch from `epoch_bounds`, pooled over the group's
    trials. For go/no-go sessions with an `sdt_type` column there is a row per pair for each `trial_groups` group;
    other sessions get the `all` rows only, with a warning.

    Args:
        perievent (pd.DataFrame): Peri-event table with `trial_index`, `event_time`, `time`, `region` and `F`
            columns, as written by `process peri-event` from a `_regions.csv`, plus any per-trial columns; checked
            by `metrics.perievent_trials`.
        response (tuple[float, float] | None, optional): Response window `[start, end]`. Defaults to all
            post-event samples.
        trials (pd.DataFrame | None, optional): Trials table to join by row index, via `metrics.join_trials`, for
            peri-event tables without trials columns. Defaults to None.
        event (str | None, optional): Event the windows are aligned to, one of `perievent.EVENTS`. Defaults to
            the event matching `event_time`, via `metrics.infer_event`.
        mask_response (bool, optional): End cue-aligned epochs at each trial's response. Defaults to True.
        min_rt (float, optional): Trials with a response time below this, in seconds, are dropped. Defaults to
            0.2.
        min_trials (int, optional): Groups with fewer trials get NaN metrics; the all-trials metrics are always
            taken. Defaults to 10.
        max_lag (float, optional): Largest lag searched for the peak cross-correlation and transfer entropy, in
            seconds. Defaults to 0.5.
        min_trial_samples (int, optional): Samples a trial needs to count towards `r_trials_avg`. Defaults to 3.
        mi_neighbours (int, optional): Neighbours for `mutual_information`. Defaults to 3.
        mutual_info (bool, optional): Estimate `mi` and `mi_residual`; False leaves them out. Defaults to True.
        te_history (int, optional): Target past samples for `epoch_transfer_entropy`. Defaults to 1.
        te_surrogates (int, optional): Trial-shuffle surrogates per group for the transfer entropy z-scores; 0
            leaves them out. Defaults to 200.
        transfer_entropy (bool, optional): Estimate the `TE_METRICS` on the residual epochs; False leaves them
            out. Defaults to False.
        seed (int | None, optional): Seed for the surrogates. Defaults to 42.

    Returns:
        pd.DataFrame: One row per unordered region pair per trial group, with the `TABLE_COLUMNS` estimated:
        `region_a` and
        `region_b` in the order the regions appear, `group`, `r` (Pearson correlation of the pooled samples),
        `r_residual` (the same after removing the group's mean response at each sample), `r_trials_avg`
        (`trial_correlation`), `mi` and `mi_residual` (`mutual_information` of the pooled samples and of the
        residuals, in bits), `partial_r` (`partial_correlation`), `lag` and `r_lag` (lag in seconds and value
        of the peak absolute cross-correlation within `max_lag`; positive when `region_b` lags `region_a`),
        `te_ab` and `te_ba` (transfer entropy of the residual epochs from `region_a` to `region_b` and back, in
        bits, the largest over source lags within `max_lag`), `te_ab_z` and `te_ba_z` (their z-scores against
        trial-shuffled sources), `n_trials` (trials used) and `n_samples` (pooled samples used).

    Example:
        >>> perievent = pd.read_csv("ses-01_regions_event-cueonset_perievent.csv")
        >>> table = connectivity_table(perievent)
        >>> table[table["group"] == "stim-go"].nlargest(5, "r_residual")
    """
    time, trial_info = pm.perievent_trials(perievent)
    info = trial_info
    if trials is not None:
        extra_columns = [column for column in trial_info.columns if column not in {"trial_index", "event_time"}]
        info = pm.join_trials(trial_info, trials.drop(columns=extra_columns, errors="ignore"))
    response = response if response is not None else (0.0, float(time[-1]))
    event = event if event is not None else pm.infer_event(info)
    start, end, keep = epoch_bounds(info, event, response, mask_response, min_rt)
    group_skip = pm.group_skip_reason(info)
    if group_skip:
        warnings.warn(f"Skipping trial groups: {group_skip}", stacklevel=2)
    groups = {"all": np.ones(len(info), dtype=bool)} if group_skip else pm.trial_groups(info)

    regions = list(perievent["region"].unique())
    cube = epoch_cube(perievent, info["trial_index"].to_numpy(), time, regions, start, end)
    step = float(np.median(np.diff(time))) if len(time) > 1 else 1.0
    lag_samples = round(max_lag / step)

    rng = np.random.default_rng(seed)
    per_group: dict[str, dict[str, typing.Any]] = {}
    for name, mask in groups.items():
        used = mask & keep
        n_trials = int(used.sum())
        if name != "all" and n_trials < min_trials:
            per_group[name] = {"n_trials": n_trials, "n_samples": 0}
            continue
        metrics = pair_metrics(
            cube[used],
            lag_samples,
            min_trial_samples,
            mi_neighbours,
            mutual_info,
            te_history,
            te_surrogates,
            transfer_entropy,
            rng,
        )
        metrics["lag"] *= step
        per_group[name] = {"n_trials": n_trials, **metrics}

    return _rows(regions, per_group)


def _rows(regions: list[str], per_group: dict[str, dict[str, typing.Any]]) -> pd.DataFrame:
    """Table of `TABLE_COLUMNS` from each group's metric matrices.

    Args:
        regions (list[str]): Region names in order.
        per_group (dict[str, dict[str, typing.Any]]): Per group, its `pair_metrics` matrices and `n_trials`;
            metrics a group lacks are NaN.

    Returns:
        pd.DataFrame: One row per unordered region pair and group, with the `TABLE_COLUMNS` some group has.
    """
    import pandas as pd

    metrics = [metric for metric in PAIR_METRICS if any(metric in values for values in per_group.values())]
    rows = []
    for a, region_a in enumerate(regions):
        for b in range(a + 1, len(regions)):
            for name, values in per_group.items():
                row: dict[str, typing.Any] = {"region_a": region_a, "region_b": regions[b], "group": name}
                row |= {metric: values[metric][a, b] if metric in values else np.nan for metric in metrics}
                rows.append(row | {"n_trials": values["n_trials"], "n_samples": values["n_samples"]})
    return pd.DataFrame(rows, columns=["region_a", "region_b", "group", *metrics, "n_trials", "n_samples"])
