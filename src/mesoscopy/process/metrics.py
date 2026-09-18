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

"""Per-trial response metrics and per-session variability from peri-event traces."""

from __future__ import annotations

import typing
import warnings

import numpy as np
import numpy.typing as npt

if typing.TYPE_CHECKING:
    import pandas as pd

# Per-trial metrics summarised per session, in output column order.
METRICS = ("onset_time", "peak_time", "amplitude", "auc", "decay_time", "offset_time", "duration")

# Columns of the long-format peri-event table written by `process peri-event`.
PERIEVENT_COLUMNS = frozenset({"trial_index", "event_time", "time", "region", "F"})

# Onset methods: `sd` thresholds at a multiple of the baseline SD, `peak` at a fraction of the amplitude, and
# `extrapolate` fits a line to the rise and takes where it crosses the baseline.
ONSET_METHODS = ("sd", "peak", "extrapolate")


def window_mask(
    time: npt.NDArray[np.float64], start: float, end: float, *, inclusive_end: bool = True
) -> npt.NDArray[np.bool_]:
    """Boolean mask over `time` for `start <= t <= end`, or `start <= t < end` when `inclusive_end` is False.

    Args:
        time (npt.NDArray[np.float64]): Sample times relative to the event, shape `(n_samples,)`.
        start (float): Window start, in seconds.
        end (float): Window end, in seconds.
        inclusive_end (bool, optional): Whether `end` is included. Defaults to True.

    Returns:
        npt.NDArray[np.bool_]: Mask of shape `(n_samples,)`.

    Raises:
        ValueError: If no sample falls within the window.
    """
    mask = (time >= start) & ((time <= end) if inclusive_end else (time < end))
    if not mask.any():
        bracket = "]" if inclusive_end else ")"
        msg = f"No samples fall within the window [{start}, {end}{bracket}."
        raise ValueError(msg)
    return mask


def baseline_stats(
    traces: npt.NDArray, time: npt.NDArray[np.float64], start: float, end: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-trial mean and standard deviation over `start <= t < end`.

    Args:
        traces (npt.NDArray): Traces of shape `(n_trials, n_samples)`.
        time (npt.NDArray[np.float64]): Sample times relative to the event, shape `(n_samples,)`.
        start (float): Baseline start, in seconds.
        end (float): Baseline end, in seconds. Exclusive.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Baseline mean and SD, each shape `(n_trials,)`.
    """
    window = window_mask(time, start, end, inclusive_end=False)
    baseline = np.asarray(traces, dtype=np.float64)[:, window]
    with warnings.catch_warnings():
        # All-NaN trials give NaN, which is the answer.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(baseline, axis=1), np.nanstd(baseline, axis=1, ddof=1)


def smooth(traces: npt.NDArray, window: int) -> npt.NDArray[np.float64]:
    """Centred moving average of each trace, ignoring NaNs and shrinking the window at the edges.

    Args:
        traces (npt.NDArray): Traces of shape `(n_trials, n_samples)`.
        window (int): Window length in samples; odd. 1 returns the traces unchanged.

    Returns:
        npt.NDArray[np.float64]: Smoothed traces, same shape. NaN where every sample in the window is NaN.

    Raises:
        ValueError: If `window` is not a positive odd integer.
    """
    if window < 1 or window % 2 == 0:
        msg = f"window must be a positive odd integer, got {window}."
        raise ValueError(msg)

    values = np.asarray(traces, dtype=np.float64)
    if window == 1:
        return values.copy()

    half = window // 2
    padded = np.pad(values, ((0, 0), (half, half)), constant_values=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.lib.stride_tricks.sliding_window_view(padded, window, axis=1), axis=-1)


def peak(
    traces: npt.NDArray, time: npt.NDArray[np.float64], start: float, end: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Signed maximum of each trace over `start <= t <= end`.

    Args:
        traces (npt.NDArray): Baseline-subtracted traces of shape `(n_trials, n_samples)`.
        time (npt.NDArray[np.float64]): Sample times relative to the event, shape `(n_samples,)`.
        start (float): Response window start, in seconds.
        end (float): Response window end, in seconds.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]: Amplitude, time of the
        peak and its sample index, each shape `(n_trials,)`. Amplitude and time are NaN, and the index -1, for
        trials that are all NaN within the window.
    """
    window = window_mask(time, start, end)
    offset = int(np.flatnonzero(window)[0])
    values = np.asarray(traces, dtype=np.float64)[:, window]
    valid = ~np.all(np.isnan(values), axis=1)

    amplitude = np.full(values.shape[0], np.nan)
    peak_time = np.full(values.shape[0], np.nan)
    index = np.full(values.shape[0], -1, dtype=np.int64)
    if valid.any():
        argmax = np.nanargmax(values[valid], axis=1)
        amplitude[valid] = values[valid][np.arange(int(valid.sum())), argmax]
        index[valid] = argmax + offset
        peak_time[valid] = time[index[valid]]
    return amplitude, peak_time, index


def auc(traces: npt.NDArray, time: npt.NDArray[np.float64], start: float, end: float) -> npt.NDArray[np.float64]:
    """Signed trapezoidal area under each trace over `start <= t <= end`.

    Args:
        traces (npt.NDArray): Baseline-subtracted traces of shape `(n_trials, n_samples)`.
        time (npt.NDArray[np.float64]): Sample times relative to the event, shape `(n_samples,)`.
        start (float): Response window start, in seconds.
        end (float): Response window end, in seconds.

    Returns:
        npt.NDArray[np.float64]: Area per trial, shape `(n_trials,)`. NaN where the trace has NaNs in the window.
    """
    window = window_mask(time, start, end)
    return np.trapezoid(np.asarray(traces, dtype=np.float64)[:, window], time[window], axis=1)


def onset_time(
    traces: npt.NDArray,
    time: npt.NDArray[np.float64],
    threshold: npt.NDArray[np.float64],
    start: float,
    end: float,
    min_samples: int = 3,
) -> npt.NDArray[np.float64]:
    """Time of the first run of `min_samples` consecutive samples above `threshold` within `start <= t <= end`.

    Args:
        traces (npt.NDArray): Baseline-subtracted traces of shape `(n_trials, n_samples)`.
        time (npt.NDArray[np.float64]): Sample times relative to the event, shape `(n_samples,)`.
        threshold (npt.NDArray[np.float64]): Per-trial threshold, shape `(n_trials,)`.
        start (float): Response window start, in seconds.
        end (float): Response window end, in seconds.
        min_samples (int, optional): Consecutive samples required above threshold. Defaults to 3.

    Returns:
        npt.NDArray[np.float64]: Onset time per trial, shape `(n_trials,)`. NaN where no such run exists or the
        threshold is NaN.

    Raises:
        ValueError: If `min_samples` is less than 1.
    """
    if min_samples < 1:
        msg = f"min_samples must be at least 1, got {min_samples}."
        raise ValueError(msg)

    window = window_mask(time, start, end)
    offset = int(np.flatnonzero(window)[0])
    values = np.asarray(traces, dtype=np.float64)[:, window]
    threshold = np.asarray(threshold, dtype=np.float64)

    onset = np.full(values.shape[0], np.nan)
    if values.shape[1] < min_samples:
        return onset

    # NaN comparisons are False, so a NaN sample or threshold breaks any run.
    above = values > threshold[:, None]
    runs = np.lib.stride_tricks.sliding_window_view(above, min_samples, axis=1).all(axis=-1)
    found = runs.any(axis=1)
    onset[found] = time[np.argmax(runs[found], axis=1) + offset]
    return onset


def extrapolated_onset(
    traces: npt.NDArray,
    time: npt.NDArray[np.float64],
    peak_index: npt.NDArray[np.int64],
    amplitude: npt.NDArray[np.float64],
    start: float,
    low: float = 0.2,
    high: float = 0.8,
) -> npt.NDArray[np.float64]:
    """Onset by extrapolating a line fitted to the rise between `low` and `high` of the amplitude to baseline.

    The rise is the last run of samples before the peak, from where the trace last crosses `low * amplitude`
    up to where it first exceeds `high * amplitude`. Onset is where the fitted line crosses zero, and may fall
    before `start`.

    Args:
        traces (npt.NDArray): Baseline-subtracted traces of shape `(n_trials, n_samples)`.
        time (npt.NDArray[np.float64]): Sample times relative to the event, shape `(n_samples,)`.
        peak_index (npt.NDArray[np.int64]): Sample index of each trial's peak, from `peak`.
        amplitude (npt.NDArray[np.float64]): Amplitude of each trial's peak, from `peak`.
        start (float): Response window start, in seconds; the rise is searched from here.
        low (float, optional): Lower amplitude fraction of the fitted rise. Defaults to 0.2.
        high (float, optional): Upper amplitude fraction of the fitted rise. Defaults to 0.8.

    Returns:
        npt.NDArray[np.float64]: Onset time per trial, shape `(n_trials,)`. NaN for trials whose amplitude is not
        positive, whose rise has fewer than two samples, or whose fitted slope is not positive.

    Raises:
        ValueError: If `low` and `high` are not within `(0, 1)` with `low < high`.
    """
    if not 0 < low < high < 1:
        msg = f"Rise fractions must satisfy 0 < low < high < 1, got {low} and {high}."
        raise ValueError(msg)

    values = np.asarray(traces, dtype=np.float64)
    first = int(np.flatnonzero(time >= start)[0])
    onset = np.full(values.shape[0], np.nan)
    for i in np.flatnonzero((amplitude > 0) & (peak_index >= 0)):
        rise = values[i, first : peak_index[i] + 1]
        rise_time = time[first : peak_index[i] + 1]
        below = np.flatnonzero(rise < low * amplitude[i])
        begin = below[-1] + 1 if below.size else 0
        above = np.flatnonzero(rise[begin:] > high * amplitude[i])
        stop = begin + above[0] if above.size else len(rise)
        if stop - begin < 2:  # noqa: PLR2004
            continue
        slope, intercept = np.polyfit(rise_time[begin:stop], rise[begin:stop], 1)
        if slope > 0:
            onset[i] = -intercept / slope
    return onset


def offset_time(
    traces: npt.NDArray,
    time: npt.NDArray[np.float64],
    threshold: npt.NDArray[np.float64],
    peak_index: npt.NDArray[np.int64],
    end: float,
    min_samples: int = 3,
) -> npt.NDArray[np.float64]:
    """Time of the first run of `min_samples` consecutive samples at or below `threshold` after the peak.

    Args:
        traces (npt.NDArray): Baseline-subtracted traces of shape `(n_trials, n_samples)`.
        time (npt.NDArray[np.float64]): Sample times relative to the event, shape `(n_samples,)`.
        threshold (npt.NDArray[np.float64]): Per-trial threshold, shape `(n_trials,)`.
        peak_index (npt.NDArray[np.int64]): Sample index of each trial's peak, from `peak`.
        end (float): Response window end, in seconds; the search stops here.
        min_samples (int, optional): Consecutive samples required at or below threshold. Defaults to 3.

    Returns:
        npt.NDArray[np.float64]: Offset time per trial, shape `(n_trials,)`. NaN where the peak is not above the
        threshold, no such run exists, or the threshold is NaN.

    Raises:
        ValueError: If `min_samples` is less than 1.
    """
    if min_samples < 1:
        msg = f"min_samples must be at least 1, got {min_samples}."
        raise ValueError(msg)

    values = np.asarray(traces, dtype=np.float64)
    threshold = np.asarray(threshold, dtype=np.float64)
    last = int(np.flatnonzero(time <= end)[-1])
    offset = np.full(values.shape[0], np.nan)
    for i in np.flatnonzero(peak_index >= 0):
        if not values[i, peak_index[i]] > threshold[i]:
            continue
        after = values[i, peak_index[i] + 1 : last + 1]
        if after.size < min_samples:
            continue
        below = after <= threshold[i]
        runs = np.lib.stride_tricks.sliding_window_view(below, min_samples).all(axis=-1)
        if runs.any():
            offset[i] = time[peak_index[i] + 1 + int(np.argmax(runs))]
    return offset


def decay_time(
    traces: npt.NDArray,
    time: npt.NDArray[np.float64],
    peak_index: npt.NDArray[np.int64],
    amplitude: npt.NDArray[np.float64],
    end: float,
    fraction: float = 0.5,
) -> npt.NDArray[np.float64]:
    """Time from the peak until the trace first falls to `fraction * amplitude`, searching up to `t <= end`.

    Args:
        traces (npt.NDArray): Baseline-subtracted traces of shape `(n_trials, n_samples)`.
        time (npt.NDArray[np.float64]): Sample times relative to the event, shape `(n_samples,)`.
        peak_index (npt.NDArray[np.int64]): Sample index of each trial's peak, from `peak`.
        amplitude (npt.NDArray[np.float64]): Amplitude of each trial's peak, from `peak`.
        end (float): Response window end, in seconds.
        fraction (float, optional): Fraction of the amplitude the trace must fall to. Defaults to 0.5.

    Returns:
        npt.NDArray[np.float64]: Decay time per trial, shape `(n_trials,)`. NaN for trials whose amplitude is not
        positive or whose trace never falls to the fraction within the window.

    Raises:
        ValueError: If `fraction` is not within `(0, 1)`.
    """
    if not 0 < fraction < 1:
        msg = f"fraction must be within (0, 1), got {fraction}."
        raise ValueError(msg)

    values = np.asarray(traces, dtype=np.float64)
    last = int(np.flatnonzero(time <= end)[-1])
    decay = np.full(values.shape[0], np.nan)
    for i in np.flatnonzero((amplitude > 0) & (peak_index >= 0)):
        after = values[i, peak_index[i] + 1 : last + 1]
        below = np.flatnonzero(after <= fraction * amplitude[i])
        if below.size:
            decay[i] = time[peak_index[i] + 1 + below[0]] - time[peak_index[i]]
    return decay


def trace_correlation(traces: npt.NDArray, time: npt.NDArray[np.float64], start: float, end: float) -> float:
    """Mean pairwise Pearson correlation between trial traces over `start <= t <= end`.

    Args:
        traces (npt.NDArray): Traces of shape `(n_trials, n_samples)`.
        time (npt.NDArray[np.float64]): Sample times relative to the event, shape `(n_samples,)`.
        start (float): Window start, in seconds.
        end (float): Window end, in seconds.

    Returns:
        float: Mean of the upper triangle of the trial-by-trial correlation matrix, ignoring NaN pairs. NaN with
        fewer than two trials.
    """
    import pandas as pd

    window = window_mask(time, start, end)
    values = np.asarray(traces, dtype=np.float64)[:, window]
    corr = pd.DataFrame(values.T).corr().to_numpy()
    pairs = corr[np.triu_indices_from(corr, k=1)]
    return float(np.nanmean(pairs)) if pairs.size and not np.all(np.isnan(pairs)) else float("nan")


def trial_metrics(
    traces: npt.NDArray,
    time: npt.NDArray[np.float64],
    baseline: tuple[float, float],
    response: tuple[float, float],
    onset: str = "sd",
    onset_sd: float = 2.0,
    onset_fraction: float = 0.2,
    onset_min_samples: int = 3,
    extrapolate_range: tuple[float, float] = (0.2, 0.8),
    decay_fraction: float = 0.5,
    smoothing: int = 1,
) -> pd.DataFrame:
    """Per-trial response metrics for one region.

    Traces are baseline-subtracted with the per-trial mean over the baseline window before any metric is taken.
    With `smoothing` above 1, the peak, onset, decay and offset are taken from the moving-average trace, while
    the baseline SD and AUC come from the raw trace.

    Args:
        traces (npt.NDArray): Traces of shape `(n_trials, n_samples)`.
        time (npt.NDArray[np.float64]): Sample times relative to the event, shape `(n_samples,)`.
        baseline (tuple[float, float]): Baseline window `[start, end)`, in seconds.
        response (tuple[float, float]): Response window `[start, end]`, in seconds.
        onset (str, optional): `sd` thresholds onset at `onset_sd` baseline SDs, `peak` at `onset_fraction` of
            the amplitude, and `extrapolate` fits the rise between `extrapolate_range` fractions of the
            amplitude and takes where the line crosses baseline. Defaults to `sd`.
        onset_sd (float, optional): Baseline SD multiple for `onset="sd"`. Defaults to 2.0.
        onset_fraction (float, optional): Amplitude fraction for `onset="peak"`. Defaults to 0.2.
        onset_min_samples (int, optional): Consecutive samples above threshold for an onset, and at or below it
            for an offset. Defaults to 3.
        extrapolate_range (tuple[float, float], optional): Amplitude fractions bounding the rise fitted for
            `onset="extrapolate"`. Defaults to `(0.2, 0.8)`.
        decay_fraction (float, optional): Amplitude fraction the trace must fall to after the peak. Defaults to 0.5.
        smoothing (int, optional): Moving-average window in samples, odd. Defaults to 1, no smoothing.

    Returns:
        pd.DataFrame: One row per trial with `baseline_mean`, `baseline_sd`, `onset_time`, `peak_time`,
        `amplitude`, `auc`, `decay_time`, `offset_time` and `duration` columns. Offset is the return to the onset
        threshold after the peak, or to the lower `extrapolate_range` fraction for `onset="extrapolate"`;
        duration is offset minus onset.

    Raises:
        ValueError: If `onset` is not one of `ONSET_METHODS`.

    Example:
        >>> metrics = trial_metrics(traces, time, baseline=(-1.0, 0.0), response=(0.0, 3.0))
        >>> metrics["amplitude"].mean()
    """
    import pandas as pd

    if onset not in ONSET_METHODS:
        msg = f"Unknown onset method {onset!r}; expected one of {', '.join(ONSET_METHODS)}."
        raise ValueError(msg)

    baseline_mean, baseline_sd = baseline_stats(traces, time, *baseline)
    subtracted = np.asarray(traces, dtype=np.float64) - baseline_mean[:, None]
    smoothed = smooth(subtracted, smoothing)

    amplitude, peak_time, peak_index = peak(smoothed, time, *response)
    if onset == "sd":
        threshold = onset_sd * baseline_sd
    else:
        fraction = onset_fraction if onset == "peak" else extrapolate_range[0]
        threshold = np.where(amplitude > 0, fraction * amplitude, np.nan)

    if onset == "extrapolate":
        onsets = extrapolated_onset(smoothed, time, peak_index, amplitude, response[0], *extrapolate_range)
    else:
        onsets = onset_time(smoothed, time, threshold, *response, min_samples=onset_min_samples)
    offsets = offset_time(smoothed, time, threshold, peak_index, response[1], min_samples=onset_min_samples)

    return pd.DataFrame(
        {
            "baseline_mean": baseline_mean,
            "baseline_sd": baseline_sd,
            "onset_time": onsets,
            "peak_time": peak_time,
            "amplitude": amplitude,
            "auc": auc(subtracted, time, *response),
            "decay_time": decay_time(smoothed, time, peak_index, amplitude, response[1], fraction=decay_fraction),
            "offset_time": offsets,
            "duration": offsets - onsets,
        }
    )


def session_metrics(metrics: pd.DataFrame, correlation: float) -> dict[str, float | int]:
    """Across-trial mean, SD and coefficient of variation of each per-trial metric.

    Args:
        metrics (pd.DataFrame): Per-trial metrics for one region, from `trial_metrics`.
        correlation (float): Trial-to-trial trace correlation, from `trace_correlation`.

    Returns:
        dict[str, float | int]: `n_trials`, `trace_correlation`, then `<metric>_mean`, `<metric>_sd` and
        `<metric>_cv` for each of `METRICS`, plus `onset_n`, `decay_n` and `offset_n` counts of trials with a
        defined onset, decay and offset. NaN values are ignored; CV is NaN where the mean is zero.
    """
    summary: dict[str, float | int] = {"n_trials": len(metrics), "trace_correlation": correlation}
    for name in METRICS:
        values = metrics[name].to_numpy(dtype=np.float64)
        valid = values[~np.isnan(values)]
        mean = float(np.mean(valid)) if valid.size else float("nan")
        sd = float(np.std(valid, ddof=1)) if valid.size > 1 else float("nan")
        summary[f"{name}_mean"] = mean
        summary[f"{name}_sd"] = sd
        summary[f"{name}_cv"] = sd / mean if mean else float("nan")
    summary["onset_n"] = int(metrics["onset_time"].notna().sum())
    summary["decay_n"] = int(metrics["decay_time"].notna().sum())
    summary["offset_n"] = int(metrics["offset_time"].notna().sum())
    return summary


def metrics_tables(
    perievent: pd.DataFrame,
    baseline: tuple[float, float] | None = None,
    response: tuple[float, float] | None = None,
    trials: pd.DataFrame | None = None,
    onset: str = "sd",
    onset_sd: float = 2.0,
    onset_fraction: float = 0.2,
    onset_min_samples: int = 3,
    extrapolate_range: tuple[float, float] = (0.2, 0.8),
    decay_fraction: float = 0.5,
    smoothing: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-trial and per-session metric tables from a long-format peri-event table.

    Args:
        perievent (pd.DataFrame): Peri-event table with `trial_index`, `event_time`, `time`, `region` and `F`
            columns, as written by `process peri-event` from a `_regions.csv`. Any other column is taken as
            per-trial information, such as the trials columns `process peri-event` joins on, and carried through.
        baseline (tuple[float, float] | None, optional): Baseline window `[start, end)`. Defaults to all
            pre-event samples.
        response (tuple[float, float] | None, optional): Response window `[start, end]`. Defaults to all
            post-event samples.
        trials (pd.DataFrame | None, optional): Trials table to join onto the per-trial table by row index, via
            `join_trials`, for peri-event tables without trials columns. Defaults to None.
        onset (str, optional): Onset method, see `trial_metrics`. Defaults to `sd`.
        onset_sd (float, optional): See `trial_metrics`. Defaults to 2.0.
        onset_fraction (float, optional): See `trial_metrics`. Defaults to 0.2.
        onset_min_samples (int, optional): See `trial_metrics`. Defaults to 3.
        extrapolate_range (tuple[float, float], optional): See `trial_metrics`. Defaults to `(0.2, 0.8)`.
        decay_fraction (float, optional): See `trial_metrics`. Defaults to 0.5.
        smoothing (int, optional): See `trial_metrics`. Defaults to 1.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The per-trial table, one row per trial per region with `trial_index`,
        `event_time`, `region`, the `trial_metrics` columns and then the per-trial columns of `perievent`, and the
        per-session table, one row per region with `region` and the `session_metrics` columns.

    Raises:
        ValueError: If `perievent` lacks any of `PERIEVENT_COLUMNS`, or has no rows.

    Example:
        >>> perievent = pd.read_csv("ses-01_regions_event-cueonset_perievent.csv")
        >>> per_trial, per_session = metrics_tables(perievent, smoothing=5)
    """
    import pandas as pd

    missing = PERIEVENT_COLUMNS - set(perievent.columns)
    if missing:
        msg = f"Peri-event table lacks the {', '.join(sorted(missing))} column(s)."
        raise ValueError(msg)
    if perievent.empty:
        msg = "Peri-event table has no rows."
        raise ValueError(msg)

    time = np.sort(perievent["time"].unique()).astype(np.float64)
    baseline = baseline if baseline is not None else (float(time[0]), 0.0)
    response = response if response is not None else (0.0, float(time[-1]))
    trial_columns = [column for column in perievent.columns if column not in {"time", "region", "F"}]
    trial_info = perievent[trial_columns].drop_duplicates(subset="trial_index").reset_index(drop=True)
    events = trial_info[["trial_index", "event_time"]]
    extra = trial_info.drop(columns="event_time")

    per_trial = []
    per_session = []
    for region in perievent["region"].unique():
        traces = (
            perievent[perievent["region"] == region]
            .pivot(index="trial_index", columns="time", values="F")
            .reindex(index=events["trial_index"], columns=time)
            .to_numpy(dtype=np.float64)
        )
        metrics = trial_metrics(
            traces,
            time,
            baseline,
            response,
            onset=onset,
            onset_sd=onset_sd,
            onset_fraction=onset_fraction,
            onset_min_samples=onset_min_samples,
            extrapolate_range=extrapolate_range,
            decay_fraction=decay_fraction,
            smoothing=smoothing,
        )
        metrics.insert(0, "region", region)
        per_trial.append(pd.concat([events, metrics], axis=1))
        correlation = trace_correlation(traces, time, *response)
        per_session.append({"region": region, **session_metrics(metrics, correlation)})

    trial_table = pd.concat(per_trial, ignore_index=True)
    if len(extra.columns) > 1:
        trial_table = trial_table.merge(extra, on="trial_index", how="left")
    if trials is not None:
        trial_table = join_trials(trial_table, trials)
    return trial_table, pd.DataFrame(per_session)


def join_trials(table: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    """Left-join trials table columns onto a per-trial table by trials row index.

    Args:
        table (pd.DataFrame): Table with a `trial_index` column, such as per-trial metrics or peri-event windows.
        trials (pd.DataFrame): Trials table as written by `visiomode-analysis session`.

    Returns:
        pd.DataFrame: `table` with the trials columns appended, minus any unnamed index column and any column
        `table` already has.
    """
    trials = trials.loc[:, ~trials.columns.str.startswith("Unnamed")].reset_index(drop=True)
    trials = trials.drop(columns=[column for column in trials.columns if column in table.columns])
    trials.index.name = "trial_index"
    return table.merge(trials.reset_index(), on="trial_index", how="left")
