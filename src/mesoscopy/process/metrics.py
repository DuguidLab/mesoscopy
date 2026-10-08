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

import dataclasses
import typing
import warnings

import numpy as np
import numpy.typing as npt

from mesoscopy.process import perievent as pev

if typing.TYPE_CHECKING:
    import pandas as pd

# Per-trial metrics summarised per session, in output column order.
METRICS = ("onset_time", "peak_time", "amplitude", "auc", "decay_time", "offset_time", "duration")

# Columns of the bootstrapped mean-trace metrics, in output order.
BOOT_COLUMNS = tuple(f"boot_{name}{stat}" for name in METRICS for stat in ("", "_ci_low", "_ci_high", "_n"))

# Columns of the long-format peri-event table written by `process peri-event`.
PERIEVENT_COLUMNS = frozenset({"trial_index", "event_time", "time", "region", "F"})

# Onset methods: `sd` thresholds at a multiple of the baseline SD, `peak` at a fraction of the amplitude, and
# `extrapolate` fits a line to the rise and takes where it crosses the baseline.
ONSET_METHODS = ("sd", "peak", "extrapolate")

# Reliability metrics per trial group, in output column order.
RELIABILITY_METRICS = (
    "epoch_correlation",
    "response_fraction",
    "variance_quench",
    "signal_fraction",
    "reliability_n",
)

# Trial types of a go/no-go session, split by the stimulus shown.
GO_TYPES = ("hit", "miss")
NOGO_TYPES = ("false_alarm", "correct_rejection")

# Trial types of a go/no-go session, split by whether the lever was pushed.
PUSH_TYPES = ("hit", "false_alarm")
NOPUSH_TYPES = ("miss", "correct_rejection")

# Trials columns the reliability metrics need.
RELIABILITY_COLUMNS = frozenset({"cue_onset", "response_time", "sdt_type"})


@dataclasses.dataclass(frozen=True)
class ReliabilityOptions:
    """Options for the trial-to-trial reliability metrics.

    Attributes:
        mask_response (bool): Mask cue-aligned epochs from each trial's response time. Defaults to True.
        min_rt (float): Trials with a response time below this, in seconds, are dropped. Defaults to 0.2.
        response_sd (float): Baseline SD multiple a trial's mean epoch response must exceed to count as a
            response. Defaults to 2.0.
        time_warp (bool): Resample each epoch onto a common 0-1 grid. Defaults to False.
        cue_baseline (tuple[float, float]): Baseline window `[start, end)` relative to each trial's cue, for
            lever-aligned windows. Defaults to `(-1.0, 0.0)`.
    """

    mask_response: bool = True
    min_rt: float = 0.2
    response_sd: float = 2.0
    time_warp: bool = False
    cue_baseline: tuple[float, float] = (-1.0, 0.0)


@dataclasses.dataclass(frozen=True)
class BootstrapOptions:
    """Options for the bootstrapped mean-trace metrics.

    Attributes:
        n_resamples (int): Resamples of the trials; 0 skips the mean-trace metrics. Defaults to 10000.
        ci (float): Width of the percentile intervals, in percent. Defaults to 95.0.
        seed (int | None): Seed for the resamples; None draws a fresh one. Defaults to 42.
    """

    n_resamples: int = 10000
    ci: float = 95.0
    seed: int | None = 42

    def __post_init__(self) -> None:
        """Check the options.

        Raises:
            ValueError: If `n_resamples` is negative or `ci` is not within `(0, 100)`.
        """
        if self.n_resamples < 0:
            msg = f"n_resamples must be at least 0, got {self.n_resamples}."
            raise ValueError(msg)
        if not 0 < self.ci < 100:  # noqa: PLR2004
            msg = f"ci must be within (0, 100), got {self.ci}."
            raise ValueError(msg)


@dataclasses.dataclass(frozen=True)
class MetricsTables:
    """Metric tables from a peri-event table, from `metrics_tables`.

    Attributes:
        per_trial (pd.DataFrame): One row per trial per region, written to `<stem>_metrics.csv`.
        per_session (pd.DataFrame): One row per region per trial group, written to `<stem>_metrics-session.csv`.
        boot (pd.DataFrame | None): Mean-trace metrics with bootstrap intervals, one row per region per trial
            group, written to `<stem>_metrics-boot.csv`. None when the bootstrap is skipped.
        boot_traces (pd.DataFrame | None): Mean traces with bootstrap intervals, one row per sample per region per
            trial group, written to `<stem>_traces-boot.csv`. None when the bootstrap is skipped.
    """

    per_trial: pd.DataFrame
    per_session: pd.DataFrame
    boot: pd.DataFrame | None = None
    boot_traces: pd.DataFrame | None = None


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
        positive, whose rise has fewer than two samples or a NaN sample, or whose fitted slope is not positive.

    Raises:
        ValueError: If `low` and `high` are not within `(0, 1)` with `low < high`.
    """
    if not 0 < low < high < 1:
        msg = f"Rise fractions must satisfy 0 < low < high < 1, got {low} and {high}."
        raise ValueError(msg)

    values = np.asarray(traces, dtype=np.float64)
    peak_index = np.asarray(peak_index)
    amplitude = np.asarray(amplitude, dtype=np.float64)
    first = int(np.flatnonzero(time >= start)[0])
    onset = np.full(values.shape[0], np.nan)
    rows = np.flatnonzero((amplitude > 0) & (peak_index >= 0))
    if not rows.size:
        return onset

    values, peak_at, height = values[rows], peak_index[rows, None], amplitude[rows, None]
    sample = np.arange(values.shape[1])
    rise = (sample >= first) & (sample <= peak_at)
    begin = np.where(rise & (values < low * height), sample, first - 1).max(axis=1) + 1
    above = (sample >= begin[:, None]) & (sample <= peak_at) & (values > high * height)
    stop = np.where(above.any(axis=1), above.argmax(axis=1), peak_at[:, 0] + 1)
    fit = (sample >= begin[:, None]) & (sample < stop[:, None])
    count = fit.sum(axis=1)

    # Least-squares line through the rise; a NaN sample in it gives a NaN onset.
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_time = np.where(fit, time, 0.0).sum(axis=1) / count
        mean_value = np.where(fit, values, 0.0).sum(axis=1) / count
        time_dev = np.where(fit, time - mean_time[:, None], 0.0)
        value_dev = np.where(fit, values - mean_value[:, None], 0.0)
        slope = (time_dev * value_dev).sum(axis=1) / (time_dev**2).sum(axis=1)
        fitted = (count >= 2) & (slope > 0)  # noqa: PLR2004
        onset[rows[fitted]] = (mean_time - mean_value / slope)[fitted]
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
    peak_index = np.asarray(peak_index)
    last = int(np.flatnonzero(time <= end)[-1])
    offset = np.full(values.shape[0], np.nan)
    rows = np.flatnonzero(peak_index >= 0)
    if not rows.size or values.shape[1] < min_samples:
        return offset

    values, peak_at, threshold = values[rows], peak_index[rows], threshold[rows]
    sample = np.arange(values.shape[1])
    # NaN comparisons are False, so a NaN sample or threshold breaks any run.
    below = (sample > peak_at[:, None]) & (sample <= last) & (values <= threshold[:, None])
    runs = np.lib.stride_tricks.sliding_window_view(below, min_samples, axis=1).all(axis=-1)
    found = (values[np.arange(rows.size), peak_at] > threshold) & runs.any(axis=1)
    offset[rows[found]] = time[runs.argmax(axis=1)[found]]
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
    peak_index = np.asarray(peak_index)
    amplitude = np.asarray(amplitude, dtype=np.float64)
    last = int(np.flatnonzero(time <= end)[-1])
    decay = np.full(values.shape[0], np.nan)
    rows = np.flatnonzero((amplitude > 0) & (peak_index >= 0))
    if not rows.size:
        return decay

    peak_at = peak_index[rows]
    sample = np.arange(values.shape[1])
    fallen = (sample > peak_at[:, None]) & (sample <= last) & (values[rows] <= fraction * amplitude[rows, None])
    found = fallen.any(axis=1)
    decay[rows[found]] = time[fallen.argmax(axis=1)[found]] - time[peak_at[found]]
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


def resample_counts(n_trials: int, n_resamples: int, seed: int | None = None) -> npt.NDArray[np.float64]:
    """How often each trial is drawn in each resample of the trials, drawn with replacement.

    Args:
        n_trials (int): Trials to resample, at least 1.
        n_resamples (int): Resamples to draw.
        seed (int | None, optional): Seed for `np.random.default_rng`. Defaults to None.

    Returns:
        npt.NDArray[np.float64]: Counts of shape `(n_resamples, n_trials)`; each row sums to `n_trials`.
    """
    draws = np.random.default_rng(seed).integers(0, n_trials, size=(n_resamples, n_trials))
    flat = (draws + n_trials * np.arange(n_resamples)[:, None]).ravel()
    return np.bincount(flat, minlength=n_resamples * n_trials).reshape(n_resamples, n_trials).astype(np.float64)


def _percentiles(values: npt.NDArray[np.float64], q: list[float]) -> npt.NDArray[np.float64]:
    """Percentiles over the first axis, ignoring NaNs.

    Returns:
        npt.NDArray[np.float64]: One row per percentile; NaN where every value is NaN.
    """
    if not np.isnan(values).any():
        return np.percentile(values, q, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(values, q, axis=0)


def bootstrap_metrics(
    traces: npt.NDArray,
    time: npt.NDArray[np.float64],
    baseline: tuple[float, float],
    response: tuple[float, float],
    counts: npt.NDArray[np.float64],
    ci: float = 95.0,
    **options: typing.Any,
) -> tuple[dict[str, float | int], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Response metrics of the trial-mean trace, with bootstrap percentile intervals.

    Each trace is baseline-subtracted with its mean over the baseline window, and the traces are averaged, ignoring
    NaNs. The `trial_metrics` of that mean trace are the estimates. Each resample's mean trace, weighting every
    trial by how often it was drawn, gives one value per metric, and the interval spans the central `ci` percent of
    the values where the metric is defined.

    Args:
        traces (npt.NDArray): Traces of shape `(n_trials, n_samples)`.
        time (npt.NDArray[np.float64]): Sample times relative to the event, shape `(n_samples,)`.
        baseline (tuple[float, float]): Baseline window `[start, end)`, in seconds.
        response (tuple[float, float]): Response window `[start, end]`, in seconds.
        counts (npt.NDArray[np.float64]): Draws of each trial per resample, shape `(n_resamples, n_trials)`, from
            `resample_counts`.
        ci (float, optional): Interval width, in percent. Defaults to 95.0.
        **options (typing.Any): Other `trial_metrics` arguments, such as `onset` and `smoothing`.

    Returns:
        tuple[dict[str, float | int], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        `boot_<metric>`, `boot_<metric>_ci_low`, `boot_<metric>_ci_high` and `boot_<metric>_n`, the resamples
        where the metric is defined, for each of `METRICS`; then the mean trace and the low and high ends of its
        pointwise interval, each shape `(n_samples,)`.

    Raises:
        ValueError: If `ci` is not within `(0, 100)`.

    Example:
        >>> counts = resample_counts(len(traces), 10000, seed=42)
        >>> summary, mean, low, high = bootstrap_metrics(traces, time, (-1.0, 0.0), (0.0, 3.0), counts, smoothing=5)
        >>> summary["boot_onset_time"], summary["boot_onset_time_ci_low"], summary["boot_onset_time_ci_high"]
    """
    if not 0 < ci < 100:  # noqa: PLR2004
        msg = f"ci must be within (0, 100), got {ci}."
        raise ValueError(msg)

    baseline_mean, _ = baseline_stats(traces, time, *baseline)
    subtracted = np.asarray(traces, dtype=np.float64) - baseline_mean[:, None]
    present = ~np.isnan(subtracted)
    filled = np.where(present, subtracted, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = filled.sum(axis=0) / present.sum(axis=0)
        if present.all():
            resampled = (counts @ filled) / subtracted.shape[0]
        else:
            resampled = (counts @ filled) / (counts @ present.astype(np.float64))

    tail = (100 - ci) / 2
    estimates = trial_metrics(mean[None, :], time, baseline, response, **options)
    resampled_metrics = trial_metrics(resampled, time, baseline, response, **options)
    summary: dict[str, float | int] = {}
    for name in METRICS:
        values = resampled_metrics[name].to_numpy(dtype=np.float64)
        defined = values[~np.isnan(values)]
        low, high = np.percentile(defined, [tail, 100 - tail]) if defined.size else (np.nan, np.nan)
        summary[f"boot_{name}"] = float(estimates[name].iloc[0])
        summary[f"boot_{name}_ci_low"] = float(low)
        summary[f"boot_{name}_ci_high"] = float(high)
        summary[f"boot_{name}_n"] = int(defined.size)
    band_low, band_high = _percentiles(resampled, [tail, 100 - tail])
    return summary, mean, band_low, band_high


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
    event: str | None = None,
    reliability: ReliabilityOptions | None = None,
    min_trials: int = 10,
    bootstrap: BootstrapOptions | None = None,
) -> MetricsTables:
    """Per-trial, per-session and bootstrapped mean-trace metric tables from a long-format peri-event table.

    For go/no-go sessions with an `sdt_type` column, the per-session and bootstrap tables have a row per region for
    each `trial_groups` group; other sessions get the `all` rows only, with a warning. For go/no-go sessions, the
    per-session table also gets the `reliability_metrics` columns. They are skipped with a warning when the
    trials columns they need are missing, the session is not go/no-go, or the windows are reward-aligned.

    The bootstrap tables hold the `bootstrap_metrics` of each region and trial group. Every region of a group
    uses the same resamples of the group's trials, from `resample_counts`.

    Args:
        perievent (pd.DataFrame): Peri-event table with `trial_index`, `event_time`, `time`, `region` and `F`
            columns, as written by `process peri-event` from a `_regions.csv`. Any other column is taken as
            per-trial information, such as the trials columns `process peri-event` joins on, and carried through.
            Checked by `perievent_trials`.
        baseline (tuple[float, float] | None, optional): Baseline window `[start, end)`. Defaults to all
            pre-event samples.
        response (tuple[float, float] | None, optional): Response window `[start, end]`. Defaults to all
            post-event samples.
        trials (pd.DataFrame | None, optional): Trials table to join onto the per-trial table by row index, via
            `join_trials`, for peri-event tables without trials columns. Columns `perievent` already carries are
            skipped. Defaults to None.
        onset (str, optional): Onset method, see `trial_metrics`. Defaults to `sd`.
        onset_sd (float, optional): See `trial_metrics`. Defaults to 2.0.
        onset_fraction (float, optional): See `trial_metrics`. Defaults to 0.2.
        onset_min_samples (int, optional): See `trial_metrics`. Defaults to 3.
        extrapolate_range (tuple[float, float], optional): See `trial_metrics`. Defaults to `(0.2, 0.8)`.
        decay_fraction (float, optional): See `trial_metrics`. Defaults to 0.5.
        smoothing (int, optional): See `trial_metrics`. Defaults to 1.
        event (str | None, optional): Event the windows are aligned to, one of `perievent.EVENTS`. Defaults to
            the event matching `event_time`, via `infer_event`.
        reliability (ReliabilityOptions | None, optional): Reliability metric options. Defaults to
            `ReliabilityOptions()`.
        min_trials (int, optional): Trial groups with fewer trials get NaN session summaries, reliability metrics
            and mean-trace metrics, and no mean trace; the all-trials session summaries are always taken. Defaults
            to 10.
        bootstrap (BootstrapOptions | None, optional): Bootstrap options. Defaults to `BootstrapOptions()`.

    Returns:
        MetricsTables: The per-trial table, one row per trial per region with `trial_index`, `event_time`,
        `region`, the `trial_metrics` columns and then the per-trial columns of `perievent`; the per-session table,
        one row per region per trial group with `region`, `group`, the `session_metrics` columns and the
        `reliability_metrics` columns; the bootstrap table, one row per region per trial group with `region`,
        `group`, `n_trials` and `BOOT_COLUMNS`; and the mean traces, one row per sample per region per trial group
        with `region`, `group`, `n_trials`, `time`, `mean`, `ci_low` and `ci_high`.

    Example:
        >>> perievent = pd.read_csv("ses-01_regions_event-cueonset_perievent.csv")
        >>> tables = metrics_tables(perievent, smoothing=5)
        >>> tables.boot[["region", "group", "boot_onset_time", "boot_onset_time_ci_low", "boot_onset_time_ci_high"]]
    """
    import pandas as pd

    time, trial_info = perievent_trials(perievent)
    baseline = baseline if baseline is not None else (float(time[0]), 0.0)
    response = response if response is not None else (0.0, float(time[-1]))
    extra_columns = [column for column in trial_info.columns if column not in {"trial_index", "event_time"}]
    events = trial_info[["trial_index", "event_time"]]

    info = trial_info
    if trials is not None:
        info = join_trials(trial_info, trials.drop(columns=extra_columns, errors="ignore"))
    group_skip = group_skip_reason(info)
    if group_skip:
        warnings.warn(f"Skipping trial groups: {group_skip}", stacklevel=2)
    groups = {"all": np.ones(len(info), dtype=bool)} if group_skip else trial_groups(info)
    reliability = reliability if reliability is not None else ReliabilityOptions()
    event = event if event is not None else infer_event(info)
    skip = _reliability_skip_reason(info, event)
    if skip:
        warnings.warn(f"Skipping reliability metrics: {skip}", stacklevel=2)
    elif event == "response":
        cue = info["cue_onset"].to_numpy(dtype=np.float64) - info["event_time"].to_numpy(dtype=np.float64)
        uncovered = int((cue + reliability.cue_baseline[0] < time[0] - _TIME_TOLERANCE_S).sum())
        if uncovered:
            warnings.warn(
                f"{uncovered} of {len(info)} trials have a pre-cue baseline outside the window; their response"
                " fraction and variance quench baselines are NaN. Extract windows with a longer --pre.",
                stacklevel=2,
            )
    bootstrap = bootstrap if bootstrap is not None else BootstrapOptions()
    counts = {
        name: resample_counts(int(mask.sum()), bootstrap.n_resamples, bootstrap.seed)
        for name, mask in groups.items()
        if bootstrap.n_resamples and mask.sum() >= min_trials
    }
    options: dict[str, typing.Any] = {
        "onset": onset,
        "onset_sd": onset_sd,
        "onset_fraction": onset_fraction,
        "onset_min_samples": onset_min_samples,
        "extrapolate_range": extrapolate_range,
        "decay_fraction": decay_fraction,
        "smoothing": smoothing,
    }

    per_trial = []
    per_session = []
    boot: list[dict[str, typing.Any]] = []
    boot_traces = []
    for region in perievent["region"].unique():
        traces = (
            perievent[perievent["region"] == region]
            .pivot(index="trial_index", columns="time", values="F")
            .reindex(index=events["trial_index"], columns=time)
            .to_numpy(dtype=np.float64)
        )
        metrics = trial_metrics(traces, time, baseline, response, **options)
        metrics.insert(0, "region", region)
        per_trial.append(pd.concat([events, metrics], axis=1))
        group_reliability: dict[str, dict[str, float | int]] = {}
        if not skip:
            group_reliability = reliability_metrics(
                traces, time, info, str(event), baseline, response, reliability, min_trials
            )
        for name, mask in groups.items():
            enough = name == "all" or int(mask.sum()) >= min_trials
            correlation = trace_correlation(traces[mask], time, *response) if enough else float("nan")
            summary = session_metrics(metrics[mask], correlation)
            if not enough:
                summary |= {key: float("nan") for key in summary if key.endswith(("_mean", "_sd", "_cv"))}
            per_session.append({"region": region, "group": name, **summary, **group_reliability.get(name, {})})

            if not bootstrap.n_resamples:
                continue
            row = {"region": region, "group": name, "n_trials": int(mask.sum())}
            if name not in counts:
                boot.append(row)
                continue
            boot_summary, mean, low, high = bootstrap_metrics(
                traces[mask], time, baseline, response, counts[name], bootstrap.ci, **options
            )
            boot.append(row | boot_summary)
            boot_traces.append(pd.DataFrame({**row, "time": time, "mean": mean, "ci_low": low, "ci_high": high}))

    trial_table = pd.concat(per_trial, ignore_index=True)
    if extra_columns:
        trial_table = trial_table.merge(trial_info[["trial_index", *extra_columns]], on="trial_index", how="left")
    if trials is not None:
        trial_table = join_trials(trial_table, trials.drop(columns=extra_columns, errors="ignore"))
    session_table = pd.DataFrame(per_session)
    if not bootstrap.n_resamples:
        return MetricsTables(trial_table, session_table)

    boot_table = pd.DataFrame(boot, columns=["region", "group", "n_trials", *BOOT_COLUMNS])
    # Resample counts stay integers, empty for groups below `min_trials`.
    resample_columns = [column for column in BOOT_COLUMNS if column.endswith("_n")]
    boot_table[resample_columns] = boot_table[resample_columns].astype("Int64")
    trace_columns = ["region", "group", "n_trials", "time", "mean", "ci_low", "ci_high"]
    trace_table = pd.concat(boot_traces, ignore_index=True) if boot_traces else pd.DataFrame(columns=trace_columns)
    return MetricsTables(trial_table, session_table, boot_table, trace_table)


def perievent_trials(perievent: pd.DataFrame) -> tuple[npt.NDArray[np.float64], pd.DataFrame]:
    """Sample times and per-trial information of a long-format peri-event table.

    Args:
        perievent (pd.DataFrame): Peri-event table with `PERIEVENT_COLUMNS`; any other column is taken as
            per-trial information.

    Returns:
        tuple[npt.NDArray[np.float64], pd.DataFrame]: Sorted sample times, shape `(n_samples,)`, and one row
        per trial with `trial_index`, `event_time` and the per-trial columns, in order of first appearance.

    Raises:
        ValueError: If `perievent` lacks any of `PERIEVENT_COLUMNS`, has no rows, or has a column other than `time`,
            `region` and `F` that varies within a trial.
    """
    missing = PERIEVENT_COLUMNS - set(perievent.columns)
    if missing:
        msg = f"Peri-event table lacks the {', '.join(sorted(missing))} column(s)."
        raise ValueError(msg)
    if perievent.empty:
        msg = "Peri-event table has no rows."
        raise ValueError(msg)

    time = np.sort(perievent["time"].unique()).astype(np.float64)
    extra_columns = [column for column in perievent.columns if column not in PERIEVENT_COLUMNS]
    trial_info = perievent[["trial_index", "event_time", *extra_columns]].drop_duplicates().reset_index(drop=True)
    if trial_info["trial_index"].duplicated().any():
        per_trial_values = perievent.groupby("trial_index")[["event_time", *extra_columns]].nunique(dropna=False)
        varying = per_trial_values.columns[(per_trial_values > 1).any()]
        msg = f"Column(s) {', '.join(varying)} vary within a trial; only per-trial columns can be carried through."
        raise ValueError(msg)
    return time, trial_info


def join_trials(table: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    """Left-join trials table columns onto a per-trial table by trials row index.

    Args:
        table (pd.DataFrame): Table with a `trial_index` column, such as per-trial metrics or peri-event windows.
        trials (pd.DataFrame): Trials table as written by `visiomode-analysis session`.

    Returns:
        pd.DataFrame: `table` with the trials columns appended, minus any unnamed index column; clashing names
        get a `_trial` suffix.
    """
    trials = trials.loc[:, ~trials.columns.str.startswith("Unnamed")].reset_index(drop=True)
    trials.index.name = "trial_index"
    return table.merge(trials.reset_index(), on="trial_index", how="left", suffixes=("", "_trial"))


# Tolerance when comparing times, in seconds.
_TIME_TOLERANCE_S = 1e-6


def infer_event(trial_info: pd.DataFrame) -> str | None:
    """Event the peri-event windows are aligned to, from which trials column `event_time` matches.

    Args:
        trial_info (pd.DataFrame): One row per trial with `event_time` and trials columns.

    Returns:
        str | None: One of `perievent.EVENTS`, or None when no trials column matches.
    """
    event_time = trial_info["event_time"].to_numpy(dtype=np.float64)
    for event in pev.EVENTS:
        try:
            times, index = pev.event_times(trial_info, event)
        except KeyError:
            continue
        if len(index) == len(trial_info) and np.allclose(times, event_time, rtol=0, atol=_TIME_TOLERANCE_S):
            return event
    return None


def group_skip_reason(trial_info: pd.DataFrame) -> str | None:
    """Why the trials cannot be split into `trial_groups`.

    Args:
        trial_info (pd.DataFrame): One row per trial.

    Returns:
        str | None: The reason, or None when they can be.
    """
    if "sdt_type" not in trial_info.columns:
        return "no sdt_type column; pass the trials CSV."
    if "protocol" in trial_info.columns and not trial_info["protocol"].eq("gonogo").all():
        return "only go/no-go sessions are supported."
    return None


def _reliability_skip_reason(trial_info: pd.DataFrame, event: str | None) -> str | None:
    """Why the reliability metrics cannot be taken.

    Returns:
        str | None: The reason, or None when they can be taken.
    """
    missing = RELIABILITY_COLUMNS - set(trial_info.columns)
    if missing:
        return f"no {', '.join(sorted(missing))} column(s); pass the trials CSV."
    if "protocol" in trial_info.columns and not trial_info["protocol"].eq("gonogo").all():
        return "only go/no-go sessions are supported."
    if event is None:
        return "event_time matches no trials column, so the aligned event is unknown."
    if event not in {"cue_onset", "trial_start", "response"}:
        return f"not defined for {event}-aligned windows."
    return None


def kept_trials(trial_info: pd.DataFrame, min_rt: float = 0.2) -> npt.NDArray[np.bool_]:
    """Trials without a response faster than `min_rt`.

    Args:
        trial_info (pd.DataFrame): One row per trial, with `response_time` in seconds from the cue when known.
        min_rt (float, optional): Shortest response time kept, in seconds. Defaults to 0.2.

    Returns:
        npt.NDArray[np.bool_]: Mask of shape `(n_trials,)`, all True without a `response_time` column.
    """
    import pandas as pd

    if "response_time" not in trial_info.columns:
        return np.ones(len(trial_info), dtype=bool)
    response_time = pd.to_numeric(trial_info["response_time"], errors="coerce").to_numpy(dtype=np.float64)
    return ~((response_time >= 0) & (response_time < min_rt))


def reliability_epochs(
    trial_info: pd.DataFrame,
    event: str,
    response: tuple[float, float],
    mask_response: bool = True,
    min_rt: float = 0.2,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Per-trial reliability epoch `[start, end)`, relative to the event.

    Cue- and trial-start-aligned epochs run over the response window, ending at each trial's response when
    `mask_response` is set; trials without a response end at the median response time of those with one.
    Lever-aligned (`response`) epochs run from the cue to the lever push.

    Args:
        trial_info (pd.DataFrame): One row per trial with `event_time`, `cue_onset` and `response_time` columns.
            `response_time` is seconds from the cue.
        event (str): `cue_onset`, `trial_start` or `response`.
        response (tuple[float, float]): Response window `[start, end]`, in seconds.
        mask_response (bool, optional): End cue-aligned epochs at the response. Defaults to True.
        min_rt (float, optional): Trials with a response time below this are dropped. Defaults to 0.2.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        Epoch start and end, the cue time, all relative to the event, and which trials are kept; each shape
        `(n_trials,)`.

    Raises:
        ValueError: If `event` is not `cue_onset`, `trial_start` or `response`.
    """
    import pandas as pd

    n_trials = len(trial_info)
    response_time = pd.to_numeric(trial_info["response_time"], errors="coerce").to_numpy(dtype=np.float64)
    responded = response_time >= 0
    keep = kept_trials(trial_info, min_rt)
    cue = trial_info["cue_onset"].to_numpy(dtype=np.float64) - trial_info["event_time"].to_numpy(dtype=np.float64)

    if event == "response":
        return cue, np.zeros(n_trials), cue, keep
    if event not in {"cue_onset", "trial_start"}:
        msg = f"Reliability epochs are not defined for {event!r}; expected cue_onset, trial_start or response."
        raise ValueError(msg)

    start = np.full(n_trials, float(response[0]))
    # The response window end is inclusive.
    window_end = np.nextafter(float(response[1]), np.inf)
    if not mask_response:
        return start, np.full(n_trials, window_end), cue, keep
    timed = responded & keep
    median_rt = float(np.median(response_time[timed])) if timed.any() else np.nan
    end = np.minimum(cue + np.where(responded, response_time, median_rt), window_end)
    return start, end, cue, keep


def epoch_traces(
    traces: npt.NDArray, time: npt.NDArray[np.float64], start: npt.NDArray[np.float64], end: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Traces with every sample outside each trial's `[start, end)` set to NaN.

    Args:
        traces (npt.NDArray): Traces of shape `(n_trials, n_samples)`.
        time (npt.NDArray[np.float64]): Sample times relative to the event, shape `(n_samples,)`.
        start (npt.NDArray[np.float64]): Per-trial epoch start, shape `(n_trials,)`.
        end (npt.NDArray[np.float64]): Per-trial epoch end, exclusive, shape `(n_trials,)`.

    Returns:
        npt.NDArray[np.float64]: Masked traces, same shape.
    """
    inside = (time[None, :] >= start[:, None]) & (time[None, :] < end[:, None])
    return np.where(inside, np.asarray(traces, dtype=np.float64), np.nan)


def anchored_window(
    traces: npt.NDArray, time: npt.NDArray[np.float64], anchor: npt.NDArray[np.float64], start: float, end: float
) -> npt.NDArray[np.float64]:
    """Each trace sampled over `anchor + start <= t < anchor + end` at the recording's sample interval.

    Args:
        traces (npt.NDArray): Traces of shape `(n_trials, n_samples)`.
        time (npt.NDArray[np.float64]): Sample times relative to the event, shape `(n_samples,)`.
        anchor (npt.NDArray[np.float64]): Per-trial anchor time relative to the event, shape `(n_trials,)`.
        start (float): Window start relative to the anchor, in seconds.
        end (float): Window end relative to the anchor, in seconds. Exclusive.

    Returns:
        npt.NDArray[np.float64]: Linearly interpolated samples, shape `(n_trials, n_window)`. Rows are NaN where the
        window is not within `time`.
    """
    values = np.asarray(traces, dtype=np.float64)
    step = float(np.median(np.diff(time)))
    offsets = start + step * np.arange(int(np.ceil((end - start) / step - _TIME_TOLERANCE_S)))
    window = np.full((values.shape[0], offsets.size), np.nan)
    for i in np.flatnonzero(~np.isnan(anchor)):
        at = anchor[i] + offsets
        if offsets.size and at[0] >= time[0] - _TIME_TOLERANCE_S and at[-1] <= time[-1] + _TIME_TOLERANCE_S:
            window[i] = np.interp(at, time, values[i])
    return window


def warp_epochs(
    epochs: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    start: npt.NDArray[np.float64],
    end: npt.NDArray[np.float64],
    n_samples: int,
) -> npt.NDArray[np.float64]:
    """Resample each epoch onto `n_samples` points spanning 0 (epoch start) to 1 (epoch end).

    Args:
        epochs (npt.NDArray[np.float64]): Masked traces from `epoch_traces`, shape `(n_trials, n_time)`.
        time (npt.NDArray[np.float64]): Sample times relative to the event, shape `(n_time,)`.
        start (npt.NDArray[np.float64]): Per-trial epoch start, shape `(n_trials,)`.
        end (npt.NDArray[np.float64]): Per-trial epoch end, shape `(n_trials,)`.
        n_samples (int): Points on the warped grid.

    Returns:
        npt.NDArray[np.float64]: Warped epochs, shape `(n_trials, n_samples)`. NaN beyond the samples a trial has,
        and for trials with fewer than two samples.
    """
    grid = np.linspace(0.0, 1.0, n_samples)
    warped = np.full((epochs.shape[0], n_samples), np.nan)
    for i in range(epochs.shape[0]):
        valid = ~np.isnan(epochs[i])
        if valid.sum() < 2 or not end[i] > start[i]:  # noqa: PLR2004
            continue
        position = (time[valid] - start[i]) / (end[i] - start[i])
        inside = (grid >= position[0] - _TIME_TOLERANCE_S) & (grid <= position[-1] + _TIME_TOLERANCE_S)
        warped[i, inside] = np.interp(grid[inside], position, epochs[i, valid])
    return warped


def epoch_correlation(epochs: npt.NDArray[np.float64], min_samples: int = 3) -> float:
    """Mean zero-lag Pearson correlation between every pair of trials, over the samples both have.

    Args:
        epochs (npt.NDArray[np.float64]): Masked traces of shape `(n_trials, n_samples)`.
        min_samples (int, optional): Shared samples a pair needs. Defaults to 3.

    Returns:
        float: Mean over pairs with a defined correlation. NaN when there is none.
    """
    import pandas as pd

    corr = pd.DataFrame(epochs.T).corr(min_periods=min_samples).to_numpy()
    pairs = corr[np.triu_indices_from(corr, k=1)]
    return float(np.nanmean(pairs)) if pairs.size and not np.all(np.isnan(pairs)) else float("nan")


def response_fraction(
    epochs: npt.NDArray[np.float64], baseline: npt.NDArray[np.float64], response_sd: float = 2.0
) -> float:
    """Fraction of trials whose mean epoch value exceeds their baseline mean by `response_sd` baseline SDs.

    Args:
        epochs (npt.NDArray[np.float64]): Masked traces of shape `(n_trials, n_samples)`.
        baseline (npt.NDArray[np.float64]): Baseline samples of shape `(n_trials, n_baseline)`.
        response_sd (float, optional): Baseline SD multiple. Defaults to 2.0.

    Returns:
        float: Fraction over trials with a defined epoch mean and baseline. NaN when there is none.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        response = np.nanmean(epochs, axis=1) - np.nanmean(baseline, axis=1)
        threshold = response_sd * np.nanstd(baseline, axis=1, ddof=1)
    valid = ~np.isnan(response) & ~np.isnan(threshold)
    return float(np.mean(response[valid] > threshold[valid])) if valid.any() else float("nan")


def _across_trial_variance(values: npt.NDArray[np.float64]) -> float:
    """Mean over samples of the across-trial variance.

    Returns:
        float: The mean, over samples with at least two trials. NaN when there are none.
    """
    enough = (~np.isnan(values)).sum(axis=0) >= 2  # noqa: PLR2004
    if not enough.any():
        return float("nan")
    return float(np.mean(np.nanvar(values[:, enough], axis=0, ddof=1)))


def variance_quench(epochs: npt.NDArray[np.float64], baseline: npt.NDArray[np.float64]) -> float:
    """Across-trial variance over the epoch divided by across-trial variance over the baseline.

    Args:
        epochs (npt.NDArray[np.float64]): Masked traces of shape `(n_trials, n_samples)`.
        baseline (npt.NDArray[np.float64]): Baseline samples of shape `(n_trials, n_baseline)`.

    Returns:
        float: The ratio; below 1 when variability drops after the event. NaN when either variance is undefined
        or the baseline variance is zero.
    """
    epoch_variance = _across_trial_variance(epochs)
    baseline_variance = _across_trial_variance(baseline)
    return epoch_variance / baseline_variance if baseline_variance > 0 else float("nan")


def signal_fraction(epochs: npt.NDArray[np.float64]) -> float:
    """Fraction of single-trial variance explained by the trial-mean trace.

    Args:
        epochs (npt.NDArray[np.float64]): Masked traces of shape `(n_trials, n_samples)`.

    Returns:
        float: One minus the residual sum of squares about the trial mean over the total sum of squares, over
        samples with at least two trials. NaN when there are none or the traces are constant.
    """
    enough = (~np.isnan(epochs)).sum(axis=0) >= 2  # noqa: PLR2004
    values = epochs[:, enough]
    if not values.size or np.all(np.isnan(values)):
        return float("nan")
    residual = np.nansum((values - np.nanmean(values, axis=0)) ** 2)
    total = np.nansum((values - np.nanmean(values)) ** 2)
    return float(1 - residual / total) if total > 0 else float("nan")


def trial_groups(trial_info: pd.DataFrame) -> dict[str, npt.NDArray[np.bool_]]:
    """Trial groups of a go/no-go session, keyed by the `group` value of the per-session table.

    Args:
        trial_info (pd.DataFrame): One row per trial with an `sdt_type` column.

    Returns:
        dict[str, npt.NDArray[np.bool_]]: `all` for all trials, `sdt-<type>` for each of `GO_TYPES` and
        `NOGO_TYPES`, `stim-go` and `stim-nogo` by the stimulus shown, then `resp-push` and `resp-nopush` by
        whether the lever was pushed; each a mask of shape `(n_trials,)`.
    """
    sdt_type = trial_info["sdt_type"].to_numpy()
    groups = {"all": np.ones(len(trial_info), dtype=bool)}
    for name in (*GO_TYPES, *NOGO_TYPES):
        groups[f"sdt-{name}"] = sdt_type == name
    groups["stim-go"] = np.isin(sdt_type, GO_TYPES)
    groups["stim-nogo"] = np.isin(sdt_type, NOGO_TYPES)
    groups["resp-push"] = np.isin(sdt_type, PUSH_TYPES)
    groups["resp-nopush"] = np.isin(sdt_type, NOPUSH_TYPES)
    return groups


def _group_reliability(
    epochs: npt.NDArray[np.float64], baseline: npt.NDArray[np.float64], options: ReliabilityOptions, min_trials: int
) -> dict[str, float | int]:
    """Reliability metrics for one group.

    Returns:
        dict[str, float | int]: One value per `RELIABILITY_METRICS`, NaN below `min_trials` trials.
    """
    n_trials = epochs.shape[0]
    if n_trials < min_trials:
        return {**{name: float("nan") for name in RELIABILITY_METRICS[:-1]}, "reliability_n": n_trials}
    return {
        "epoch_correlation": epoch_correlation(epochs),
        "response_fraction": response_fraction(epochs, baseline, options.response_sd),
        "variance_quench": variance_quench(epochs, baseline),
        "signal_fraction": signal_fraction(epochs),
        "reliability_n": n_trials,
    }


def reliability_metrics(
    traces: npt.NDArray,
    time: npt.NDArray[np.float64],
    trial_info: pd.DataFrame,
    event: str,
    baseline: tuple[float, float],
    response: tuple[float, float],
    options: ReliabilityOptions | None = None,
    min_trials: int = 10,
) -> dict[str, dict[str, float | int]]:
    """Trial-to-trial reliability of one region's response, per trial group.

    Metrics are taken over each trial's epoch from `reliability_epochs`: `epoch_correlation`, `response_fraction`,
    `variance_quench` and `signal_fraction`, plus `reliability_n`, the trials used. The baseline is `baseline`
    relative to the event, or `options.cue_baseline` relative to each trial's cue for lever-aligned windows. Groups
    with fewer than `min_trials` trials get NaN.

    Args:
        traces (npt.NDArray): Traces of shape `(n_trials, n_samples)`, in the row order of `trial_info`.
        time (npt.NDArray[np.float64]): Sample times relative to the event, shape `(n_samples,)`.
        trial_info (pd.DataFrame): One row per trial with `event_time`, `cue_onset`, `response_time` and
            `sdt_type` columns.
        event (str): `cue_onset`, `trial_start` or `response`.
        baseline (tuple[float, float]): Baseline window `[start, end)` relative to the event, for cue- and
            trial-start-aligned windows.
        response (tuple[float, float]): Response window `[start, end]`, for cue- and trial-start-aligned windows.
        options (ReliabilityOptions | None, optional): Defaults to `ReliabilityOptions()`.
        min_trials (int, optional): Groups with fewer trials get NaN. Defaults to 10.

    Returns:
        dict[str, dict[str, float | int]]: The `RELIABILITY_METRICS` of each `trial_groups` group, keyed by group.

    Example:
        >>> summary = reliability_metrics(traces, time, trial_info, "cue_onset", (-1.0, 0.0), (0.0, 3.0))
        >>> summary["stim-go"]["epoch_correlation"]
    """
    options = options if options is not None else ReliabilityOptions()
    start, end, cue, keep = reliability_epochs(trial_info, event, response, options.mask_response, options.min_rt)
    epochs = epoch_traces(traces, time, start, end)
    if options.time_warp:
        lengths = (~np.isnan(epochs[keep])).sum(axis=1)
        lengths = lengths[lengths >= 2]  # noqa: PLR2004
        epochs = warp_epochs(epochs, time, start, end, int(np.median(lengths)) if lengths.size else 2)
    if event == "response":
        base = anchored_window(traces, time, cue, *options.cue_baseline)
    else:
        base = anchored_window(traces, time, np.zeros(len(cue)), *baseline)

    return {
        name: _group_reliability(epochs[mask & keep], base[mask & keep], options, min_trials)
        for name, mask in trial_groups(trial_info).items()
    }
