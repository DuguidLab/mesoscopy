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

"""Peri-event window extraction from behaviour-aligned recordings."""

from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt

if typing.TYPE_CHECKING:
    import pandas as pd

# Trials CSV column, or columns, that give the event time for each `--event` choice.
EVENTS = ("cue_onset", "trial_start", "response", "reward")

# Outcomes for which a reward is delivered at `stop_time`.
REWARDED_OUTCOMES = frozenset({"hit", "correct_rejection"})

# Tolerance when deciding whether the last grid point is still within `post`, in seconds.
_GRID_TOLERANCE_S = 1e-9


def event_name(event: str) -> str:
    """Event name as used in output filenames, with `_` and `-` removed.

    Args:
        event (str): One of `EVENTS`.

    Returns:
        str: The event name, e.g. `cueonset`.
    """
    return event.replace("_", "").replace("-", "")


def event_times(trials: pd.DataFrame, event: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Absolute event time per trial, dropping trials without the event.

    Event is what is being aligned to (i.e. the zero of the peri-event window).

    Args:
        trials (pd.DataFrame): Trials table as written by `visiomode-analysis session`, with `start_time`,
            `cue_onset`, `stop_time`, `response_time` and `sdt_type` columns. Times are seconds from
            behaviour start.
        event (str): One of `cue_onset`, `trial_start`, `response` or `reward`.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: Event times in seconds from behaviour start,
        and the row index into `trials` of each kept trial.

    Raises:
        ValueError: If `event` is not one of `EVENTS`.
    """
    if event == "cue_onset":
        times = trials["cue_onset"].to_numpy(dtype=np.float64)
        keep = ~np.isnan(times)
    elif event == "trial_start":
        times = trials["start_time"].to_numpy(dtype=np.float64)
        keep = ~np.isnan(times)
    elif event == "response":
        import pandas as pd

        response_time = pd.to_numeric(trials["response_time"], errors="coerce").to_numpy(dtype=np.float64)
        times = trials["start_time"].to_numpy(dtype=np.float64) + response_time
        keep = ~np.isnan(response_time) & (response_time >= 0) & ~np.isnan(times)
    elif event == "reward":
        times = trials["stop_time"].to_numpy(dtype=np.float64)
        keep = trials["sdt_type"].isin(REWARDED_OUTCOMES).to_numpy() & ~np.isnan(times)
    else:
        msg = f"Unknown event {event!r}; expected one of {', '.join(EVENTS)}."
        raise ValueError(msg)

    trial_index = np.flatnonzero(keep).astype(np.int64)
    return times[keep], trial_index


def window_grid(pre: float, post: float, fs: float) -> npt.NDArray[np.float64]:
    """Sample times relative to the event, from `-pre` to `post` inclusive at `fs` Hz.

    Args:
        pre (float): Seconds before the event.
        post (float): Seconds after the event.
        fs (float): Sampling rate of the grid, in Hz.

    Returns:
        npt.NDArray[np.float64]: Grid times in seconds, shape `(n_samples,)`.
    """
    step = 1.0 / fs
    grid = np.arange(-pre, post + step, step, dtype=np.float64)
    return grid[grid <= post + _GRID_TOLERANCE_S]


def extract(
    data: npt.NDArray,
    time_aligned: npt.NDArray[np.float64],
    events: npt.NDArray[np.float64],
    grid: npt.NDArray[np.float64],
    method: str = "interp",
    dtype: npt.DTypeLike = np.float32,
) -> tuple[npt.NDArray, npt.NDArray[np.bool_]]:
    """Sample a recording at `event + grid` for every event.

    Events whose window runs outside the recording are dropped.

    Args:
        data (npt.NDArray): Recording with time on the first axis, shape `(n_frames, ...)`.
        time_aligned (npt.NDArray[np.float64]): Time of each frame, shape `(n_frames,)`, increasing.
        events (npt.NDArray[np.float64]): Event times on the same clock, shape `(n_events,)`.
        grid (npt.NDArray[np.float64]): Sample times relative to each event, from `window_grid`.
        method (str, optional): `interp` for linear interpolation, `nearest` for the nearest frame.
            Defaults to `interp`.
        dtype (npt.DTypeLike, optional): Output dtype. Defaults to float32.

    Returns:
        tuple[npt.NDArray, npt.NDArray[np.bool_]]: Traces of shape `(n_kept, n_samples, ...)`, and a boolean
        mask over `events` marking the kept events.

    Raises:
        ValueError: If `method` is not `interp` or `nearest`.
    """
    if method not in {"interp", "nearest"}:
        msg = f"Unknown method {method!r}; expected 'interp' or 'nearest'."
        raise ValueError(msg)

    time_aligned = np.asarray(time_aligned, dtype=np.float64)
    events = np.asarray(events, dtype=np.float64)
    kept = (events + grid[0] >= time_aligned[0]) & (events + grid[-1] <= time_aligned[-1])

    n_frames = data.shape[0]
    trailing_shape = data.shape[1:]
    flat = np.asarray(data).reshape(n_frames, -1)

    traces = np.empty((int(kept.sum()), len(grid), flat.shape[1]), dtype=dtype)
    for i, event in enumerate(events[kept]):
        sample_times = event + grid
        if method == "interp":
            for j in range(flat.shape[1]):
                traces[i, :, j] = np.interp(sample_times, time_aligned, flat[:, j])
        else:
            traces[i] = flat[_nearest_index(time_aligned, sample_times)]

    return traces.reshape(traces.shape[0], len(grid), *trailing_shape), kept


def apply_baseline(traces: npt.NDArray, grid: npt.NDArray[np.float64], start: float, end: float) -> npt.NDArray:
    """Subtract the per-trial mean over `start <= t < end` from every sample.

    Args:
        traces (npt.NDArray): Traces of shape `(n_trials, n_samples, ...)`, as returned by `extract`.
        grid (npt.NDArray[np.float64]): Sample times relative to the event, shape `(n_samples,)`.
        start (float): Start of the baseline window, in seconds relative to the event.
        end (float): End of the baseline window, also in seconds relative to the event. Exclusive.

    Returns:
        npt.NDArray: Baseline-subtracted traces, same shape and dtype as `traces`.

    Raises:
        ValueError: If no grid point falls within the baseline window.

    Example:
        >>> grid = window_grid(pre=1.0, post=3.0, fs=25.0)
        >>> traces, kept = extract(deltaf_series, time_aligned, events, grid)
        >>> traces = apply_baseline(traces, grid, start=-1.0, end=0.0)
    """
    window = (grid >= start) & (grid < end)
    if not window.any():
        msg = f"No samples fall within the baseline window [{start}, {end})."
        raise ValueError(msg)
    baseline = np.nanmean(traces[:, window], axis=1, keepdims=True)
    return (traces - baseline).astype(traces.dtype, copy=False)


def _nearest_index(time_aligned: npt.NDArray[np.float64], sample_times: npt.NDArray[np.float64]) -> npt.NDArray:
    """Index of the frame closest in time to each sample time.

    Args:
        time_aligned (npt.NDArray[np.float64]): Frame times, increasing.
        sample_times (npt.NDArray[np.float64]): Times to look up.

    Returns:
        npt.NDArray: Frame index per sample time.
    """
    upper = np.searchsorted(time_aligned, sample_times).clip(1, len(time_aligned) - 1)
    lower = upper - 1
    choose_upper = np.abs(time_aligned[upper] - sample_times) < np.abs(sample_times - time_aligned[lower])
    return np.where(choose_upper, upper, lower)
