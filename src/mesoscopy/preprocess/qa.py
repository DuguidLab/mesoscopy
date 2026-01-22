#  Copyright (c) 2025 Constantinos Eleftheriou <Constantinos.Eleftheriou@ed.ac.uk>.
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
"""Module for quality assurance (QA) functions in the mesoscopy preprocessing pipeline."""

import typing
import diptest
from datetime import datetime
import scipy.stats as stats
import numpy as np
import numpy.typing as npt
import plotly.express as px
import plotly.graph_objects as go


def check_histogram_separation(filter_statistic_timeseries: npt.NDArray, alpha: float = 0.05) -> bool:
    """Check whether array histogram contains two separable peaks using Hartigan's dip test for unimodality.

    Args:
        filter_statistic_timeseries (npt.NDArray): Filter array used for signal separation (e.g. frame means or frame
        standard deviation timeseries)
        alpha (float, optional): Significance level for the dip test. Defaults to 0.05.

    Returns:
        bool: Indicates whether histogram separates successfully (i.e. distribution is bimodal).
    """
    _dip, p_value = diptest.diptest(filter_statistic_timeseries)  # pyright: ignore[reportAssignmentType]
    return p_value < alpha


def check_timestamp_consistency(
    timestamps: npt.NDArray | list, threshold: float = 2.0, percentile: float = 99.0
) -> bool:
    """Check for timestamp consistency by analyzing the standard deviation of frame intervals.

    Consistent timestamps should have low variability in timestamp intervals.
    Generally, anything above two standard deviations is considered inconsistent, i.e. we want most of our timestamp
    intervals to fall within two standard deviations.
    This function computes the z-score of the frame intervals and checks if the specified percentile exceeds the given
    standard deviation threshold.

    Args:
        timestamps (npt.NDArray | list): Sequence of timestamp values.
        threshold (float, optional): Threshold for standard deviation of frame intervals to indicate drift.
        Defaults to 2.0.
        percentile (float, optional): Percentile of the z-scored frame intervals to consider for drift detection.
        Defaults to 99.0.

    Returns:
        bool: Indicates whether timestamp drift is detected (True if drift is detected).
    """
    try:
        timestamps = np.array([datetime.fromisoformat(str(ts, encoding="utf-8")) for ts in np.array(timestamps)])
    except ValueError:
        timestamps = np.array([float(ts) for ts in timestamps])
    timedeltas = np.diff(timestamps).astype("timedelta64[ms]").astype(float)
    zscored_intervals = stats.zscore(timedeltas)
    critical_value = np.percentile(zscored_intervals, percentile)  # type: ignore[reportArgumentType]

    return critical_value <= threshold


def check_timestamp_jumps(timestamps: npt.NDArray | list, std_threshold: float = 2.0) -> bool:
    """Check for timestamp jumps by analyzing the standard deviation of frame intervals.

    Jumps in timestamps are defined as anything above two standard deviations in the z-scored frame intervals.

    Args:
        timestamps (npt.NDArray | list): Sequence of timestamp values.
        std_threshold (float, optional): Threshold for standard deviation of frame intervals to indicate jumps.
        Defaults to 2.0.

    Returns:
        bool: Indicates whether timestamp jumps are detected (True if jumps are detected).
    """
    try:
        timestamps = np.array([datetime.fromisoformat(str(ts, encoding="utf-8")) for ts in np.array(timestamps)])
    except ValueError:
        timestamps = np.array([float(ts) for ts in timestamps])
    timedeltas = np.diff(timestamps).astype("timedelta64[ms]").astype(float)
    zscored_intervals = stats.zscore(timedeltas)
    max_deviation = np.abs(np.array(zscored_intervals)).max()

    return max_deviation <= std_threshold


def calculate_noise(data: npt.NDArray, framerate: int = 25) -> npt.NDArray:
    """Calculate noise levels for each pixel in a the preprocessed delta F signal.

    Based on Cascade calculation for 2p imaging (https://github.com/HelmchenLabSoftware/Cascade/blob/8735f5022447d4942e5d466ab0775a4bfca1c7e8/cascade2p/utils.py#L97).

    From the Cascade documentation:
        The noise level is computed as the median absolute dF/F difference
        between two subsequent time points. This is a outlier-robust measurement
        that converges to the simple standard deviation of the dF/F trace for
        uncorrelated and outlier-free dF/F traces.

        Afterwards, the value is divided by the square root of the frame rate
        in order to make it comparable across recordings with different frame rates.

    Args:
        data (npt.NDArray): deltaF array (time x height x width).
        framerate (int, optional): Frame rate of extracted deltaF trace. Defaults to 25 Hz.

    Returns:
        npt.NDArray: Map of normalised noise level per pixel, scaled as a percentage (height x width).
    """
    return np.array((np.nanmedian(np.abs(np.diff(data, axis=0)), axis=0) / np.sqrt(framerate)) * 100)


def calculate_snr(data: npt.NDArray, noise_levels: npt.NDArray | None = None, framerate: int = 25) -> float:
    """Estimate the signal-to-noise ratio (SNR) for a preprocessed delta F recording.

    SNR is calculated as the median of the 99th percentile of deltaF across each pixel divided by the (unscaled) noise
    level per pixel.

    Args:
        data (npt.NDArray): deltaF array (time x height x width).
        noise_levels (npt.NDArray | None, optional): Map of normalised noise level per pixel as a percentage, as would
        be returned by the `calculate_noise` function. This will be computed if not provided. Defaults to None.
        framerate (int, optional): Frame rate of extracted deltaF trace. Only used if noise_levels are calculated on the
        fly. Defaults to 25.

    Returns:
        float: Signal-to-noise ratio.
    """
    if noise_levels is None:
        noise_levels = calculate_noise(data, framerate)
    return float(np.median(np.percentile(np.array(data), 99, axis=0) / np.percentile((noise_levels / 100), 99)))


def check_noise(
    data: npt.NDArray, framerate: int = 25, threshold: float = 10.0, return_noise: bool = False
) -> bool | tuple[bool, npt.NDArray]:
    """Check whether recording noise is below a predetermined acceptable noise threshold.

    This function will first call `calculate_noise` to calculate the noise levels in the deltaF trace and then check
    whether the median noise level is below a pre-defined threshold.

    Based on https://gcamp6f.com/2021/10/04/large-scale-calcium-imaging-noise-levels/ it looks like
    anything above 10% shot is in the upper bound of acceptability, likely more so for mesoscale data.

    This function will optionally return the calculated noise levels per pixel.

    Args:
        data (npt.NDArray): deltaF array (time x height x width).
        framerate (int, optional): Frame rate of extracted deltaF trace. Defaults to 25.
        threshold (float, optional): Shot noise percentage threshold. Defaults to 10.0.
        return_noise (bool, optional): Return the calculate noise levels per pixel alongside the check result.
        Defaults to False.

    Returns:
        bool | tuple[bool, npt.NDArray]: Returns true if check is passed (i.e. median noise level is below threshold).
        If `return_noise` is True, returns a tuple with check result alongside the pixel-wise calculated noise levels as
        an array of width x height.
    """
    noise_levels = calculate_noise(data, framerate)
    passed = bool(np.median(noise_levels) <= threshold)

    if return_noise:
        return passed, np.array(noise_levels)
    return passed


def check_snr(
    data: npt.NDArray,
    noise_levels: npt.NDArray | None = None,
    framerate: int = 25,
    threshold: float = 1.5,
    return_snr: bool = False,
) -> bool | tuple[bool, float]:
    """Check whether recording noise is below a predetermined acceptable SNR threshold.

    This function will first call `calculate_snr` to calculate the signal-to-noise ratio in the deltaF trace and then
    check whether SNR is above a pre-defined threshold.

    This function will optionally return the calculated SNR for the session.

    Args:
        data (npt.NDArray): deltaF array (time x height x width).
        noise_levels (npt.NDArray | None, optional): Map of normalised noise level per pixel as a percentage, as would
        be returned by the `calculate_noise` function. This will be computed if not provided. Defaults to None.
        framerate (int, optional): Frame rate of extracted deltaF trace. Only used if noise_levels are calculated on the
        fly. Defaults to 25.
        threshold (float, optional): SNR threshold. Defaults to 1.5.
        return_snr (bool, optional): Return the calculate SNR alongside the check result.
        Defaults to False.

    Returns:
        bool | tuple[bool, float]: Returns true if check is passed (i.e. SNR is above
          threshold).
        If `return_noise` is True, returns a tuple with check result alongside the SNR as a float.
    """
    if noise_levels is None:
        noise_levels = calculate_noise(data, framerate)
    snr = calculate_snr(data, noise_levels)
    passed = bool(snr >= threshold)

    if return_snr:
        return passed, snr
    return passed


def check_bleaching(
    f_mean_timeseries: npt.NDArray, threshold: float = 5e-7, return_slope: bool = False
) -> bool | tuple[bool, float]:
    """Check mean deltaF signal for photobleaching.

    This function will fit a straight line through the average delta F trace. If the slope of this line is lower than a
    critical negative threshold (i.e. the delta F signal is decreasing in slope across the recording)
    this would suggest that photobleaching is quite likely present.

    Args:
        f_mean_timeseries (npt.NDArray): Average delta F signal over time as a one-dimensional array. If the recording
        was preprocessed with mesoscopy, this is found under "/qa/f_mean_timeseries"
        threshold (float, optional): Critical slope threshold value. Can be expressed as an absolute value, or as a
        negative value. Defaults to 5e-7.
        return_slope (bool, optional): Return the calculated slope in addition to the check result. Defaults to False.

    Returns:
        bool | tuple[bool, float]: Returns True if the check passed (i.e. no photobleaching was detected).
        Optionally returns the calculated slope if `return_slope` is True.
    """
    f_mean_timeseries = np.array(f_mean_timeseries)
    slope, _ = np.polyfit(x=range(f_mean_timeseries.shape[0]), y=f_mean_timeseries, deg=1)

    threshold = -abs(threshold)
    passed = slope >= threshold

    if return_slope:
        return passed, slope
    return passed


def plot_timestamps(timestamps: list | npt.NDArray, as_html: bool = False) -> str | go.Figure:
    """Plots a sequence of timestamps as a line plot using Plotly.

    Args:
        timestamps (list | npt.NDArray): Sequence of timestamp values to plot.
        as_html (bool, optional): If True, returns the plot as an HTML string. If False, returns a Plotly Figure object.

    Returns:
        str | go.Figure: The plot as an HTML string if `as_html` is True, otherwise a Plotly Figure object.
    """
    fig = go.Figure(
        layout=go.Layout(
            xaxis={"title": {"text": "Frame index"}},
            yaxis={"title": {"text": "Timestamp"}},
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        )
    )

    fig.add_trace(go.Scatter(y=timestamps, mode="lines", name="Timestamp"))
    if as_html:
        return fig.to_html(full_html=False)
    return fig


def plot_timestamp_jumps(
    timestamps: npt.NDArray | list, std_threshold: float = 2.0, as_html: bool = False
) -> str | go.Figure:
    try:
        timestamps = np.array([datetime.fromisoformat(str(ts, encoding="utf-8")) for ts in np.array(timestamps)])
    except ValueError:
        timestamps = np.array([float(ts) for ts in timestamps])
    timedeltas = np.diff(timestamps).astype("timedelta64[ms]").astype(float)
    zscored_intervals = stats.zscore(timedeltas)

    fig = go.Figure(
        layout=go.Layout(
            xaxis={"title": {"text": "Frame index"}},
            yaxis={"title": {"text": "Timedelta (ms)"}},
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        )
    )

    fig.add_trace(go.Scatter(y=timedeltas, mode="lines", name="Timestamp"))
    if as_html:
        return fig.to_html(full_html=False)
    return fig


def plot_raw_timeseries(raw_timeseries: npt.NDArray, as_html: bool = False) -> str | go.Figure:
    """Plots a raw timeseries signal using Plotly.

    Args:
        raw_timeseries (npt.NDArray): The raw timeseries data to plot, as a NumPy array.
        as_html (bool, optional): If True, returns the plot as an HTML string. If False, returns a Plotly Figure object.

    Returns:
        str | go.Figure: The plot as an HTML string if `as_html` is True, otherwise a Plotly Figure object.
    """
    fig = go.Figure(
        layout=go.Layout(
            xaxis={"title": {"text": "Frame index"}},
            yaxis={"title": {"text": "Signal"}},
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        )
    )

    fig.add_trace(go.Scatter(y=raw_timeseries, mode="markers", name="Signal"))
    if as_html:
        return fig.to_html(full_html=False)
    return fig


def plot_raw_histogram(data: npt.NDArray, as_html: bool = False) -> str | go.Figure:
    """Plots a histogram of raw data using Plotly.

    Args:
        data (npt.NDArray): The raw data to plot as a histogram, as a NumPy array.
        as_html (bool, optional): If True, returns the plot as an HTML string. If False, returns a Plotly Figure object.

    Returns:
        str | go.Figure: The plot as an HTML string if `as_html` is True, otherwise a Plotly Figure object.
    """
    fig = go.Figure(
        layout=go.Layout(
            xaxis={"title": {"text": "Signal"}},
            yaxis={"title": {"text": "Count"}},
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        )
    )

    fig.add_trace(go.Histogram(x=data))
    if as_html:
        return fig.to_html(full_html=False)
    return fig


def plot_channels_timeseries(
    gcamp_channel: npt.NDArray, isosb_channel: npt.NDArray, as_html: bool = False
) -> str | go.Figure:
    """Plots the timeseries of GCaMP and Isosb channels.

    Args:
        gcamp_channel (npt.NDArray): The GCaMP channel timeseries data
        isosb_channel (npt.NDArray): The Isosb channel timeseries data
        as_html (bool, optional): If True, returns the plot as an HTML string.

    Returns:
        str | go.Figure: The plot as an HTML string if `as_html` is True, otherwise a Plotly Figure object.
    """
    fig = go.Figure(
        layout=go.Layout(
            xaxis={"title": {"text": "Frame index"}},
            yaxis={"title": {"text": "Signal"}},
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        )
    )

    fig.add_trace(go.Scatter(y=gcamp_channel, mode="lines", name="GCaMP channel"))
    fig.add_trace(go.Scatter(y=isosb_channel, mode="lines", name="Isosb channel"))
    if as_html:
        return fig.to_html(full_html=False)
    return fig


def plot_projection(data: npt.NDArray, as_html: bool = False) -> str | go.Figure:
    """Plots a channel projection using Plotly.

    Args:
        data (npt.NDArray): The projection data to plot as a NumPy array.
        as_html (bool, optional): If True, returns the plot as an HTML string. If False, returns a Plotly Figure object.

    Returns:
        str | go.Figure: The plot as an HTML string if `as_html` is True, otherwise a Plotly Figure object.
    """
    if as_html:
        return px.imshow(data).to_html(full_html=False)
    return px.imshow(data)


def plot_frame_ids(gcamp_ids: npt.NDArray, isosb_ids: npt.NDArray, as_html: bool = False) -> str | go.Figure:
    """Plots the frame IDs of GCaMP and Isosb channels.

    Args:
        gcamp_ids (npt.NDArray): The GCaMP channel frame IDs.
        isosb_ids (npt.NDArray): The Isosb channel frame IDs.
        as_html (bool, optional): If True, returns the plot as an HTML string. If False, returns a Plotly Figure object.

    Returns:
        str | go.Figure: The plot as an HTML string if `as_html` is True, otherwise a Plotly Figure object.
    """
    fig = go.Figure(
        layout=go.Layout(
            xaxis={"title": {"text": "Processed frame index"}},
            yaxis={"title": {"text": "Original frame index"}},
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        )
    )

    fig.add_trace(go.Scatter(y=gcamp_ids, mode="lines", name="GCaMP channel IDs"))
    fig.add_trace(go.Scatter(y=isosb_ids, mode="lines", name="Isosb channel IDs"))
    if as_html:
        return fig.to_html(full_html=False)
    return fig


def plot_filter_pie(gcamp_ids: npt.NDArray, isosb_ids: npt.NDArray, as_html: bool = False) -> str | go.Figure:
    """Plots a pie chart of the frame IDs for GCaMP and Isosb channels.

    Args:
        gcamp_ids (npt.NDArray): The GCaMP channel frame IDs.
        isosb_ids (npt.NDArray): The Isosb channel frame IDs.
        as_html (bool, optional): If True, returns the plot as an HTML string. If False, returns a Plotly Figure object.

    Returns:
        str | go.Figure: The plot as an HTML string if `as_html` is True, otherwise a Plotly Figure object.
    """
    labels = ["GCaMP frame IDs", "Isosb frame IDs"]
    values = [len(gcamp_ids), len(isosb_ids)]
    fig = go.Figure(
        go.Pie(labels=labels, values=values),
        layout=go.Layout(
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        ),
    )
    if as_html:
        return fig.to_html(full_html=False)
    return fig


def plot_channels_dff(gcamp_channel: npt.NDArray, isosb_channel: npt.NDArray, as_html: bool = False) -> str | go.Figure:
    """Plots the ∆F/F time series of GCaMP and Isosb channels.

    Args:
        gcamp_channel (npt.NDArray): The GCaMP channel ∆F/F time series data.
        isosb_channel (npt.NDArray): The Isosb channel ∆F/F time series data.
        as_html (bool, optional): If True, returns the plot as an HTML string. If False, returns a Plotly Figure object.

    Returns:
        str | go.Figure: The plot as an HTML string if `as_html` is True, otherwise a Plotly Figure object.
    """
    fig = go.Figure(
        layout=go.Layout(
            xaxis={"title": {"text": "Frame index"}},
            yaxis={"title": {"text": "∆F/F"}},
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        )
    )

    fig.add_trace(go.Scatter(y=gcamp_channel, mode="lines", name="GCaMP channel ∆F/F"))
    fig.add_trace(go.Scatter(y=isosb_channel, mode="lines", name="Isosb channel ∆F/F"))
    if as_html:
        return fig.to_html(full_html=False)
    return fig


def plot_corrected_dff(data: npt.NDArray, as_html: bool = False) -> str | go.Figure:
    """Plots the corrected ∆F/F time series.

    Args:
        data (npt.NDArray): The corrected ∆F/F time series data.
        as_html (bool, optional): If True, returns the plot as an HTML string. If False, returns a Plotly Figure object.

    Returns:
        str | go.Figure: The plot as an HTML string if `as_html` is True, otherwise a Plotly Figure object.
    """
    fig = go.Figure(
        layout=go.Layout(
            xaxis={"title": {"text": "Frame index"}},
            yaxis={"title": {"text": "∆F/F"}},
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        )
    )

    fig.add_trace(go.Scatter(y=data, mode="lines", name="Corrected ∆F/F"))
    if as_html:
        return fig.to_html(full_html=False)
    return fig


def plot_frame(data: npt.NDArray, as_html: bool = False) -> str | go.Figure:
    """Plots a single frame of data using Plotly.

    Args:
        data (npt.NDArray): The frame data to plot, as a NumPy array.
        as_html (bool, optional): If True, returns the plot as an HTML string. If False, returns a Plotly Figure object.

    Returns:
        str | go.Figure: The plot as an HTML string if `as_html` is True, otherwise a Plotly Figure object.
    """
    if as_html:
        return px.imshow(data).to_html(full_html=False)
    return px.imshow(data)
