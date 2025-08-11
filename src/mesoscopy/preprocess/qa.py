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

import numpy.typing as npt
import plotly.express as px
import plotly.graph_objects as go


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
