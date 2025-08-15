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
"""Module for quality assurance (QA) functions in the mesoscopy registration pipeline."""

import numpy.typing as npt
import plotly.express as px
import plotly.graph_objects as go


def plot_landmarks(
    source_landmarks: npt.NDArray,
    target_landmarks: npt.NDArray,
    as_html: bool = False,
) -> str | go.Figure:
    fig = go.Figure(
        layout=go.Layout(
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
            yaxis={
                "autorange": "reversed",
            },
        )
    )

    fig.add_trace(
        go.Scatter(x=source_landmarks[:, 1], y=source_landmarks[:, 0], mode="markers", name="Source landmarks")
    )
    fig.add_trace(
        go.Scatter(x=target_landmarks[:, 1], y=target_landmarks[:, 0], mode="markers", name="Target landmarks")
    )

    if as_html:
        return fig.to_html(full_html=False)
    return fig


def plot_frame(data: npt.NDArray, landmarks=npt.NDArray | None, as_html: bool = False) -> str | go.Figure:
    """Plots a single frame of data using Plotly.

    Args:
        data (npt.NDArray): The frame data to plot, as a NumPy array.
        as_html (bool, optional): If True, returns the plot as an HTML string. If False, returns a Plotly Figure object.

    Returns:
        str | go.Figure: The plot as an HTML string if `as_html` is True, otherwise a Plotly Figure object.
    """
    fig = px.imshow(data)
    fig.update_layout(
        {
            "margin": {"l": 20, "r": 20, "t": 20, "b": 20},
        }
    )

    if landmarks:
        fig.add_trace(
            go.Scatter(
                x=landmarks[:, 1],
                y=landmarks[:, 0],
                mode="markers",
                name="Template landmarks",
                marker={"color": "Green"},
            )
        )

    if as_html:
        return fig.to_html(full_html=False)
    return fig
