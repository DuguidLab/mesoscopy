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

import os
import mesoscopy.io as io
import mesoscopy.plots as plots

import numpy.typing as npt


def plot_frame_statistics(
    frame_means: npt.NDArray, frame_stds: npt.NDArray, qa_dir: str = ".", session_id: str = "null"
) -> None:
    """Plot and save quality assurance (QA) statistics for frame means and standard deviations.

    Args:
        qa_dir (str): Directory to save QA plots.
        frame_means (npt.NDArray): Array of mean values for each frame.
        frame_stds (npt.NDArray): Array of standard deviation values for each frame.
        session_id (str, optional): Session identifier for interim path. Defaults to "null".
    """
    outpath = qa_dir + os.sep + session_id + "_qa_frame_means_histogram.png"
    msg = "Saved histogram for frame means at {}".format(outpath)
    plots.plot_hist(frame_means, outpath, message=msg)

    outpath = qa_dir + os.sep + session_id + "_qa_frame_means_line.png"
    msg = "Saved lineplot for frame means at {}".format(outpath)
    plots.plot_line(frame_means, outpath, message=msg)

    outpath = qa_dir + os.sep + session_id + "_qa_frame_std_histogram.png"
    msg = "Saved histogram for frame means at {}".format(outpath)
    plots.plot_hist(frame_stds, outpath, message=msg)

    outpath = qa_dir + os.sep + session_id + "_qa_frame_std_line.png"
    msg = "Saved lineplot for frame means at {}".format(outpath)
    plots.plot_line(frame_stds, outpath, message=msg)


def plot_dual_channel_timeseries(
    channel_arrays: tuple[npt.NDArray, npt.NDArray], qa_dir: str = ".", session_id: str = "null"
) -> None:
    outpath = qa_dir + os.sep + session_id + "_qa_channel_means.png"
    plots.plot_lines(
        [channel_arrays[0], channel_arrays[1]],
        outpath,
        message="Saved channel means at {}".format(outpath),
    )


def dff_qa_plots(): ...


def signal_correction_qa_plots(): ...
