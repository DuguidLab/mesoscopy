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
import mesoscopy.plots as plots

import numpy.typing as npt


def plot_frame_statistics(
    frame_means: npt.NDArray, frame_stds: npt.NDArray, qa_dir: str = ".", session_id: str = "null"
) -> dict[str, str]:
    """Plot and save quality assurance (QA) statistics for frame means and standard deviations.

    Args:
        qa_dir (str): Directory to save QA plots.
        frame_means (npt.NDArray): Array of mean values for each frame.
        frame_stds (npt.NDArray): Array of standard deviation values for each frame.
        session_id (str, optional): Session identifier for interim path. Defaults to "null".

    Returns:
        dict[str, str]: Dictionary containing paths to the saved QA plots.
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

    return {
        "frame_means_histogram": outpath,
        "frame_means_line": outpath,
        "frame_stds_histogram": outpath,
        "frame_stds_line": outpath,
    }


def plot_dual_channel_timeseries(
    channel_arrays: tuple[npt.NDArray, npt.NDArray], qa_dir: str = ".", session_id: str = "null"
) -> dict[str, str]:
    """Plot and save dual channel timeseries for quality assurance (QA).

    Args:
        channel_arrays (tuple[npt.NDArray, npt.NDArray]): Tuple containing the two channel arrays.
        qa_dir (str): Directory to save QA plots.
        session_id (str, optional): Session identifier for interim path. Defaults to "null".

    Returns:
        dict[str, str]: Dictionary containing paths to the saved QA plots.
    """
    if len(channel_arrays) != 2:
        raise ValueError("Expected a tuple of two channel arrays.")

    outpath = qa_dir + os.sep + session_id + "_qa_channel_means.png"
    plots.plot_lines(
        [channel_arrays[0], channel_arrays[1]],
        outpath,
        message="Saved channel means at {}".format(outpath),
    )

    return {
        "channel_means": outpath,
    }


def plot_channel_projection_images(
    channel_array: npt.NDArray, qa_dir: str = ".", session_id: str = "null", channel: str = "null"
) -> dict[str, str]:
    """Plot and save channel mean, standard deviation and maximum intensity projection frames.
    Args:
        channel_array (npt.NDArray): Array containing channel statistics.
        qa_dir (str): Directory to save QA plots.
        session_id (str, optional): Session identifier for interim path. Defaults to "null".
        channel (str, optional): Name of the channel for labeling the plots. Defaults to "null".

    Returns:
        dict[str, str]: Dictionary containing paths to the saved QA plots.
    """
    mean_frame, std_frame, maxip = dask.compute(
        channel_array.mean(axis=0),
        channel_array.std(axis=0),
        channel_array.max(axis=0),
    )

    outpath = qa_dir + os.sep + session_id + "_qa_{}_mean.png".format(channel)
    plots.plot_frame(mean_frame, outpath, message="Saved mean frame at {}".format(outpath))

    outpath = qa_dir + os.sep + session_id + "_qa_{}_std.png".format(channel)
    plots.plot_frame(std_frame, outpath, message="Saved std frame at {}".format(outpath))

    outpath = qa_dir + os.sep + session_id + "_qa_{}_maxip.png".format(channel)
    plots.plot_frame(maxip, outpath, message="Saved maxip frame at {}".format(outpath))

    return {
        f"{channel}_mean_projection": outpath,
        f"{channel}_std_projection": outpath,
        f"{channel}_maxip_projection": outpath,
    }


def plot_dual_dff_timeseries(
    dff_arrays: tuple[npt.NDArray, npt.NDArray], qa_dir: str = ".", session_id: str = "null"
) -> dict[str, str]:
    """Plot and save dual channel DFF timeseries.

    Args:
        dff_arrays (tuple[npt.NDArray, npt.NDArray]): Tuple containing the ∆F for each channel.
        qa_dir (str): Directory to save QA plots.
        session_id (str, optional): Session identifier for interim path. Defaults to "null".

    Returns:
        dict[str, str]: Dictionary containing paths to the saved QA plots.
    """
    outpath = qa_dir + os.sep + session_id + "_qa_dual_dff_timeseries.png"
    plots.plot_lines(
        [dff_arrays[0], dff_arrays[1]],
        outpath,
        message="Saved dual channel DFF timeseries at {}".format(outpath),
    )

    return {
        "dual_dff_timeseries": outpath,
    }


def plot_f_example(f_signal: npt.NDArray, qa_dir: str = ".", session_id: str = "null") -> dict[str, str]:
    """Plot and save an example frame of the F signal.

    Args:
        f_signal (npt.NDArray): Array containing the F signal.
        qa_dir (str): Directory to save QA plots.
        session_id (str, optional): Session identifier for interim path. Defaults to "null".

    Returns:
        dict[str, str]: Dictionary containing paths to the saved QA plots.
    """
    outpath = qa_dir + os.sep + session_id + "_qa_f_example.png"
    plots.plot_frame(f_signal[200], outpath, message="Saved F signal example at {}".format(outpath))

    return {
        "f_example": outpath,
    }


def plot_mean_f_timeseries(f_signal: npt.NDArray, qa_dir: str = ".", session_id: str = "null") -> dict[str, str]:
    """Plot and save the mean F signal timeseries.

    Args:
        f_signal (npt.NDArray): Array containing the F signal.
        qa_dir (str): Directory to save QA plots.
        session_id (str, optional): Session identifier for interim path. Defaults to "null".

    Returns:
        dict[str, str]: Dictionary containing paths to the saved QA plots.
    """
    outpath = qa_dir + os.sep + session_id + "_qa_mean_f_timeseries.png"
    plots.plot_line(f_signal.mean(axis=(1, 2)), outpath, message="Saved mean F timeseries at {}".format(outpath))

    return {
        "mean_f_timeseries": outpath,
    }
