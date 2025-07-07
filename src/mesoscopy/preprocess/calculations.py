#  Copyright (c) 2024 Constantinos Eleftheriou <Constantinos.Eleftheriou@ed.ac.uk>.
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

import os
import dask
import zarr
import numpy as np
import numpy.typing as npt
from dask import array as da

import mesoscopy.io as io
import mesoscopy.plots as plots


def bin_array(
    array: da.Array | npt.NDArray,
    bins: int,
    interim_dir: str = ".",
    session_id: str = "null",
) -> zarr.core.Array:
    """Bin a 3D image array across its x and y axes.

    The function bins the width and height of a 3D image array by a factor of `bins`. It does not bin the z-axis (time).

    Args:
        array (Dask or NumPy Array): Array to be binned.
        bins (int): Number of bins in x and y directions (i.e. width and height).
        interim_dir (str or PathLike object, optional): Directory to store interim binned array data. Defaults to current working directory (".").
        session_id (str, optional): Session identifier for interim path. Defaults to "null".

    Returns:
        zarr.core.Array: Binned array as a persistent Zarr array object.
    """
    binned_array = array.reshape(
        array.shape[0],
        1,
        int(array.shape[1] / bins),
        int(array.shape[1] // (array.shape[1] / bins)),
        int(array.shape[2] / bins),
        int(array.shape[2] // (array.shape[2] / bins)),
    ).mean(axis=(-1, 1, 3), dtype=np.float32)
    interim_path = f"{interim_dir}{os.sep}{session_id}_binned"
    return io.store_interim(binned_array, interim_path)


def separate_channels(
    array: da.Array | npt.NDArray,
    qa_dir: str = ".",
    session_id: str = "null",
    use_means: bool = False,
    flip_channels: bool = False,
) -> tuple[list, list]:
    """Separate channels in a mixed-channel array.

    Args:
        array (Dask or NumPy Array): Array to be separated.
        qa_dir (str or PathLike object, optional): Directory to store QA plots. Defaults to current working directory (".").
        session_id (str, optional): Session identifier for interim path. Defaults to "null".
        use_means (bool, optional): Use means instead of standard deviations for filtering. Defaults to False.
        flip_channels (bool, optional): Flip the channels. Defaults to False.

    Returns:
        tuple[list, list]: Tuple of two lists, containing the frame indices for each channel.
    """
    frame_means, frame_stds = dask.compute(
        array.mean(axis=(1, 2), dtype=np.float32),
        array.std(axis=(1, 2), dtype=np.float32),
    )

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

    threshold = frame_stds.mean()
    gcamp_filter = np.nonzero(frame_stds > threshold)[0].tolist()
    isosb_filter = np.nonzero(frame_stds < threshold)[0].tolist()

    if use_means:
        threshold = frame_means.mean()
        gcamp_filter = np.nonzero(frame_means > threshold)[0].tolist()
        isosb_filter = np.nonzero(frame_means < threshold)[0].tolist()

    if flip_channels:
        return isosb_filter, gcamp_filter

    return gcamp_filter, isosb_filter


def channel_dff(
    array: da.Array | npt.NDArray,
    channel_filter: list,
    window_width: int = 750,
    channel_name: str = "null",
    interim_dir: str = ".",
    session_id: str = "null",
) -> zarr.core.Array:
    """Calculate dF/F for a channel in a mixed-channel array.

    Args:
        array (Dask or NumPy Array): Array to be separated.
        channel_filter (list): List of frame indices for the channel.
        window_width (int, optional): Window width for dF/F calculation. Defaults to 750.
        channel_name (str, optional): Channel name for interim path. Defaults to "null".
        interim_dir (str, optional): Directory to store interim dF/F data. Defaults to current working directory (".").
        session_id (str, optional): Session identifier for interim path. Defaults to "null".

    Returns:
        zarr.core.Array: dF/F array as a persistent Zarr array object.
    """
    if type(array) == np.ndarray:
        array = da.from_array(array, chunks=(100, array.shape[1], array.shape[2]))

    if window_width > len(array):
        raise ValueError("Window width must be less than the number of frames.")

    # If window width is an odd number, add 1 to make it even to avoid broadcast errors
    if window_width % 2 != 0:
        window_width = window_width + 1

    cumsum_vec = da.cumsum(array[channel_filter], dtype=np.uint32, axis=0)

    interim_path = interim_dir + os.sep + session_id + "_" + channel_name + "_cumsum"
    cumsum_vec = io.store_interim(cumsum_vec, interim_path)

    f0 = da.true_divide(
        (cumsum_vec[window_width:] - cumsum_vec[:-window_width]),
        window_width,
        dtype=np.float32,
    )

    interim_path = interim_dir + os.sep + session_id + "_" + channel_name + "_f0"
    f0 = io.store_interim(f0, interim_path)

    f0_start = da.mean(f0[: window_width // 2]).compute()
    padding_start = da.zeros((window_width // 2, *array.shape[1:3])) + f0_start
    f0_end = da.mean(f0[-(window_width // 2) :]).compute()
    padding_end = da.zeros((window_width // 2, *array.shape[1:3])) + f0_end

    f0 = da.insert(f0, [0], padding_start, axis=0)
    f0 = da.insert(f0, [f0.shape[0]], padding_end, axis=0)

    interim_path = (
        interim_dir + os.sep + session_id + "_" + channel_name + "_f0_appended"
    )
    f0 = io.store_interim(f0, interim_path)

    dff = da.true_divide(da.subtract(array[channel_filter], f0), f0, dtype=np.float32)

    interim_path = interim_dir + os.sep + session_id + "_" + channel_name + "_dff"

    return io.store_interim(dff, interim_path)
