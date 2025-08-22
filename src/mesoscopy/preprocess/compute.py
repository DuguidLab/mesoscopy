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
import numpy as np
import numpy.typing as npt
import zarr
from dask import array as da

from mesoscopy import io


def bin_array(
    array: da.Array | npt.NDArray,
    bins: int,
    interim_dir: str = ".",
    session_id: str = "null",
) -> zarr.Array:
    """Bin a 3D image array across its x and y axes.

    The function bins the width and height of a 3D image array by a factor of `bins`. It does not bin the z-axis (time).

    Args:
        array (Dask or NumPy Array): Array to be binned.
        bins (int): Number of bins in x and y directions (i.e. width and height).
        interim_dir (str or PathLike object, optional): Directory to store interim binned array data.
            Defaults to current working directory (".").
        session_id (str, optional): Session identifier for interim path. Defaults to "null".

    Returns:
        zarr.core.Array: Binned array as a persistent Zarr array object.
    """
    binned_array = array.reshape(  # pyright: ignore[reportAttributeAccessIssue]
        array.shape[0],  # pyright: ignore[reportArgumentType]
        1,
        int(array.shape[1] / bins),
        int(array.shape[1] // (array.shape[1] / bins)),
        int(array.shape[2] / bins),
        int(array.shape[2] // (array.shape[2] / bins)),
    ).mean(axis=(-1, 1, 3), dtype=np.float32)
    interim_path = f"{interim_dir}{os.sep}{session_id}_binned"
    return io.store_interim(binned_array, interim_path)


def frame_statistics(array: da.Array | npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Calculate mean and standard deviation for each frame in a 3D image array.

    Args:
        array (Dask or NumPy Array): Imaging array to calculate statistics for.

    Returns:
        tuple[npt.NDArray, npt.NDArray]: Tuple containing two NumPy arrays: means and standard deviations for each frame.
    """
    if type(array) is np.ndarray:
        array = da.from_array(array, chunks=(100, array.shape[1], array.shape[2]))  # pyright: ignore[reportArgumentType]

    frame_means, frame_stds = dask.compute(  # pyright: ignore[reportPrivateImportUsage]
        array.mean(axis=(1, 2), dtype=np.float32),
        array.std(axis=(1, 2), dtype=np.float32),
    )

    return frame_means, frame_stds


def channel_separation_filters(
    frame_means: npt.NDArray,
    frame_stds: npt.NDArray,
    use_means: bool = False,
    flip_channels: bool = False,
) -> tuple[list, list]:
    """Generate filters for separating channels based on frame means and standard deviations.

    Args:
        frame_means (npt.NDArray): Array of mean values for each frame.
        frame_stds (npt.NDArray): Array of standard deviation values for each frame.
        use_means (bool, optional): Use means instead of standard deviations for filtering. Defaults to False.
        flip_channels (bool, optional): Flip the channels. Defaults to False.

    Returns:
        tuple[list, list]: Tuple of two lists, containing the frame indices for each channel.
    """
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


def rolling_dff(
    array: da.Array | npt.NDArray,
    window_width: int = 750,
    channel_name: str = "null",
    interim_dir: str = ".",
    session_id: str = "null",
) -> zarr.Array:
    """Calculate dF/F using a rolling window.

    Args:
        array (Dask or NumPy Array): Array to be separated. If array is a multi-channel recording,
            it needs to be filtered before it's passed to this function.
        window_width (int, optional): Window width for dF/F calculation. Defaults to 750.
        channel_name (str, optional): Channel name for interim path. Defaults to "null".
        interim_dir (str, optional): Directory to store interim dF/F data. Defaults to current working directory (".").
        session_id (str, optional): Session identifier for interim path. Defaults to "null".

    Returns:
        zarr.core.Array: dF/F array as a persistent Zarr array object.

    Raises:
        ValueError: If the window width is greater than the number of frames in the array.
    """
    if type(array) is np.ndarray:
        array = da.from_array(array, chunks=(100, array.shape[1], array.shape[2]))  # pyright: ignore[reportArgumentType]

    if window_width > len(array):
        msg = "Window width must be less than the number of frames."
        raise ValueError(msg)

    # If window width is an odd number, add 1 to make it even to avoid broadcast errors
    if window_width % 2 != 0:
        window_width += 1

    cumsum_vec = da.cumsum(array, dtype=np.uint32, axis=0)

    interim_path = interim_dir + os.sep + session_id + "_" + channel_name + "_cumsum"
    cumsum_vec = io.store_interim(cumsum_vec, interim_path)

    f0 = da.true_divide(
        (cumsum_vec[window_width:] - cumsum_vec[:-window_width]),
        window_width,
        dtype=np.float32,
    )

    interim_path = interim_dir + os.sep + session_id + "_" + channel_name + "_f0"
    f0 = io.store_interim(f0, interim_path)

    f0_start = da.mean(f0[: window_width // 2], axis=0).compute()
    padding_start = da.zeros((window_width // 2, *array.shape[1:3])) + f0_start
    f0_end = da.mean(f0[-(window_width // 2) :], axis=0).compute()
    padding_end = da.zeros((window_width // 2, *array.shape[1:3])) + f0_end

    f0 = da.insert(f0, [0], padding_start, axis=0)
    f0 = da.insert(f0, [f0.shape[0]], padding_end, axis=0)

    interim_path = interim_dir + os.sep + session_id + "_" + channel_name + "_f0_appended"
    f0 = io.store_interim(f0, interim_path)

    dff = da.true_divide(da.subtract(array, f0), f0, dtype=np.float32)

    interim_path = interim_dir + os.sep + session_id + "_" + channel_name + "_dff"

    return io.store_interim(dff, interim_path)


def projections(
    array: da.Array | npt.NDArray,
) -> dict[str, npt.NDArray]:
    """Calculate mean, standard deviation and maximum intensity projection frames.

    Args:
        array (npt.NDArray): Frame array to calculate projections for.

    Returns:
        dict[str, npt.NDArray]: Dictionary containing mean, standard deviation and maximum intensity projection frames.
    """
    mean_frame, std_frame, maxip = dask.compute(  # pyright: ignore[reportPrivateImportUsage]
        array.mean(axis=0),
        array.std(axis=0),
        array.max(axis=0),
    )

    return {
        "mean": mean_frame,
        "std": std_frame,
        "maxip": maxip,
    }
