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
import csv
import typing
from collections import OrderedDict

import h5py
import numpy.typing as npt
import xmltodict
import zarr
from dask import array as da
from pynwb import NWBHDF5IO
from pynwb import NWBFile


@typing.overload
def read_nwb(path: str, mode: str = "a") -> NWBFile: ...  # pyright: ignore[reportOverlappingOverload]


@typing.overload
def read_nwb(path: str, mode: str = "a", return_io: bool = True) -> tuple[NWBFile, NWBHDF5IO]: ...


def read_nwb(path: str, mode: str = "a", return_io: bool = False) -> NWBFile | tuple[NWBFile, NWBHDF5IO]:
    """Read an NWB file.

    Args:
        path (str): Path to the NWB file.
        mode (str, optional): File read mode (i.e. read/write/append). Defaults to "a".
        return_io (bool, optional): Return IO object alongside the NWB file object. Defaults to False.

    Returns:
        NWBFile: NWB file object.
        NWBHDF5IO: IO object (if return_io=True).
    """
    io = NWBHDF5IO(path, mode=mode)
    nwbfile = io.read()
    if return_io:
        return nwbfile, io
    return nwbfile


def write_nwb(path: str, nwbfile: NWBFile, mode: str = "w", io: NWBHDF5IO = None, **kwargs) -> None:
    """Write an NWB file.

    Args:
        path (str): Path to the NWB file.
        nwbfile (NWBFile): NWB file object.
        mode (str, optional): File write mode (i.e. write/append). Defaults to "w".
        **kwargs: Parameters passed to NWBHDF5IO.write.
    """
    if io:
        return io.write(nwbfile, **kwargs)
    with NWBHDF5IO(path, mode=mode) as io:
        return io.write(nwbfile, **kwargs)


def read_h5(path: str) -> h5py.File:
    """Read an HDF5 file.

    Args:
        path (str): Path to the HDF5 file.

    Returns:
        h5py.File: HDF5 file object.
    """
    return h5py.File(path, "r")


def write_h5(path: str, data: dict, compression: str = "lzf", attributes: dict = {}) -> str:
    """Write a dictionary to an HDF5 file.

    Args:
        path (str): Path to the HDF5 file.
        data (dict): Dictionary containing datasets to write in {'dataset_name': data_array} format.
        compression (str, optional): Compression method for the datasets. Defaults to "lzf".
        attributes (dict, optional): Attributes to write to the HDF5 file. Defaults to {}.

    Returns:
        str: Path to the written HDF5 file.

    Example:
        >>> data = {"dataset1": np.array([1, 2, 3]), "dataset2": np.array([[1, 2], [3, 4]])}
        >>> write_h5("output.h5", data)
    """
    with h5py.File(path, "w") as h5file:
        for key, value in data.items():
            h5file.create_dataset(key, data=value, compression=compression)
        if attributes:
            h5file.attrs.update(attributes)
    return path


def store_interim(
    array: da.Array | npt.ArrayLike,
    interim_path: str,
    compute: bool = True,
    chunks: int = 500,
) -> zarr.core.Array:
    """Store an array in an interim Zarr file.

    Args:
        array (Dask or Numpy Array): Dask or Numpy array to persist on disk.
        interim_path (str): Path to the interim Zarr file. The .zarr extension is added automatically.
        compute (bool, optional): Whether to compute the array before storing, applies only to Dask arrays. Defaults to True.
        chunks (int, optional): Chunk size for the Zarr file. Defaults to 500.

    Returns:
        Zarr Array: Persistent Zarr array object
    """
    if not interim_path.endswith(".zarr"):
        interim_path += ".zarr"

    if isinstance(array, da.Array):
        z_interim = zarr.open_array(
            interim_path,
            shape=array.shape,
            dtype=array.dtype,
            chunks=(chunks, array.shape[1], array.shape[2]),
        )
        return array.store(z_interim, return_stored=True, compute=compute)  # type: ignore

    zarr.save(interim_path, array)
    return zarr.load(interim_path)


def read_points(path: str) -> dict[str, tuple[float, float]]:
    """Read a landmark points file.

    Args:
        path (str): Path to the points file.

    Returns:
        dict[str, tuple[float, float]]: Dictionary with the landmark names as keys and their x-y coordinates
    """
    if path.endswith(".xml") or path.endswith(".points"):
        return _read_fiji_points(path)
    if path.endswith(".csv"):
        return _read_csv_points(path)
    raise ValueError("Unsupported file format.")


def _read_fiji_points(path: str) -> dict[str, tuple[float, float]]:
    """Read a FIJI landmark points file.

    Args:
        path (str): Path to the points file.

    Returns:
        dict[str, tuple[float, float]]: Dictionary with the landmark names as keys and their x-y coordinates
    """
    with open(path) as fp:
        points = xmltodict.parse(fp.read())
        points = OrderedDict(
            {
                point["@name"]: (float(point["@y"]), float(point["@x"]))
                for point in points["namedpointset"]["pointworld"]
            }
        )

    return points


def _read_csv_points(path: str) -> dict[str, tuple[float, float]]:
    """Read a CSV file with landmark points.

    Args:
        path (str): Path to the CSV file.

    Returns:
        dict[str, tuple[float, float]]: Dictionary with the landmark names as keys and their x-y coordinates
    """
    with open(path) as fp:
        csv_reader = csv.DictReader(fp)
        points = OrderedDict({row["landmark"]: (float(row["y"]), float(row["x"])) for row in csv_reader})

    return points


def write_points(path: str, points: dict[str, tuple[float, float]]) -> None:
    """Write a dictionary of landmark points to a CSV file.

    Args:
        path (str): Path to output CSV file.
        points (dict[str, tuple[float, float]]): Dictionary with the landmark names as keys and their x-y coordinates
    """
    if not path.endswith(".csv"):
        path += ".csv"
    with open(path, "w") as fp:
        csv_writer = csv.DictWriter(fp, fieldnames=["landmark", "y", "x"])
        csv_writer.writeheader()
        for landmark, (y, x) in points.items():
            csv_writer.writerow({"landmark": landmark, "y": y, "x": x})
