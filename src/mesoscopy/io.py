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

import h5py
import zarr

import numpy.typing as npt
from dask import array as da

from typing import Tuple

from pynwb import NWBHDF5IO, NWBFile


def read_nwb(
    path: str, mode: str = "a", return_io: bool = False
) -> NWBFile | Tuple[NWBFile, NWBHDF5IO]:
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


def write_nwb(path: str, nwbfile: NWBFile, mode: str = "w") -> None:
    """Write an NWB file.

    Args:
        path (str): Path to the NWB file.
        nwbfile (NWBFile): NWB file object.
        mode (str, optional): File write mode (i.e. write/append). Defaults to "w".
    """
    with NWBHDF5IO(path, mode=mode) as io:
        io.write(nwbfile)


def read_h5(path: str) -> h5py.File:
    """Read an HDF5 file.

    Args:
        path (str): Path to the HDF5 file.

    Returns:
        h5py.File: HDF5 file object.
    """
    return h5py.File(path, "r")


def write_h5():
    raise NotImplementedError


def store_interim(
    array: da.Array | npt.ArrayLike,
    interim_path: str,
    compute: bool = True,
    chunks: int = 500,
) -> zarr.core.Array:
    """Store an array in an interim Zarr file.

    Args:
        array (Dask or Numpy Array): Dask or Numpy array to persist on disk.
        interim_path (str): Path to the interim Zarr file.
        compute (bool, optional): Whether to compute the array before storing, applies only to Dask arrays. Defaults to True.
        chunks (int, optional): Chunk size for the Zarr file. Defaults to 500.

    Returns:
        Zarr Array: Persistent Zarr array object
    """
    if isinstance(array, da.Array):
        z_interim = zarr.open_array(
            interim_path,
            shape=array.shape,
            dtype=array.dtype,
            chunks=(chunks, array.shape[1], array.shape[2]),
        )
        return array.store(z_interim, return_stored=True, compute=compute)

    zarr.save(interim_path, array)
    return zarr.load(interim_path)
