#  Copyright (c) 2022-2025 Constantinos Eleftheriou <Constantinos.Eleftheriou@ed.ac.uk>.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this
#   software and associated documentation files (the "Software"), to deal in the
#   Software without restriction, including without limitation the rights to use, copy,
#   modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#   and to permit persons to whom the Software is furnished to do so, subject to the
#   following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies
#  or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#  HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
#  IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
#  IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

"""Video conversion utilities."""

import datetime
import os
import typing

import h5py as h5
import numpy as np
import pandas as pd
import pims
from tqdm import tqdm

import mesoscopy.convert.hdf5 as conv_h5


def to_hdf5(
    input_path: str,
    out_dir: str,
    ts_path: str = "",
    ts_delimiter: str = ",",
    ts_column: str | int = 0,
    ts_has_header: bool = False,
    **kwargs: typing.Any,
) -> str:
    video_data = pims.as_gray(pims.PyAVVideoReader(input_path))

    if ts_path:
        header_row = 0 if ts_has_header else None
        ts_df = pd.read_csv(
            ts_path,
            delimiter=ts_delimiter,
            header=header_row,
        )

        timestamps = ts_df.loc[:, ts_column].values
    else:
        timestamps = [str(datetime.datetime.now()).replace(" ", "T") for _ in range(len(video_data))]

    fname = input_path.split(os.sep)[-1].split(".")[0]
    out_path = f"{out_dir}{os.sep}{fname}.h5"

    with h5.File(out_path, "w") as f:
        f.create_dataset("timestamps", data=timestamps)
        f.create_dataset(
            "frames",
            shape=(len(video_data), *video_data[0].shape),
            chunks=(100, video_data[0].shape[0], video_data[0].shape[1]),
            dtype=np.int8,
        )

        for idx, frame in tqdm(
            enumerate(video_data),
            desc="Converting video to HDF5",
            total=len(video_data),
        ):
            f["frames"][idx] = frame

    return out_path


def to_nwb(input_path: str, **kwargs: typing.Any) -> str:
    h5_path = to_hdf5(input_path, **kwargs)

    return conv_h5.to_nwb(h5_path, **kwargs)
