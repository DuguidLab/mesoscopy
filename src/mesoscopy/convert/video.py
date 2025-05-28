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

import typing
import skvideo.io
import pandas as pd

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
    video_data = skvideo.io.vread(input_path)

    if ts_path:
        header_row = 0 if ts_has_header else None
        ts_df = pd.read_csv(
            ts_path,
            delimiter=ts_delimiter,
            header=header_row,
        )

        timestamps = pd.to_datetime(ts_df.loc[:, ts_column]).values

    return ""


def to_nwb(input_path: str, **kwargs: typing.Any) -> str:
    h5_path = to_hdf5(input_path, **kwargs)

    return conv_h5.to_nwb(h5_path, **kwargs)
