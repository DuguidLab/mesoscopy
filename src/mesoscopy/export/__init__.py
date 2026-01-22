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
"""Export functions for mesoscopy-generated files."""

import os
import typing

import click
import imageio.v2 as iio
import numpy as np

import mesoscopy.export.nwb as exp_nwb
import mesoscopy.process as proc

from tqdm import tqdm


@click.group("export")
def export_cmd() -> None:
    """Export mesoscopy-generated files."""


@export_cmd.command("nwb")
@click.argument("nwb_path", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option(
    "--out-path",
    type=click.Path(dir_okay=False, path_type=str),
    default="",
    help="Path to save the exported NWB file. "
    "If not provided, the output file will be named as the input file with '_export.nwb' appended.",
)
def export_nwb_cmd(**kwargs: typing.Any) -> None:
    """Create a sharable copy of an NWB file by resolving external data links."""
    click.echo("Exporting NWB file...")
    export_path = export_nwb(**kwargs)
    click.echo(f"NWB file exported to {export_path}")


def export_nwb(nwb_path: str, out_path: str = "") -> str:
    """Create a sharable copy of an NWB file by resolving external data links.

    Args:
        nwb_path (str): Path to the source NWB file.
        out_path (str, optional): Path to save the exported NWB file. If not provided, the output file will be named as
            the input file with '_export.nwb' appended. Defaults to "".

    Returns:
        str: Path to the exported NWB file.
    """
    return exp_nwb.export_standalone(nwb_path, out_path)


@export_cmd.command("deltaf")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option(
    "--out-path",
    type=click.Path(dir_okay=True, path_type=str),
    default=".",
    help="Path to save the exported video file."
    "If not provided, the output file will be named as the input file with '_deltaf.mp4' appended.",
)
def export_deltaf_cmd(**kwargs: typing.Any) -> None:
    """Export delta F frames as a video file."""
    export_deltaf(**kwargs)


def export_deltaf(path: str, out_path: str) -> str:
    """Export delta F frames as a video file.

    Args:
        path (str): Path to the source file (HDF5 or NWB).
        out_path (str): Path to save the exported video file.

    Returns:
        str: Path to the exported video file.
    """
    session_id = path.split("/")[-1].replace(".h5", "").replace(".nwb", "")

    nwb = bool(path.endswith(".nwb"))

    click.echo("Loading delta F data...")
    _, deltaf, _ = proc.load_deltaf(path, nwb)

    # Rescale deltaf to positive integer values between 0 and 255
    click.echo("Rescaling delta F for export...")
    deltaf_scaled = np.array((deltaf + abs(deltaf.min())) / (deltaf + abs(deltaf.min())).max() * 255, dtype=np.uint8)

    out_path = f"{out_path}{os.sep}{session_id}_deltaf.mp4"
    video_writer = iio.get_writer(out_path, format="FFMPEG", fps=25, pixelformat="yuv420p", mode="I")

    for _, frame in tqdm(
        enumerate(deltaf_scaled),
        desc="Converting delta F trace to video",
        total=len(deltaf_scaled),
    ):
        video_writer.append_data(np.stack([frame] * 3, axis=2))

    return out_path
