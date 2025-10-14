#  Copyright (c) 2022 Constantinos Eleftheriou <Constantinos.Eleftheriou@ed.ac.uk>.
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

"""Processing submodule."""

import os
import click
import numpy as np
import dask.array as da
from pynwb.image import ImageSeries

import mesoscopy.timer as timer

import mesoscopy.io as io
import mesoscopy.process.smooth as psm
import mesoscopy.process.zscore as pzs


@click.group("process")
def process_cmd(): ...


@process_cmd.command("smooth")
@click.argument(
    "path",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--out_dir",
    type=click.Path(dir_okay=True),
    default="./",
    help="Output directory for smoothed recording.",
)
@click.option(
    "-s",
    "--sigma",
    type=int,
    default=2,
    help="Output directory for smoothed recording.",
)
def smooth_cmd(path: str, out_dir: str, sigma: int = 2) -> None:
    """Generate a smoothed DeltaF/F recording using a Laplace of Gaussian filter."""
    if not os.path.exists(out_dir):
        click.echo(f"Creating output directory {out_dir}...")
        os.makedirs(out_dir)

    click.echo(f"Loading preprocessed recording from {path}...")
    # Determine whether we're working with an NWB file
    nwb = bool(path.endswith(".nwb"))
    session_id, deltaf_series, timestamps = load_deltaf(path, nwb=nwb)

    outpath = out_dir + os.sep + session_id + "_smoothed.h5"

    with timer.Timer(message="Smoothing with LoG"):
        smoothed_deltaf = psm.laplace_gaussian(deltaf_series)

        outpath = io.write_h5(
            path=outpath,
            data={
                "/F": smoothed_deltaf,
                "/timestamps": timestamps,
            },
        )
    click.echo(f"Saved smoothed recording at {outpath}")


@process_cmd.command("zscore")
@click.argument(
    "path",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--out_dir",
    type=click.Path(dir_okay=True),
    default="./",
    help="Output directory for smoothed recording.",
)
def zscore_cmd(path: str, out_dir: str) -> None:
    """Pixel-wise z-score ∆F/F signal."""
    if not os.path.exists(out_dir):
        click.echo(f"Creating output directory {out_dir}...")
        os.makedirs(out_dir)

    click.echo(f"Loading preprocessed recording from {path}...")
    # Determine whether we're working with an NWB file
    nwb = bool(path.endswith(".nwb"))
    session_id, deltaf_series, timestamps = load_deltaf(path, nwb=nwb)

    h5_outpath = out_dir + os.sep + session_id + "_zscored.h5"
    with timer.Timer(message="Z-scoring DeltaF/F"):
        zscored = pzs.zscore_deltaf(deltaf_series)
        h5_outpath = io.write_h5(
            path=h5_outpath,
            data={
                "/F": zscored,
                "/timestamps": timestamps,
            },
        )

    click.echo(f"Saved z-scored recording at {h5_outpath}")

    # Append to NWB file
    if nwb:
        click.echo("Appending to NWB file...")
        nwbfile, nwbio = io.read_nwb(path, return_io=True)
        f = io.read_h5(h5_outpath)
        try:
            ophys_module = nwbfile.create_processing_module(
                name="ophys", description="optical physiology processed data"
            )
        except ValueError:
            click.echo("Processing module already exists...")
            ophys_module = nwbfile.processing["ophys"]

        zscored_series = ImageSeries(
            name="zScoredDeltaF",
            data=f["/F"],
            timestamps=f["/timestamps"],
            unit="df/f",
            description="z-scored dF/F widefield cortical imaging series",
        )

        ophys_module.add(zscored_series)

        io.write_nwb(path, nwbfile, io=nwbio)


def load_deltaf(path: str, nwb: bool = False) -> tuple[str, np.ndarray, np.ndarray]:
    """Load preprocessed deltaf from an HDF5 or NWB file.

    Args:
        path (str): Path to the preprocessed file.
        nwb (bool, optional): Whether the file is an NWB file. Defaults to False.

    Returns:
        tuple[str, np.ndarray, np.ndarray]: Session identifier, dF/F series, and timestamps.
    """
    if nwb:
        nwbfile = io.read_nwb(path)
        session_id = nwbfile.identifier
        deltaf_series = nwbfile.processing["ophys"]["DeltaFSeries"].data
        timestamps = nwbfile.processing["ophys"]["DeltaFSeries"].timestamps
    else:
        session_id = path.split("/")[-1].replace(".h5", "")
        f_preproc = io.read_h5(path)
        deltaf_series = f_preproc["/F"]
        timestamps = f_preproc["/timestamps"]

    return session_id, np.array(deltaf_series), np.array(timestamps)
