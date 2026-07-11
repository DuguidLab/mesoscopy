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
from pathlib import Path

import click
import pandas as pd
from pynwb.image import ImageSeries

import mesoscopy.process.region as pr
import mesoscopy.process.smooth as psm
import mesoscopy.process.zscore as pzs
from mesoscopy import io
from mesoscopy import timer


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
        Path(out_dir).mkdir(parents=True)

    click.echo(f"Loading preprocessed recording from {path}...")
    # Determine whether we're working with an NWB file
    nwb = bool(path.endswith(".nwb"))
    session_id, deltaf_series, timestamps = io.load_deltaf(path, nwb=nwb)

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
    if not Path(out_dir).exists():
        click.echo(f"Creating output directory {out_dir}...")
        Path(out_dir).mkdir(parents=True)

    click.echo(f"Loading preprocessed recording from {path}...")
    # Determine whether we're working with an NWB file
    nwb = bool(path.endswith(".nwb"))
    session_id, deltaf_series, timestamps = io.load_deltaf(path, nwb=nwb)

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


@process_cmd.command("regions")
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
def regions_cmd(path: str, out_dir: str) -> None:
    """Extract ∆F signal averages from ABA-defined regions."""
    if not Path(out_dir).exists():
        click.echo(f"Creating output directory {out_dir}...")
        Path(out_dir).mkdir(parents=True)

    click.echo(f"Loading preprocessed recording from {path}...")
    # Determine whether we're working with an NWB file
    nwb = bool(path.endswith(".nwb"))
    session_id, deltaf_series, timestamps = io.load_deltaf(path, nwb=nwb)

    outpath = out_dir + os.sep + session_id + "_regions.csv"

    with timer.Timer(message="Extracting region activity"):
        region_activity = pd.DataFrame(pr.extract_all_regions(deltaf_series, as_dataframe=True))
        region_activity["time_idx"] = timestamps[region_activity["time_idx"]]
        region_activity.rename(columns={"time_idx": "timestamps"}, inplace=True)
        region_activity.to_csv(outpath, index=False)

    click.echo(f"Saved region activity at {outpath}")
