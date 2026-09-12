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
import numpy as np
import pandas as pd
from pynwb.image import ImageSeries

import mesoscopy.process.region as pr
import mesoscopy.process.regression as regr
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
    help="Standard deviation of the Gaussian kernel, in pixels.",
)
def smooth_cmd(path: str, out_dir: str, sigma: int = 2) -> None:
    """Generate a smoothed DeltaF/F recording using a Laplace of Gaussian filter."""
    if not Path(out_dir).exists():
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
    help="Output directory for z-scored recording.",
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
    help="Output directory for region activity file.",
)
@click.option(
    "-m",
    "--mask",
    "mask_paths",
    type=click.Path(exists=True),
    multiple=True,
    help=(
        "Path to a custom region mask file (NPY, NPZ or TIFF), drawn in registered frame coordinates. A boolean mask"
        " is extracted as a single region, named after the file (or, for NPZ files, after each key); an"
        " integer-labelled mask is extracted as one region per label, named '<name>_<label>'. Masks are used as"
        " drawn and are not mirrored across hemispheres. May be passed multiple times to combine masks from several"
        " files. Supplying a mask replaces the ABA region extraction unless --include-aba is given."
    ),
)
@click.option(
    "--include-aba",
    is_flag=True,
    default=False,
    help="Extract ABA-defined regions alongside any custom masks. Has no effect if no mask is supplied.",
)
# A click command's docstring doubles as its --help text, so it deliberately carries no Raises: section.
def regions_cmd(path: str, out_dir: str, mask_paths: tuple[str, ...], include_aba: bool) -> None:
    """Extract ∆F signal averages from ABA-defined regions, custom region masks, or both."""  # noqa: DOC501
    if not Path(out_dir).exists():
        click.echo(f"Creating output directory {out_dir}...")
        Path(out_dir).mkdir(parents=True)

    masks: dict[str, np.ndarray] = {}
    for mask_path in mask_paths:
        click.echo(f"Loading region masks from {mask_path}...")
        try:
            file_masks = io.read_mask(mask_path)
        except ValueError as err:
            msg = f"Could not read region masks from {mask_path}: {err}"
            raise click.ClickException(msg) from err

        duplicates = [name for name in file_masks if name in masks]
        if duplicates:
            msg = (
                f"Region name(s) {', '.join(duplicates)} from {mask_path} are already defined by an earlier mask"
                " file. Rename the mask file, or its NPZ keys, so that every region name is unique."
            )
            raise click.ClickException(msg)

        masks.update(file_masks)

    if masks:
        click.echo(f"Loaded {len(masks)} region mask(s): {', '.join(masks)}")

    click.echo(f"Loading preprocessed recording from {path}...")
    # Determine whether we're working with an NWB file
    nwb = bool(path.endswith(".nwb"))
    session_id, deltaf_series, timestamps = io.load_deltaf(path, nwb=nwb)

    outpath = out_dir + os.sep + session_id + "_regions.csv"

    with timer.Timer(message="Extracting region activity"):
        activity = []
        # Custom masks stand in for the ABA regions unless the ABA regions are explicitly asked for as well.
        if include_aba or not masks:
            activity.append(pd.DataFrame(pr.extract_all_regions(deltaf_series, as_dataframe=True)))
        if masks:
            try:
                activity.append(pd.DataFrame(pr.extract_all_masks(deltaf_series, masks, as_dataframe=True)))
            except ValueError as err:
                msg = str(err)
                raise click.ClickException(msg) from err

        region_activity = pd.concat(activity, ignore_index=True)
        region_activity["time_idx"] = [
            str(timestamp, encoding="utf-8") for timestamp in timestamps[region_activity["time_idx"]]
        ]
        region_activity.rename(columns={"time_idx": "timestamp"}, inplace=True)
        region_activity.to_csv(outpath, index=False)

    click.echo(f"Saved region activity at {outpath}")


@process_cmd.command("regression")
@click.argument(
    "recording_path",
    type=click.Path(exists=True),
)
@click.argument(
    "regressor_path",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--out_dir",
    type=click.Path(dir_okay=True),
    default="./",
    help="Output directory for regression results.",
)
@click.option(
    "-a",
    "--alpha",
    type=float,
    default=1.0,
    help="Ridge regularisation strength. Defaults to 1.0.",
)
@click.option(
    "-n",
    "--nuisance-regressors",
    "nuisance_regressor_paths",
    type=click.Path(exists=True),
    multiple=True,
    help=(
        "Path to an external nuisance regressor file (NPZ or HDF5), e.g. behavioural motion energy. Every"
        " array/dataset in the file other than 'timestamps' is treated as one nuisance regressor and interpolated"
        " onto the recording's own timestamps before being z-scored and appended to the regressor matrix. May be"
        " passed multiple times to add nuisance regressors from several files."
    ),
)
@click.option(
    "-f",
    "--fast",
    is_flag=True,
    default=False,
    help=(
        "Use fast vectorised implementation of ridge regression. This is an experimental feature and may not work for"
        " all datasets. Use with caution. Defaults to False."
    ),
)
@click.option(
    "--npz",
    "file_format",
    flag_value="npz",
    default="npz",
    help="Save regression results as a compressed NumPy .npz file. Defaults to True.",
)
@click.option(
    "--h5",
    "file_format",
    flag_value="h5",
    help="Save regression results as an HDF5 file. Defaults to False.",
)
def regression_cmd(
    recording_path: str,
    regressor_path: str,
    out_dir: str,
    alpha: float,
    nuisance_regressor_paths: tuple[str, ...],
    fast: bool,
    file_format: str,
) -> None:
    """Perform pixel-wise ridge regression on a preprocessed ∆F/F recording."""
    if not Path(out_dir).exists():
        click.echo(f"Creating output directory {out_dir}...")
        Path(out_dir).mkdir(parents=True)

    click.echo(f"Loading preprocessed recording from {recording_path}...")
    # Determine whether we're working with an NWB file
    nwb = bool(recording_path.endswith(".nwb"))
    session_id, deltaf_series, timestamps = io.load_deltaf(recording_path, nwb=nwb)

    click.echo(f"Loading regressors from {regressor_path}...")
    regressors, labels, trial_idx = io.read_regressors(regressor_path)
    labels = list(labels)

    if nuisance_regressor_paths:
        # Nuisance regressors are recorded on their own clock (e.g. a behavioural camera), so both series are
        # anchored to elapsed seconds since their own first sample before interpolating one onto the other -- see
        # `regr.elapsed_seconds`.
        target_timestamps = regr.elapsed_seconds(timestamps)

    for nuisance_path in nuisance_regressor_paths:
        click.echo(f"Loading nuisance regressors from {nuisance_path}...")
        nuisance, nuisance_labels, nuisance_timestamps = io.read_nuisance_regressors(nuisance_path)
        nuisance = regr.interpolate_regressors(nuisance, regr.elapsed_seconds(nuisance_timestamps), target_timestamps)
        if trial_idx is not None:
            nuisance = nuisance[trial_idx]
        regressors, labels = regr.append_nuisance_regressors(regressors, labels, nuisance, nuisance_labels)
        click.echo(f"Added nuisance regressors: {nuisance_labels}")

    outpath = out_dir + os.sep + session_id + f"_regression.{file_format}"

    trial_idx_used = False
    if trial_idx is not None:
        deltaf_series = deltaf_series[trial_idx]
        trial_idx_used = True

    with timer.Timer(message="Running regression"):
        if fast:
            coefs, r2, mse = regr.ridge_regression_fast(deltaf_series, regressors, alpha=alpha)
        else:
            coefs, r2, mse = regr.ridge_regression(deltaf_series, regressors)

        if file_format == "npz":
            outpath = io.write_npz(
                path=outpath,
                data={
                    "coefficients": coefs,
                    "r2": r2,
                    "mse": mse,
                    "labels": labels,
                    "trial_idx": trial_idx if trial_idx_used else [],
                },
            )
        elif file_format == "h5":
            outpath = io.write_h5(
                path=outpath,
                data={
                    "/coefficients": coefs,
                    "/r2": r2,
                    "/mse": mse,
                    "/labels": np.array(labels).astype("S"),
                    "/trial_idx": trial_idx if trial_idx_used else [],
                },
            )
    click.echo(f"Saved regression results at {outpath}")
