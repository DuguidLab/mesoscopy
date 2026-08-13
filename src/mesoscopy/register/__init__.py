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
import json
import os
import pathlib

import click
import h5py
import numpy as np
from pynwb import TimeSeries
from pynwb.image import ImageSeries
from pynwb.ophys import CorrectedImageStack

import mesoscopy.preprocess as preproc
import mesoscopy.preprocess.compute as preproc_compute
import mesoscopy.register.landmarks_gui as reg_gui
import mesoscopy.register.transform as trf
import mesoscopy.resources as res
from mesoscopy import io
from mesoscopy import timer as timer


@click.group("register")
def register_cmd() -> None:
    """Register recordings to a template."""


@register_cmd.command("label")
@click.argument(
    "path",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--out_dir",
    type=click.Path(dir_okay=True),
    default="./",
    help="Output directory for registered recording.",
)
@click.option(
    "-t",
    "--template-points",
    type=click.Path(dir_okay=False),
    help="Path to template landmark points in CSV or Fiji XML points format",
)
@click.option(
    "--session-id",
    type=str,
    help="Session ID for the recording.",
)
def label_cmd(path, out_dir, template_points, session_id) -> dict:
    """Mark landmarks on a recording for registration to a template using the landmarks GUI.

    Args:
        path (str): Path to preprocessed HDF5 file or NWB file.
        out_dir (str): Output directory for registration landmarks file.
        template_points (str): Path to template landmark points in CSV or Fiji XML points format.
        session_id (str): Session ID for the recording.

    Returns:
        dict: Dictionary with the landmarks and their x-y coordinates.
              Dictionary keys are landmark names, while x-y coordinates are stored as an (x, y)
              tuple, i.e. (column, row).
    """
    click.echo("Loading imaging data...")
    nwb = bool(path.endswith(".nwb"))

    if not session_id:
        session_id = path.split("/")[-1].replace(".nwb", "") if nwb else path.split("/")[-1].replace(".h5", "")
        session_id = session_id.replace("_preprocessed", "")

    maxip = None
    isosb_maxip = None

    # Load maxip from preprocessed file.
    # if it does not exist, generate maxip from raw data.
    if path.endswith("_preprocessed.h5"):
        maxip, isosb_maxip = load_maxips(path)
    else:
        # generate maxip from raw data
        click.echo("⚠️ No maximum intensity projection found. Generating from raw data, this might take some time...")
        with timer.Timer("Generating maximum intensity projection"):
            _, raw_data, _ = preproc.load_raw(path, nwb=nwb)
            maxip = preproc_compute.projections(raw_data)["maxip"]

    click.echo("Loading template landmarks...")
    template_landmarks = res.get_default_landmarks()
    if template_points:
        template_landmarks = io.read_points(template_points)

    click.echo("Launching landmark identification GUI...")
    recording_landmarks = reg_gui.mark_landmarks(maxip, isosb_maxip, template_landmarks)

    click.echo("Saving recording landmarks...")
    outpath = out_dir + os.sep + session_id + "_landmarks.csv"
    io.write_points(outpath, recording_landmarks)

    click.echo(f"Recording landmarks saved at {outpath}.")

    return recording_landmarks


@register_cmd.command("landmarks")
@click.argument(
    "path",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--out_dir",
    type=click.Path(dir_okay=True),
    default="./",
    help="Output directory for registered recording.",
)
@click.option(
    "-r",
    "--recording-points",
    type=click.Path(dir_okay=False),
    help="Path to recording landmark points in Fiji XML points format",
)
@click.option(
    "-t",
    "--template-points",
    type=click.Path(dir_okay=False),
    help="Path to template landmark points in Fiji XML points format",
)
@click.option(
    "--output-width",
    type=int,
    default=None,
    help="Width of the registered frames. Defaults to the width of the Allen CCF template.",
)
@click.option(
    "--output-height",
    type=int,
    default=None,
    help="Height of the registered frames. Defaults to the height of the Allen CCF template.",
)
def landmarks_cmd(
    path: str,
    out_dir: str,
    recording_points: str,
    template_points: str,
    output_width: int | None = None,
    output_height: int | None = None,
) -> str:
    """Register a recording to a template based on defined landmarks.

    Args:
        path (str): Path to preprocessed recording HDF5 or NWB file.
        out_dir (str): Output directory for registered recording.
        recording_points (str, optional): Path to recording landmark points in CSV or Fiji XML points format.
        template_points (str, optional): Path to template landmark points in CSV or Fiji XML points format.
        output_width (int, optional): Width of the registered frames. Defaults to the Allen CCF template width.
        output_height (int, optional): Height of the registered frames. Defaults to the Allen CCF template height.

    Returns:
        str: Path to the registered recording file.

    Raises:
        ValueError: If the path to recording landmarks cannot be inferred.
    """
    click.echo(f"Registering recording {path} to template.")

    os.makedirs(out_dir, exist_ok=True)

    click.echo("Loading imaging data...")

    # Determine whether we're working with an NWB file
    nwb = True if path.endswith(".nwb") else False

    session_id, deltaf_series, timestamps = io.load_deltaf(path, nwb)

    click.echo("Loading landmarks...")
    template_landmarks = res.get_default_landmarks()
    if template_points:
        template_landmarks = io.read_points(template_points)

    if not recording_points:
        if nwb and pathlib.Path(path.replace(".nwb", "_landmarks.csv")).exists():
            recording_points = path.replace(".nwb", "_landmarks.csv")
        elif pathlib.Path(path.replace(".h5", "_landmarks.csv")).exists():
            recording_points = path.replace(".h5", "_landmarks.csv")
        else:
            msg = "Path to recording landmarks could not be inferred. Please supply a recording landmarks file."
            raise ValueError(msg)
    recording_landmarks = io.read_points(recording_points)

    # Registered frames land in template space, so default their shape to that of the CCF atlas.
    output_shape = None
    if output_width or output_height:
        atlas_height, atlas_width = res.get_atlas()[0].shape
        output_shape = (output_height or atlas_height, output_width or atlas_width)

    warped, tform = trf.landmarks_affine(
        deltaf_series,
        recording_landmarks,
        template_landmarks,
        output_shape=output_shape,
    )

    # Save warped frames and timestamps
    outpath = out_dir + os.sep + session_id + "_registered.h5"
    outpath = io.write_h5(
        path=outpath,
        data={
            "/F": warped,
            "/timestamps": timestamps,
            "/tform": tform.params,
            "/qa/recording_landmarks": np.array(list(recording_landmarks.values())),
            "/qa/template_landmarks": np.array(list(template_landmarks.values())),
            "/qa/registered_landmarks": tform.inverse(np.array(list(recording_landmarks.values()))),
        },
    )
    click.echo(f"Saved registered frames at {outpath}")

    if nwb:
        click.echo("Updating NWB file...")
        update_nwb(path, outpath, tform)
        click.echo(f"Updated NWB file at {path}")

    return outpath


def load_maxips(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load maximum intensity projections from a preprocessed HDF5 file.

    Args:
        path (str): Path to the preprocessed HDF5 file.

    Returns:
        tuple[np.ndarray, np.ndarray]: Maximum intensity projection for gcamp and isosb channels.
    """
    f_preproc = h5py.File(path, "r")
    gcamp_maxip_projection = np.array(f_preproc["/qa/gcamp_maxip_projection"])
    isosb_maxip_projection = np.array(f_preproc["/qa/isosb_maxip_projection"])

    return gcamp_maxip_projection, isosb_maxip_projection


def update_nwb(nwb_path: str, h5_path: str, tform_params: np.ndarray) -> None:
    """Update an NWB file with registered imaging data stored in an HDF5 file.

    Creates a link between the NWB file and the HDF5 file. See https://pynwb.readthedocs.io/en/stable/tutorials/advanced_io/linking_data.html.

    Args:
        nwb_path (str): Path to the NWB file.
        h5_path (str): Path to the HDF5 file containing the registered images.
        tform_params (np.ndarray): Affine transformation parameters.
    """
    nwbfile, nwbio = io.read_nwb(nwb_path, return_io=True)
    f = h5py.File(h5_path, "r")

    try:
        ophys_module = nwbfile.create_processing_module(name="ophys", description="optical physiology processed data")
    except ValueError:
        click.echo("Processing module already exists...")
        ophys_module = nwbfile.processing["ophys"]

    registered_series = ImageSeries(
        name="corrected",
        data=f["/F"],
        timestamps=f["/timestamps"],
        unit="df/f",
        description="dF/F widefield cortical imaging series.",
        comments="This is the haemodynamic corrected series registered to the Allen Brain Atlas CCFv3.",
    )

    xy_translation = TimeSeries(
        name="xy_translation",
        data=np.repeat(tform_params, len(f["/timestamps"]), axis=0),
        unit="pixels",
        timestamps=f["/timestamps"],
        description="Affine transformation parameters for image registration to the ABA CCFv3.",
    )

    corrected_image_stack = CorrectedImageStack(
        name="CCFRegisteredSeries",
        corrected=registered_series,
        original=nwbfile.acquisition["DualChannelImagingSeries"],
        xy_translation=xy_translation,
    )

    ophys_module.add(corrected_image_stack)

    io.write_nwb(nwb_path, nwbfile, io=nwbio)

    return nwbfile
