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
import os
import click
import h5py
import numpy as np

import time

import skimage.io as skio

import mesoscopy.io as io
import mesoscopy.plots as plots
import mesoscopy.register.landmarks_gui as reg_gui
import mesoscopy.register.transform as trf
import mesoscopy.resources as res

from pynwb import TimeSeries
from pynwb.image import ImageSeries
from pynwb.ophys import CorrectedImageStack


@click.group("register")
def register_cmd() -> None:
    """Register recordings to a template."""
    pass


@register_cmd.command("mark-landmarks")
@click.argument(
    "maxip_path",
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
def mark_landmarks(maxip_path, out_dir, template_points, session_id) -> dict:
    """Mark landmarks on a recording for registration to a template using the landmarks GUI.

    Args:
        maxip_path (str): Path to maximum intensity projection image.
        out_dir (str): Output directory for registered recording.
        template_points (str): Path to template landmark points in CSV or Fiji XML points format.
        session_id (str): Session ID for the recording.
    """

    click.echo("Loading imaging data...")
    maxip = skio.imread(maxip_path)

    click.echo("Loading template landmarks...")
    template_landmarks = res.get_default_landmarks()
    if template_points:
        template_landmarks = io.read_points(template_points)

    click.echo("Launching landmark identification GUI...")
    recording_landmarks = reg_gui.mark_landmarks(maxip, template_landmarks)

    if not session_id:
        session_id = os.path.basename(maxip_path).split(".")[0].split("_qa")[0]

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
@click.option("--crop-x", default=0, help="Crop recording along the x-axis.")
@click.option("--crop-y", default=0, help="Crop recording along the y-axis.")
def register_landmarks(
    path: str,
    out_dir: str,
    recording_points: str,
    template_points: str,
    crop_x: int = 0,
    crop_y: int = 0,
) -> None:
    """Register a recording to a template based on defined landmarks.

    Args:
        path (str): Path to preprocessed recording HDF5 or NWB file.
        out_dir (str): Output directory for registered recording.
        recording_points (str): Path to recording landmark points in CSV or Fiji XML points format.
        template_points (str): Path to template landmark points in CSV or Fiji XML points format.
        crop_x (int, optional): Number of pixels to crop from the x-axis of the recording. Defaults to 0.
        crop_y (int, optional): Number of pixels to crop from the y-axis of the recording. Defaults to 0.
    """
    click.echo("Registering recording {} to template.".format(path))

    registration_start = time.time()

    os.makedirs(out_dir, exist_ok=True)

    qa_dir = out_dir + os.sep + "qa"
    os.makedirs(qa_dir, exist_ok=True)

    click.echo("Loading imaging data...")

    # Determine whether we're working with an NWB file
    nwb = True if path.endswith(".nwb") else False

    session_id, deltaf_series, timestamps = _load_preprocessed(path, nwb)

    click.echo("Loading landmarks...")
    template_landmarks = res.get_default_landmarks()
    if template_points:
        template_landmarks = io.read_points(template_points)

    if not recording_points:
        if nwb and os.path.exists(path.replace(".nwb", "_landmarks.csv")):
            recording_points = path.replace(".nwb", "_landmarks.csv")
        elif os.path.exists(path.replace(".h5", "_landmarks.csv")):
            recording_points = path.replace(".h5", "_landmarks.csv")
        else:
            raise ValueError(
                "Path to recording landmarks could not be inferred. Please supply a recording landmarks file."
            )
    recording_landmarks = io.read_points(recording_points)

    warped, tform = trf.landmarks_affine(
        deltaf_series,
        recording_landmarks,
        template_landmarks,
        crop_x=crop_x,
        crop_y=crop_y,
        qa_dir=qa_dir,
        session_id=session_id,
    )

    plots.plot_frame(
        warped[100],
        outpath=qa_dir + os.sep + session_id + "_qa_registration_registered-frame.png",
        message="Saved frame of registered session.",
    )

    # Save warped frames and timestamps
    h5_path = out_dir + os.sep + session_id + "-registered.h5"
    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("F", data=warped)
        hf.create_dataset("ts", data=timestamps)
    click.echo("Saved registered frames at {}".format(h5_path))

    registration_end = time.time()
    click.echo(
        "Registration took a total of {} mins.".format(
            (registration_end - registration_start) / 60
        )
    )

    if nwb:
        click.echo("Updating NWB file...")
        _update_nwb(path, h5_path, tform)
        click.echo("Updated NWB file at {}".format(path))


def _load_preprocessed(
    path: str, nwb: bool = False
) -> tuple[str, np.ndarray, np.ndarray]:
    """Load preprocessed data from an HDF5 or NWB file.

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
        f_preproc = h5py.File(path)
        deltaf_series = f_preproc["/frames"]
        timestamps = f_preproc["/timestamps"]

    return session_id, deltaf_series, timestamps


def _update_nwb(nwb_path: str, h5_path: str, tform_params: np.ndarray) -> None:
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
        ophys_module = nwbfile.create_processing_module(
            name="ophys", description="optical physiology processed data"
        )
    except ValueError:
        click.echo("Processing module already exists...")
        ophys_module = nwbfile.processing["ophys"]

    registered_series = ImageSeries(
        name="corrected",
        data=f["/F"],
        timestamps=f["/ts"],
        unit="df/f",
        description="dF/F widefield cortical imaging series.",
        comments="This is the haemodynamic corrected series registered to the Allen Brain Atlas CCFv3.",
    )

    xy_translation = TimeSeries(
        name="xy_translation",
        data=np.repeat(tform_params, len(f["/ts"]), axis=0),
        unit="pixels",
        timestamps=f["/ts"],
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
