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
import xmltodict
import numpy as np
from collections import OrderedDict
import h5py

import time

from skimage import transform as trf

import mesoscopy.io as io
import mesoscopy.plots as plots

from pynwb import TimeSeries
from pynwb.image import ImageSeries
from pynwb.ophys import CorrectedImageStack


@click.command()
@click.argument(
    "path",
    type=click.Path(exists=True),
    help="Path to preprocessed recording HDF5 or NWB file.",
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
def register(
    path: str,
    out_dir: str,
    recording_points: str,
    template_points: str,
    crop_x: int = 0,
    crop_y: int = 0,
) -> None:
    """Register a recording to a template.

    Args:
        path (str): Path to preprocessed recording HDF5 or NWB file.
        out_dir (str): Output directory for registered recording.
        recording_points (str): Path to recording landmark points in Fiji XML points format.
        template_points (str): Path to template landmark points in Fiji XML points format.
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
    template_landmarks = _get_landmarks(template_points)
    recording_landmarks = _get_landmarks(recording_points)

    plots.plot_scatters(
        xs=[template_landmarks[:, 0], recording_landmarks[:, 0]],
        ys=[template_landmarks[:, 1], recording_landmarks[:, 1]],
        outpath=qa_dir
        + os.sep
        + session_id
        + "_qa_registration_unregistered-landmarks.png",
        labels=["template", "recording"],
        message="Saved scatter of unregistered landmarks.",
    )

    click.echo("Estimating transform...")
    start = time.time()
    tform = trf.estimate_transform("affine", template_landmarks, recording_landmarks)
    end = time.time()
    click.echo("Transform estimated in {} s".format(end - start))

    plots.plot_scatters(
        xs=[template_landmarks[:, 0], tform.inverse(recording_landmarks)[:, 0]],
        ys=[template_landmarks[:, 1], tform.inverse(recording_landmarks)[:, 1]],
        outpath=qa_dir
        + os.sep
        + session_id
        + "_qa_registration_registered-landmarks.png",
        labels=["template", "registered"],
        message="Saved scatter of registered landmarks.",
    )

    start = time.time()
    _warped = []
    with click.progressbar(
        range(deltaf_series.shape[0]), label="Registering recording to template..."
    ) as frame_ids:
        for idx in frame_ids:
            if crop_x > 0 or crop_y > 0:
                _warped.append(
                    trf.warp(deltaf_series[idx, :crop_y, :crop_x], tform, order=3)
                )
            else:
                _warped.append(trf.warp(deltaf_series[idx], tform, order=3))
    warped = np.array(_warped)
    end = time.time()
    click.echo("Session registered in {} s".format(end - start))

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
        _update_nwb(path, h5_path, tform.params[None, :])
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


def _get_landmarks(points_path: str) -> np.ndarray:
    """Load landmark points from a Fiji XML points file.

    Args:
        points_path (str): Path to the points file.

    Returns:
        np.ndarray: Array of landmark points.
    """
    with open(points_path, "r") as fp:
        pts = xmltodict.parse(fp.read())
        pts = OrderedDict(
            {
                point["@name"]: (point["@x"], point["@y"])
                for point in pts["namedpointset"]["pointworld"]
            }
        )

    return np.array(list(pts.values()), dtype=np.float32)


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
