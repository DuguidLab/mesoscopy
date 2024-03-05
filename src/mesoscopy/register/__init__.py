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

import matplotlib.pyplot as plt
from skimage import transform as trf

import mesoscopy.io as io
import mesoscopy.plots as plots

from pynwb import NWBHDF5IO, TimeSeries
from pynwb.image import ImageSeries
from pynwb.ophys import CorrectedImageStack


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("-o", "--out_dir", type=click.Path(dir_okay=True), default="./")
@click.option("-r", "--recording-points", type=click.Path(dir_okay=False))
@click.option("-t", "--template-points", type=click.Path(dir_okay=False))
@click.option("--crop-x", default=0, help="Crop recording along the x-axis.")
@click.option("--crop-y", default=0, help="Crop recording along the y-axis.")
def register(
    path,
    out_dir,
    recording_points,
    template_points,
    crop_x=0,
    crop_y=0,
):
    """Register a recording to a template."""
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
    warped = []
    with click.progressbar(
        range(deltaf_series.shape[0]), label="Registering recording to template..."
    ) as frame_ids:
        for idx in frame_ids:
            if crop_x > 0 or crop_y > 0:
                warped.append(
                    trf.warp(deltaf_series[idx, :crop_y, :crop_x], tform, order=3)
                )
            else:
                warped.append(trf.warp(deltaf_series[idx], tform, order=3))
    warped = np.array(warped)
    end = time.time()
    click.echo("Session registered in {} s".format(end - start))

    plots.plot_frame(
        warped[100],
        outpath=qa_dir + os.sep + session_id + "_qa_registration_registered-frame.png",
        message="Saved frame of registered session.",
    )

    # Save warped frames and timestamps
    outpath = out_dir + os.sep + session_id + "-registered.h5"
    with h5py.File(outpath, "w") as hf:
        hf.create_dataset("F", data=warped)
        hf.create_dataset("ts", data=timestamps)
    click.echo("Saved registered frames at {}".format(outpath))

    registration_end = time.time()
    click.echo(
        "Registration took a total of {} mins.".format(
            (registration_end - registration_start) / 60
        )
    )

    if nwb:
        click.echo("Updating NWB file...")
        f = h5py.File(outpath, "r")

        try:
            ophys_module = nwbfile.create_processing_module(
                name="ophys", description="optical physiology processed data"
            )
        except ValueError:
            print("Processing module already exists...")
            ophys_module = nwbfile.processing["ophys"]

        registered_series = ImageSeries(
            name="corrected",
            data=f["/F"],
            timestamps=timestamps,
            unit="df/f",
            description="dF/F widefield cortical imaging series.",
            comments="This is the haemodynamic corrected series registered to the Allen Brain Atlas CCFv3.",
        )

        xy_translation = TimeSeries(
            name="xy_translation",
            data=np.repeat(tform.params[None, :], len(timestamps), axis=0),
            unit="pixels",
            timestamps=timestamps,
            description="Affine transformation parameters for image registration to the ABA CCFv3.",
        )

        corrected_image_stack = CorrectedImageStack(
            name="CCFRegisteredSeries",
            corrected=registered_series,
            original=nwbfile.acquisition["DualChannelImagingSeries"],
            xy_translation=xy_translation,
        )

        ophys_module.add(corrected_image_stack)

        print("Writing to file...")
        io.write(nwbfile)
        io.close()

    print("Cleaning up...")
    f.close()


def _load_preprocessed(path, nwb=False):
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


def _get_landmarks(points_path):
    with open(points_path, "r") as fp:
        pts = xmltodict.parse(fp.read())
        pts = OrderedDict(
            {
                point["@name"]: (point["@x"], point["@y"])
                for point in pts["namedpointset"]["pointworld"]
            }
        )

    return np.array(list(pts.values()), dtype=np.float32)
