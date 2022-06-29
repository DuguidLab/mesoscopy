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
"""Points registration CLI."""
import os
import click
import xmltodict
import numpy as np
from collections import OrderedDict
import h5py

import time

import matplotlib.pyplot as plt
from skimage import transform as trf


@click.command()
@click.argument("recording_path", type=click.Path(exists=True))
@click.option("-o", "--out_dir", type=click.Path(dir_okay=True), default="./")
@click.option("-r", "--recording-points", type=click.Path(dir_okay=False))
@click.option("-t", "--template-points", type=click.Path(dir_okay=False))
@click.option("--crop-x", default=0, help="Crop recording along the x-axis.")
@click.option("--crop-y", default=0, help="Crop recording along the y-axis.")
def landmarks(
    recording_path,
    out_dir,
    recording_points,
    template_points,
    crop_x=0,
    crop_y=0,
):
    """Register a recording to a template based on manually predefined landmarks."""
    click.echo("Registering recording {} to template.".format(recording_path))

    registration_start = time.time()

    session_id = recording_path.split("/")[-1].replace(".h5", "")
    os.makedirs(out_dir, exist_ok=True)

    qa_dir = out_dir + os.sep + "qa"
    os.makedirs(qa_dir, exist_ok=True)

    click.echo("Loading imaging data...")
    f = h5py.File(recording_path)
    frames = f["/F"]

    click.echo("Loading landmarks...")
    template_landmarks = get_landmarks(template_points)
    recording_landmarks = get_landmarks(recording_points)

    plt.clf()
    plt.scatter(template_landmarks[:, 0], template_landmarks[:, 1], color="darkorange")
    plt.scatter(recording_landmarks[:, 0], recording_landmarks[:, 1], color="purple")
    plt.xlim(0, frames.shape[2])
    plt.ylim(frames.shape[1], 0)
    plt.legend(["template", "recording"])
    outpath = (
        qa_dir + os.sep + session_id + "_qa_registration_unregistered-landmarks.png"
    )
    plt.savefig(outpath)
    click.echo("Saved scatter of unregistered landmarks at {}".format(outpath))

    plt.clf()
    plt.imshow(frames[100])
    plt.scatter(
        recording_landmarks[:, 0],
        recording_landmarks[:, 1],
        color="purple",
    )
    outpath = qa_dir + os.sep + session_id + "_qa_registration_unregistered-frame.png"
    plt.savefig(outpath)
    click.echo("Saved frame overlay of unregistered landmarks at {}".format(outpath))

    click.echo("Estimating transform...")
    start = time.time()
    tform = trf.estimate_transform("affine", template_landmarks, recording_landmarks)
    end = time.time()
    click.echo("Transform estimated in {} s".format(end - start))

    plt.clf()
    plt.scatter(template_landmarks[:, 0], template_landmarks[:, 1], color="darkorange")
    plt.scatter(
        tform.inverse(recording_landmarks)[:, 0],
        tform.inverse(recording_landmarks)[:, 1],
        color="green",
    )
    plt.xlim(0, frames.shape[2])
    plt.ylim(frames.shape[1], 0)
    plt.legend(["template", "registered"])
    outpath = qa_dir + os.sep + session_id + "_qa_registration_registered-landmarks.png"
    plt.savefig(outpath)
    click.echo("Saved scatter of registered landmarks at {}".format(outpath))

    start = time.time()
    warped = []
    with click.progressbar(
        range(frames.shape[0]), label="Registering recording to template..."
    ) as frame_ids:
        for idx in frame_ids:
            if crop_x > 0 or crop_y > 0:
                warped.append(trf.warp(frames[idx, :crop_y, :crop_x], tform, order=3))
            else:
                warped.append(trf.warp(frames[idx], tform, order=3))
    warped = np.array(warped)
    end = time.time()
    click.echo("Session registered in {} s".format(end - start))

    plt.clf()
    plt.imshow(warped[100])
    plt.scatter(template_landmarks[:, 0], template_landmarks[:, 1], color="darkorange")
    plt.scatter(
        tform.inverse(recording_landmarks)[:, 0],
        tform.inverse(recording_landmarks)[:, 1],
        color="green",
    )
    plt.xlim(0, warped.shape[2])
    plt.ylim(warped.shape[1], 0)
    plt.legend(["template", "registered"])
    outpath = qa_dir + os.sep + session_id + "_qa_registration_registered-frame.png"
    plt.savefig(outpath)
    click.echo("Saved frame overlay of registered landmarks at {}".format(outpath))

    # Save warped frames and timestamps
    outpath = out_dir + os.sep + session_id + "-registered.h5"
    with h5py.File(outpath, "w") as hf:
        hf.create_dataset("F", data=warped)
        hf.create_dataset("ts", data=f["ts"][:])
    click.echo("Saved registered frames at {}".format(outpath))

    registration_end = time.time()
    click.echo(
        "Registration took a total of {} mins.".format(
            (registration_end - registration_start) / 60
        )
    )


def get_landmarks(points_path):
    with open(points_path, "r") as fp:
        pts = xmltodict.parse(fp.read())
        pts = OrderedDict(
            {
                point["@name"]: (point["@x"], point["@y"])
                for point in pts["namedpointset"]["pointworld"]
            }
        )

    return np.array(list(pts.values()), dtype=np.float32)
