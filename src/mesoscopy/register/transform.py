#  Copyright (c) 2024 Constantinos Eleftheriou <Constantinos.Eleftheriou@ed.ac.uk>.
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
import time
import numpy as np

from skimage import transform as trf

import mesoscopy.plots as plots


def landmarks_affine(
    deltaf_series: np.ndarray,
    recording_landmarks: dict,
    template_landmarks: dict,
    crop_x: int = 0,
    crop_y: int = 0,
    qa_dir: str = "",
    session_id: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a DeltaF/F series to match a template using anatomical landmarks.

    Args:
        deltaf_series (np.ndarray): DeltaF/F series.
        recording_landmarks (dict): Recording landmarks.
        template_landmarks (dict): Template landmarks.
        crop_x (int, optional): Crop x-axis. Defaults to 0.
        crop_y (int, optional): Crop y-axis. Defaults to 0.
        qa_dir (str, optional): Directory to save QA plots. Defaults to "".
        session_id (str, optional): Session ID. Defaults to "".

    Returns:
        tuple[np.ndarray, np.ndarray]: Registered DeltaF/F series and affine transformation matrix.
    """
    template = np.array(list(template_landmarks.values()), dtype=np.float32)
    recording = np.array(list(recording_landmarks.values()), dtype=np.float32)

    click.echo("Estimating transform...")
    start = time.time()
    tform = trf.estimate_transform("affine", template, recording)
    end = time.time()
    click.echo("Transform estimated in {} s".format(end - start))

    if qa_dir:
        plots.plot_scatters(
            xs=[template[:, 1], recording[:, 1]],
            ys=[template[:, 0], recording[:, 0]],
            outpath=qa_dir
            + os.sep
            + session_id
            + "_qa_registration_unregistered-landmarks.png",
            labels=["template", "recording"],
            message="Saved scatter of unregistered landmarks.",
        )

        plots.plot_scatters(
            xs=[template[:, 1], tform.inverse(recording)[:, 1]],
            ys=[template[:, 0], tform.inverse(recording)[:, 0]],
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

    return warped, tform.params[None, :]
