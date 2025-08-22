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
import time

import click
import numpy as np
from skimage import transform as trf


def landmarks_affine(
    deltaf_series: np.ndarray,
    recording_landmarks: dict,
    template_landmarks: dict,
    crop_x: int = 0,
    crop_y: int = 0,
) -> tuple[np.ndarray, trf.ProjectiveTransform]:
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
    click.echo(f"Transform estimated in {end - start} s")

    warped_ = []
    with click.progressbar(range(deltaf_series.shape[0]), label="Registering recording to template...") as frame_ids:
        for idx in frame_ids:
            if crop_x > 0 or crop_y > 0:
                warped_.append(trf.warp(deltaf_series[idx, :crop_y, :crop_x], tform.inverse, order=3))
            else:
                warped_.append(trf.warp(deltaf_series[idx, :, :], tform.inverse, order=3))
    warped = np.array(warped_)

    return warped, tform
