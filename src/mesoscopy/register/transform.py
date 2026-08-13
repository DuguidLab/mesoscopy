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
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import click
import numpy as np
from skimage import transform as trf

import mesoscopy.resources as res


def landmarks_affine(
    deltaf_series: np.ndarray,
    recording_landmarks: dict,
    template_landmarks: dict,
    output_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, trf.ProjectiveTransform]:
    """Warp a DeltaF/F series to match a template using anatomical landmarks.

    Both landmark sets must use the same (x, y) — i.e. (column, row) — coordinate convention as
    ``skimage.transform``.

    The registered frames are in template space, so their shape is that of the template rather than
    that of the recording. It defaults to the shape of the Allen CCF atlas, but can be overridden with ``output_shape``.

    Args:
        deltaf_series (np.ndarray): DeltaF/F series.
        recording_landmarks (dict): Recording landmarks, as {name: (x, y)}.
        template_landmarks (dict): Template landmarks, as {name: (x, y)}.
        output_shape (tuple[int, int], optional): Shape of the registered frames, as (height, width).
            Defaults to the shape of the Allen CCF atlas template.

    Returns:
        tuple[np.ndarray, trf.ProjectiveTransform]: Registered DeltaF/F series and affine
            transformation matrix.
    """
    if output_shape is None:
        output_shape = res.get_atlas()[0].shape
    if not isinstance(deltaf_series, np.ndarray):
        click.echo("Loading imaging data into memory...")
        deltaf_series = np.asarray(deltaf_series)

    template = np.array(list(template_landmarks.values()), dtype=np.float32)
    recording = np.array(list(recording_landmarks.values()), dtype=np.float32)

    click.echo("Estimating transform...")
    start = time.time()
    tform = trf.estimate_transform("affine", template, recording)
    end = time.time()
    click.echo(f"Transform estimated in {end - start} s")

    n_frames = deltaf_series.shape[0]

    def _warp_frame(idx: int) -> np.ndarray:
        return trf.warp(deltaf_series[idx], tform, order=3, output_shape=output_shape)

    start = time.time()
    results: list[np.ndarray | None] = [None] * n_frames
    n_workers = os.cpu_count() or 1
    with (
        click.progressbar(
            length=n_frames,
            label=f"Registering recording to template ({output_shape[1]}x{output_shape[0]} frames)...",
        ) as bar,
        ThreadPoolExecutor(max_workers=n_workers) as executor,
    ):
        futures = {executor.submit(_warp_frame, i): i for i in range(n_frames)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            bar.update(1)
    end = time.time()
    click.echo(f"Recording registered in {end - start} s")

    return np.stack(results), tform
