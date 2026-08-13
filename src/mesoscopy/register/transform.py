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

# An affine fit needs at least three non-collinear point pairs.
MIN_LANDMARKS = 3

# Warn when the fit leaves landmarks further than this fraction of the template's extent from their
# target positions. 0.05 is roughly 6 px on the Allen CCF template.
RESIDUAL_WARN_FRACTION = 0.05


def align_landmarks(recording_landmarks: dict, template_landmarks: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Pair two landmark sets by name.

    The transform is fitted from two arrays of points, so the pairing between them is positional.
    Building those arrays from ``dict.values()`` silently mispairs the points whenever the two sets
    are ordered differently or one of them is missing a landmark, so they are matched by name here.

    Args:
        recording_landmarks (dict): Recording landmarks, as {name: (x, y)}.
        template_landmarks (dict): Template landmarks, as {name: (x, y)}.

    Returns:
        tuple[list[str], np.ndarray, np.ndarray]: The shared landmark names in template order, and
            the matching template and recording points as (n, 2) arrays of (x, y) coordinates.

    Raises:
        ValueError: If the two sets share fewer than MIN_LANDMARKS landmarks.
    """
    shared = [name for name in template_landmarks if name in recording_landmarks]

    unmarked = [name for name in template_landmarks if name not in recording_landmarks]
    if unmarked:
        click.echo(f"⚠️ Template landmarks with no matching recording landmark: {', '.join(unmarked)}")

    unknown = [name for name in recording_landmarks if name not in template_landmarks]
    if unknown:
        click.echo(f"⚠️ Recording landmarks with no matching template landmark: {', '.join(unknown)}")

    if len(shared) < MIN_LANDMARKS:
        msg = (
            f"Only {len(shared)} landmark(s) are common to the recording and the template, "
            f"at least {MIN_LANDMARKS} are needed to fit an affine transform."
        )
        raise ValueError(msg)

    template = np.array([template_landmarks[name] for name in shared], dtype=np.float64)
    recording = np.array([recording_landmarks[name] for name in shared], dtype=np.float64)

    return shared, template, recording


def _is_collinear(points: np.ndarray) -> bool:
    """Check whether a set of points lies on a single line (or is a single repeated point).

    Args:
        points (np.ndarray): Points as an (n, 2) array.

    Returns:
        bool: True if the points span fewer than two dimensions.
    """
    centred = points - points.mean(axis=0)
    singular_values = np.linalg.svd(centred, compute_uv=False)
    return bool(singular_values[1] <= 1e-8 * singular_values[0])


def landmark_residuals(
    tform: trf.ProjectiveTransform,
    template_points: np.ndarray,
    recording_points: np.ndarray,
) -> np.ndarray:
    """Measure how far each marked landmark lands from its template position once registered.

    Residuals are reported in template pixels, i.e. in the space of the registered frames, so they
    are comparable across recordings of different sizes.

    Note that this does not detect a transposed coordinate convention: an affine least-squares fit
    solves each output axis independently, so swapping the axes of one point set permutes the rows
    of the fitted matrix and leaves the residuals unchanged.

    Args:
        tform (trf.ProjectiveTransform): The fitted transform, mapping template to recording space.
        template_points (np.ndarray): Template points as an (n, 2) array of (x, y) coordinates.
        recording_points (np.ndarray): Matching recording points, in the same order.

    Returns:
        np.ndarray: Residual distance per landmark, in template pixels.
    """
    return np.linalg.norm(tform.inverse(recording_points) - template_points, axis=1)


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

    Raises:
        ValueError: If the landmarks do not define a usable affine transform.
    """
    if output_shape is None:
        output_shape = res.get_atlas()[0].shape
    if not isinstance(deltaf_series, np.ndarray):
        click.echo("Loading imaging data into memory...")
        deltaf_series = np.asarray(deltaf_series)

    names, template, recording = align_landmarks(recording_landmarks, template_landmarks)

    click.echo(f"Estimating transform from {len(names)} landmarks...")
    start = time.time()
    tform = trf.estimate_transform("affine", template, recording)
    end = time.time()

    # skimage returns a least-squares solution without complaint for degenerate point sets, so the
    # points have to be checked rather than the fit: collinear landmarks leave the fit
    # underdetermined but still produce a plausible-looking, non-singular matrix.
    for name, points in (("template", template), ("recording", recording)):
        if _is_collinear(points):
            msg = (
                f"The {name} landmarks are collinear or coincident, so they do not define an affine "
                "transform. Check the marked points."
            )
            raise ValueError(msg)

    if not np.isfinite(tform.params).all():
        msg = "Could not estimate a transform from these landmarks, the fit did not converge."
        raise ValueError(msg)

    residuals = landmark_residuals(tform, template, recording)
    worst = int(np.argmax(residuals))
    rmse = float(np.sqrt((residuals**2).mean()))
    click.echo(
        f"Landmark fit: RMSE {rmse:.2f} px, worst is '{names[worst]}' at {residuals[worst]:.2f} px "
        "(in template pixels)."
    )

    template_extent = float(np.linalg.norm(template.max(axis=0) - template.min(axis=0)))
    if rmse > RESIDUAL_WARN_FRACTION * template_extent:
        click.echo(
            f"⚠️ Landmark fit RMSE is over {RESIDUAL_WARN_FRACTION:.0%} of the template extent - "
            "the registration may be poor. Check the landmarks and the registration QA report."
        )
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
