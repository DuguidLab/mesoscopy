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
from functools import lru_cache
from importlib import resources

import imageio
import numpy as np
import pandas as pd

import mesoscopy.resources
from mesoscopy import io

# Name of the shipped template, recorded in registered files.
TEMPLATE_NAME = "ccf_top_140x142"


def get_default_landmarks(shape: tuple[int, int] | None = None) -> dict:
    """Get default landmark coordinates.

    Args:
        shape (tuple[int, int], optional): Shape of the scaled template the landmarks should be given
            in, as (height, width). Defaults to the shipped atlas shape.

    Returns:
        dict: Default landmark coordinates, as {name: (x, y)}.
    """
    landmarks = io.read_points(str(resources.files(mesoscopy.resources).joinpath("ccf_template_landmarks_140x142.csv")))
    if shape is None:
        return landmarks
    return scale_landmarks(landmarks, template_scale(shape))


def get_atlas(shape: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Get the left and right hemisphere atlas label maps.

    Args:
        shape (tuple[int, int], optional): Shape to resample the atlas to, as (height, width). Labels
            are resampled nearest-neighbour, so no new region ids are introduced. Defaults to the
            shipped atlas shape.

    Returns:
        tuple[np.ndarray, np.ndarray]: Left and right hemisphere label maps. The right map is the
            mirror image of the left.
    """
    left_hemisphere_aba = _native_atlas()
    if shape is not None and tuple(shape) != left_hemisphere_aba.shape:
        left_hemisphere_aba = resize_labels(left_hemisphere_aba, shape)
    else:
        left_hemisphere_aba = left_hemisphere_aba.copy()
    right_hemisphere_aba = np.flip(left_hemisphere_aba, axis=1)
    return left_hemisphere_aba, right_hemisphere_aba


def get_atlas_annotations():
    return pd.read_csv(
        str(resources.files(mesoscopy.resources).joinpath("ccf_annotations.csv")), delimiter=", ", engine="python"
    )


def atlas_shape() -> tuple[int, int]:
    """Get the shape of the shipped atlas.

    Returns:
        tuple[int, int]: The atlas shape, as (height, width).
    """
    return _native_atlas().shape


def template_shape(scale: float) -> tuple[int, int]:
    """Get the shape of the atlas scaled by a uniform factor, as (height, width).

    Args:
        scale (float): Scale factor relative to the shipped atlas.

    Returns:
        tuple[int, int]: The scaled shape, rounded to whole pixels.

    Raises:
        ValueError: If the scale is not positive.
    """
    if scale <= 0:
        msg = f"Template scale must be positive, got {scale}."
        raise ValueError(msg)
    height, width = atlas_shape()
    return max(1, round(height * scale)), max(1, round(width * scale))


def template_scale(shape: tuple[int, int]) -> tuple[float, float]:
    """Get the per-axis scale of a template shape relative to the shipped atlas.

    Args:
        shape (tuple[int, int]): Template shape, as (height, width).

    Returns:
        tuple[float, float]: Scale factors as (x, y), i.e. (width, height), matching the landmark
            coordinate convention.
    """
    height, width = atlas_shape()
    return shape[1] / width, shape[0] / height


def scale_landmarks(landmarks: dict, scale: tuple[float, float]) -> dict:
    """Scale landmark coordinates to a resampled image.

    Pixel centres sit at integer coordinates, so a coordinate is shifted by half a pixel before and
    after scaling. This keeps a landmark over the same pixel as ``resize_labels`` and the image
    midline in place at any scale.

    Args:
        landmarks (dict): Landmarks as {name: (x, y)}.
        scale (tuple[float, float]): Scale factors as (x, y).

    Returns:
        dict: The scaled landmarks, in the same order.
    """
    scale_x, scale_y = scale
    return {name: ((x + 0.5) * scale_x - 0.5, (y + 0.5) * scale_y - 0.5) for name, (x, y) in landmarks.items()}


def resize_labels(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resample a label map to a new shape by nearest neighbour.

    Each output pixel takes the label under its centre in the input, so no new labels are created.

    Args:
        labels (np.ndarray): 2D label map.
        shape (tuple[int, int]): Output shape, as (height, width).

    Returns:
        np.ndarray: The resampled label map, with the same dtype as the input.

    Raises:
        ValueError: If the shape is not two positive integers.
    """
    if len(shape) != labels.ndim or any(int(n) != n or n < 1 for n in shape):
        msg = f"Shape must be two positive integers, got {shape}."
        raise ValueError(msg)
    height, width = labels.shape
    rows = np.floor((np.arange(shape[0]) + 0.5) * height / shape[0]).astype(int)
    cols = np.floor((np.arange(shape[1]) + 0.5) * width / shape[1]).astype(int)
    return labels[np.ix_(rows, cols)]


@lru_cache(maxsize=1)
def _native_atlas() -> np.ndarray:
    atlas = imageio.imread(str(resources.files(mesoscopy.resources).joinpath("ccf_template_top_140x142.tiff")))
    atlas.setflags(write=False)
    return atlas
