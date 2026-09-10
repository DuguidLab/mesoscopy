#  Copyright (c) 2026 Constantinos Eleftheriou <Constantinos.Eleftheriou@ed.ac.uk>.
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

from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from mesoscopy import resources

DEFAULT_EXCLUDE = [
    "FRP1",
    "VISpl1",
    "VISpor1",
    "VISli1",
    "TEa1",
    "AUDd1",
    "AUDp1",
    "AUDpo1",
    "AUDv1",
    "ORBm1",
]

# extract_all_regions works through frames in blocks, so its NaN-aware temporaries stay bounded on long
# recordings.
_BLOCK_FRAMES = 1024


def extract_region_activity(
    deltaf_series: npt.NDArray, region_acronym: str, hemisphere: Literal["left", "right", "both"]
) -> npt.NDArray:
    """Extracts the mean ∆F/F signal of a specific cortical region in the Allen Brain Atlas from a registered recording.

    Args:
        deltaf_series (npt.NDArray): A 3D array of shape (time, height, width) representing the DeltaF/F signal.
            Should be registered to the Allen Brain Atlas.
        region_acronym (str): The acronym of the cortical region to extract (e.g., 'VISp', 'MOp etc.).
        hemisphere (Literal["left", "right", "both"]): The hemisphere to extract the activity from.
            Options are 'left', 'right', or 'both'.

    Returns:
        npt.NDArray: A 1D array of shape (time,) representing the mean ∆F/F signal of the specified cortical
            region over time. NaN pixels are ignored, so a frame is only NaN if every pixel in the region is
            NaN in that frame.

    Raises:
        ValueError: If the specified region acronym is not recognized or if the hemisphere option is invalid.
    """
    annotations = resources.get_atlas_annotations()
    region_id = annotations.loc[annotations["acronym"] == region_acronym, "id"].values[0]

    if not region_id:
        msg = f"Region acronym {region_acronym} not recognised."
        raise ValueError(msg)

    left_aba, right_aba = resources.get_atlas()

    if hemisphere.lower() == "left":
        region_mask = left_aba == region_id
    elif hemisphere.lower() == "right":
        region_mask = right_aba == region_id
    elif hemisphere.lower() == "both":
        region_mask = (left_aba + right_aba) == region_id
    else:
        msg = f"Could not recognise hemisphere option {hemisphere}, select left, right or both."
        raise ValueError(msg)

    return np.nanmean(deltaf_series[:, region_mask], axis=1)


def extract_all_regions(
    deltaf_series: npt.NDArray,
    exclude: list | None = None,
    ignore_default_exclude: bool = False,
    as_dataframe: bool = False,
) -> dict | pd.DataFrame:
    """Extracts the mean ∆F/F signal of all cortical regions in the Allen Brain Atlas from a registered recording.

    Args:
        deltaf_series (npt.NDArray): A 3D array of shape (time, height, width) representing the DeltaF/F signal.
        exclude (list, optional): A list of region acronyms to exclude from the extraction. Defaults to None.
        ignore_default_exclude (bool, optional): If True, ignores the default excluded regions. Defaults to False.
        as_dataframe (bool, optional): If True, returns the result as a pandas DataFrame. Defaults to False.

    Returns:
        dict | pd.DataFrame: A dictionary with region acronyms as keys and their mean activity as values,
            or a DataFrame with columns 'region', 'time_idx', and 'F'. NaN pixels are ignored, so a region is
            only NaN in a frame if every one of its pixels is NaN there.
    """
    annotations = resources.get_atlas_annotations()
    left_aba, right_aba = resources.get_atlas()

    if exclude is None:
        exclude = []

    excluded_regions = list(DEFAULT_EXCLUDE) if not ignore_default_exclude else []
    excluded_regions.extend(exclude)
    regions = [region for region in annotations.acronym.unique() if region not in excluded_regions]

    region_ids = {region: annotations.loc[annotations["acronym"] == region, "id"].values[0] for region in regions}

    hw = left_aba.shape[0] * left_aba.shape[1]
    left_masks = np.array([(left_aba == region_ids[r]).ravel() for r in regions])  # (n_regions, H*W)
    right_masks = np.array([(right_aba == region_ids[r]).ravel() for r in regions])

    data_flat = deltaf_series.reshape(deltaf_series.shape[0], hw)  # (T, H*W)
    left_activities = _region_means(data_flat, left_masks)  # (T, n_regions)
    right_activities = _region_means(data_flat, right_masks)

    region_activity = {}
    for i, region in enumerate(regions):
        region_activity[f"L_{region}"] = left_activities[:, i]
        region_activity[f"R_{region}"] = right_activities[:, i]

    if as_dataframe:
        df = pd.DataFrame(region_activity)
        return df.unstack().reset_index().rename(columns={"level_0": "region", "level_1": "time_idx", 0: "F"})

    return region_activity


def extract_mask_activity(deltaf_series: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    """Extracts the mean ∆F/F signal of a custom region mask from a registered recording.

    The mask is applied to the recording as drawn, in registered frame coordinates.

    A `ValueError` is raised if the mask does not match the frame shape of the recording, or if it does
    not select any pixels.

    Args:
        deltaf_series (npt.NDArray): A 3D array of shape (time, height, width) representing the DeltaF/F signal.
            Should be registered to the Allen Brain Atlas.
        mask (npt.NDArray): A 2D array of shape (height, width) selecting the pixels to average over. Non-boolean
            masks are treated as boolean, with every non-zero pixel counting as part of the region.

    Returns:
        npt.NDArray: A 1D array of shape (time,) representing the mean ∆F/F signal of the masked region over time.
            NaN pixels are ignored, so a frame is only NaN if every masked pixel is NaN in that frame.
    """
    mask = _validate_mask(mask, deltaf_series.shape[1:])
    return np.nanmean(deltaf_series[:, mask], axis=1)


def extract_all_masks(
    deltaf_series: npt.NDArray,
    masks: dict[str, npt.NDArray],
    as_dataframe: bool = False,
) -> dict | pd.DataFrame:
    """Extracts the mean ∆F/F signal of each of a set of custom region masks from a registered recording.

    A `ValueError` is raised if a mask does not match the frame shape of the recording, or if it does not select
    any pixels.

    Args:
        deltaf_series (npt.NDArray): A 3D array of shape (time, height, width) representing the DeltaF/F signal.
        masks (dict[str, npt.NDArray]): A dictionary with region names as keys and their 2D masks as values, as
            returned by `mesoscopy.io.read_mask`.
        as_dataframe (bool, optional): If True, returns the result as a pandas DataFrame. Defaults to False.

    Returns:
        dict | pd.DataFrame: A dictionary with region names as keys and their mean activity as values,
            or a DataFrame with columns 'region', 'time_idx', and 'F'.
    """
    mask_activity = {}
    for name, mask in masks.items():
        # Validated here as well as in extract_mask_activity, so that the error names the offending mask.
        validated = _validate_mask(mask, deltaf_series.shape[1:], name)
        mask_activity[name] = extract_mask_activity(deltaf_series, validated)

    if as_dataframe:
        df = pd.DataFrame(mask_activity)
        return df.unstack().reset_index().rename(columns={"level_0": "region", "level_1": "time_idx", 0: "F"})

    return mask_activity


def _region_means(data_flat: npt.NDArray, masks: npt.NDArray) -> npt.NDArray:
    """Mean of each region mask per frame, ignoring NaN pixels.

    Args:
        data_flat (npt.NDArray): A 2D array of shape (time, height * width) of ∆F/F values.
        masks (npt.NDArray): A 2D boolean array of shape (n_regions, height * width) selecting each region.

    Returns:
        npt.NDArray: A 2D array of shape (time, n_regions). A region is NaN in a frame if it has no non-NaN
            pixel in that frame.
    """
    dtype = np.promote_types(data_flat.dtype, np.float32)
    masks_t = masks.T.astype(dtype)  # (H*W, n_regions)
    activities = np.empty((data_flat.shape[0], masks.shape[0]), dtype=np.float64)

    for start in range(0, data_flat.shape[0], _BLOCK_FRAMES):
        block = data_flat[start : start + _BLOCK_FRAMES]
        valid = ~np.isnan(block)
        # Zeroing the NaNs keeps them out of every region's dot product, not just out of their own.
        # Some BLAS backends raise spurious FP flags on any matmul, so only the division below is left
        # to warn, which it does when a region has no valid pixel in a frame.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sums = np.where(valid, block, 0) @ masks_t
            counts = (valid.astype(dtype) @ masks_t).astype(np.float64)
        activities[start : start + _BLOCK_FRAMES] = sums / counts  # a zero count gives NaN

    return activities


def _validate_mask(mask: npt.NDArray, frame_shape: tuple[int, ...], name: str | None = None) -> npt.NDArray[np.bool_]:
    """Check a region mask against the frame shape of a recording and return it as a boolean array.

    Args:
        mask (npt.NDArray): A 2D array selecting the pixels of a region.
        frame_shape (tuple[int, ...]): Shape of a single recording frame, i.e. (height, width).
        name (str | None, optional): Name of the mask, used in error messages. Defaults to None.

    Returns:
        npt.NDArray[np.bool_]: The mask as a boolean array.

    Raises:
        ValueError: If the mask shape does not match the frame shape, or if the mask does not select any pixels.
    """
    label = f"Mask {name}" if name else "Mask"
    mask = np.asarray(mask)

    if mask.shape != tuple(frame_shape):
        msg = f"{label} has shape {mask.shape}, which does not match the recording frame shape {tuple(frame_shape)}."
        raise ValueError(msg)

    mask = mask.astype(bool)
    if not mask.any():
        msg = f"{label} does not select any pixels."
        raise ValueError(msg)

    return mask
