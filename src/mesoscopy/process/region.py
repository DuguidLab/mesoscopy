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
            region over time.

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
        region_mask = np.broadcast_to(left_aba == region_id, deltaf_series.shape)
    elif hemisphere.lower() == "right":
        region_mask = np.broadcast_to(right_aba == region_id, deltaf_series.shape)
    elif hemisphere.lower() == "both":
        aba = left_aba + right_aba
        region_mask = np.broadcast_to(aba == region_id, deltaf_series.shape)
    else:
        msg = f"Could not recognise hemisphere option {hemisphere}, select left, right or both."
        raise ValueError(msg)

    return np.ma.array(deltaf_series, mask=~region_mask).mean(axis=(1, 2))


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
            or a DataFrame with columns 'region', 'time_idx', and 'F'.
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
    left_masks = np.array([(left_aba == region_ids[r]).ravel() for r in regions])   # (n_regions, H*W)
    right_masks = np.array([(right_aba == region_ids[r]).ravel() for r in regions])

    left_counts = left_masks.sum(axis=1).astype(float)   # (n_regions,)
    right_counts = right_masks.sum(axis=1).astype(float)

    data_flat = deltaf_series.reshape(deltaf_series.shape[0], hw)  # (T, H*W)
    left_activities = (data_flat @ left_masks.T) / left_counts     # (T, n_regions)
    right_activities = (data_flat @ right_masks.T) / right_counts

    region_activity = {}
    for i, region in enumerate(regions):
        region_activity[f"L_{region}"] = left_activities[:, i]
        region_activity[f"R_{region}"] = right_activities[:, i]

    if as_dataframe:
        df = pd.DataFrame(region_activity)
        return df.unstack().reset_index().rename(columns={"level_0": "region", "level_1": "time_idx", 0: "F"})

    return region_activity
