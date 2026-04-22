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

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Literal
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


def extract_all_regions(deltaf_series): ...
