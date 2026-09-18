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
"""Data preparation for the peri-event HTML report."""

import base64
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage import measure

from mesoscopy import resources

# Per-trial metrics columns embedded for the trace markers and box plots.
TRIAL_METRIC_COLUMNS = [
    "trial_index",
    "region",
    "baseline_mean",
    "onset_time",
    "peak_time",
    "amplitude",
    "auc",
    "decay_time",
    "offset_time",
    "duration",
]

UNLABELLED = "unlabelled"


def read_perievent(path: str) -> pd.DataFrame:
    """Read a long-format peri-event CSV.

    Args:
        path (str): `*_perievent.csv` written by `process peri-event`.

    Returns:
        pd.DataFrame: The table.

    Raises:
        ValueError: If the table is empty or lacks the peri-event columns.
    """
    table = pd.read_csv(path)
    missing = {"trial_index", "event_time", "time", "region", "F"} - set(table.columns)
    if missing:
        msg = f"{path} is missing peri-event columns: {', '.join(sorted(missing))}."
        raise ValueError(msg)
    if table.empty:
        msg = f"{path} has no trials."
        raise ValueError(msg)
    return table


def perievent_cube(table: pd.DataFrame) -> tuple[list[str], npt.NDArray, npt.NDArray, npt.NDArray]:
    """Pivot a long-format peri-event table into a (trial, sample, region) cube.

    Args:
        table (pd.DataFrame): Peri-event table with `trial_index`, `time`, `region` and `F` columns.

    Returns:
        tuple[list[str], npt.NDArray, npt.NDArray, npt.NDArray]: Region names in first-seen order, the
            sample times, the trial indices in first-seen order, and the float32 cube. Missing
            (trial, time, region) combinations are NaN.
    """
    regions = list(table["region"].unique())
    trial_index = table["trial_index"].unique()
    time = np.sort(table["time"].unique())
    wide = table.pivot_table(index=["trial_index", "time"], columns="region", values="F", aggfunc="first")
    wide = wide.reindex(pd.MultiIndex.from_product([trial_index, time]), columns=regions)
    cube = wide.to_numpy(dtype=np.float32).reshape(len(trial_index), len(time), len(regions))
    return regions, time, trial_index, cube


def trial_types(table: pd.DataFrame, trial_index: npt.NDArray, trials: pd.DataFrame | None = None) -> list[str] | None:
    """Trial type per trial, from the peri-event table's `sdt_type` or a trials table.

    Args:
        table (pd.DataFrame): Peri-event table.
        trial_index (npt.NDArray): Trials to label, as trials CSV row indices.
        trials (pd.DataFrame, optional): Trials table used when the peri-event table has no `sdt_type`.

    Returns:
        list[str] | None: One label per trial, with missing labels as `unlabelled`; None when
            neither source has an `sdt_type` column.
    """
    if "sdt_type" in table.columns:
        labels = table.groupby("trial_index", sort=False)["sdt_type"].first().reindex(trial_index)
    elif trials is not None and "sdt_type" in trials.columns:
        labels = trials["sdt_type"].reset_index(drop=True).reindex(trial_index)
    else:
        return None
    return [UNLABELLED if pd.isna(label) or not label else str(label) for label in labels]


def pack_float32(array: npt.NDArray) -> str:
    """Pack an array as base64 little-endian float32, C order.

    Args:
        array (npt.NDArray): Array to pack.

    Returns:
        str: Base64 text.
    """
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode("ascii")


def unpack_float32(text: str, shape: tuple[int, ...]) -> npt.NDArray:
    """Inverse of `pack_float32`.

    Args:
        text (str): Base64 text.
        shape (tuple[int, ...]): Array shape.

    Returns:
        npt.NDArray: The float32 array.
    """
    return np.frombuffer(base64.b64decode(text), dtype="<f4").reshape(shape)


def atlas_outlines(step: int = 2) -> dict[str, str]:
    """Allen CCF top-view region outlines as SVG path data, keyed `L_<acronym>` / `R_<acronym>`.

    Args:
        step (int, optional): Keep every `step`-th contour vertex. Defaults to 2.

    Returns:
        dict[str, str]: SVG path data per region, in atlas pixel coordinates.
    """
    left, right = resources.get_atlas()
    annotations = resources.get_atlas_annotations()
    acronyms = dict(zip(annotations["id"], annotations["acronym"], strict=True))
    outlines = {}
    for prefix, labels in (("L", left), ("R", right)):
        for region_id in np.unique(labels):
            if region_id == 0 or region_id not in acronyms:
                continue
            mask = np.pad((labels == region_id).astype(float), 1)
            parts = []
            for contour in measure.find_contours(mask, 0.5):
                points = " ".join(f"{x - 1:.1f},{y - 1:.1f}" for y, x in contour[::step])
                parts.append(f"M{points}Z")
            outlines[f"{prefix}_{acronyms[region_id]}"] = " ".join(parts)
    return outlines


def metrics_paths(path: str) -> tuple[Path, Path]:
    """Paths of the metrics tables `process metrics` writes next to a peri-event CSV.

    Args:
        path (str): `*_perievent.csv` path.

    Returns:
        tuple[Path, Path]: `<stem>_metrics.csv` and `<stem>_metrics-session.csv` paths, whether or not
            they exist.
    """
    stem = Path(path).with_suffix("").name.removesuffix("_perievent")
    parent = Path(path).parent
    return parent / f"{stem}_metrics.csv", parent / f"{stem}_metrics-session.csv"


def _records(table: pd.DataFrame) -> list[dict]:
    return table.astype(object).where(table.notna(), None).to_dict(orient="records")


def report_payload(path: str, trials_path: str | None = None) -> tuple[dict, list[str]]:
    """Build the JSON payload for the peri-event report.

    Args:
        path (str): `*_perievent.csv` written by `process peri-event`.
        trials_path (str, optional): Trials CSV giving `sdt_type` when the peri-event file lacks it.

    Returns:
        tuple[dict, list[str]]: The payload and any warnings to show the user.
    """
    table = read_perievent(path)
    trials = pd.read_csv(trials_path) if trials_path else None
    regions, time, trial_index, cube = perievent_cube(table)
    warnings = []

    types = trial_types(table, trial_index, trials)
    if types is None:
        warnings.append("No sdt_type column in the peri-event file or a trials file; trial-type filtering is off.")

    event_time = table.groupby("trial_index", sort=False)["event_time"].first().reindex(trial_index)

    trial_metrics_path, session_metrics_path = metrics_paths(path)
    trial_metrics = session_metrics = None
    if trial_metrics_path.exists() and session_metrics_path.exists():
        trial_metrics = pd.read_csv(trial_metrics_path)
        session_metrics = pd.read_csv(session_metrics_path)
        keep = [column for column in TRIAL_METRIC_COLUMNS if column in trial_metrics.columns]
        trial_metrics = trial_metrics[keep]
    else:
        warnings.append(f"No metrics tables found next to {Path(path).name}; the metrics section is omitted.")

    event = Path(path).name.rsplit("_event-", 1)[-1].removesuffix("_perievent.csv") if "_event-" in path else None
    payload = {
        "regions": regions,
        "time": [float(t) for t in time],
        "trial_index": [int(t) for t in trial_index],
        "event_time": [None if pd.isna(t) else float(t) for t in event_time],
        "sdt_type": types,
        "event": event,
        "shape": list(cube.shape),
        "traces": pack_float32(cube),
        "atlas": {"shape": list(resources.atlas_shape()), "paths": atlas_outlines()},
        "trial_metrics": None if trial_metrics is None else _records(trial_metrics),
        "session_metrics": None if session_metrics is None else _records(session_metrics),
    }
    return payload, warnings
