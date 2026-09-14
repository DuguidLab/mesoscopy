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

"""Behaviour alignment submodule."""

import json
from datetime import datetime
from pathlib import Path

import click
import h5py
import numpy as np

from mesoscopy import io

DEFAULT_MAX_OFFSET_S = 600.0  # Maximum allowed offset of the first frame from the behaviour session start, in seconds.


@click.command("align")
@click.argument(
    "path",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--session-start",
    type=str,
    default=None,
    help="Behaviour session start time, ISO 8601. Mutually exclusive with --behaviour-json.",
)
@click.option(
    "--behaviour-json",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Visiomode behaviour session JSON file; its 'timestamp' key gives the session start.",
)
@click.option(
    "--behaviour-id",
    type=str,
    default=None,
    help="Behaviour session identifier. Required with --session-start, defaults to the JSON file stem otherwise.",
)
@click.option(
    "--max-offset",
    type=float,
    default=DEFAULT_MAX_OFFSET_S,
    show_default=True,
    help="Largest allowed offset, in seconds, between the first frame and the behaviour session start.",
)
def align_cmd(
    path: str,
    session_start: str | None,
    behaviour_json: str | None,
    behaviour_id: str | None,
    max_offset: float,
) -> None:
    """Write behaviour-aligned frame timestamps to a preprocessed or registered HDF5 recording."""  # noqa: DOC501
    if (session_start is None) == (behaviour_json is None):
        msg = "Exactly one of --session-start or --behaviour-json is required."
        raise click.UsageError(msg)

    if behaviour_json is not None:
        session_start = read_session_start(behaviour_json)
        if behaviour_id is None:
            behaviour_id = Path(behaviour_json).stem
    elif behaviour_id is None:
        msg = "--behaviour-id is required with --session-start."
        raise click.UsageError(msg)

    assert session_start is not None  # noqa: S101
    try:
        offset_s, n_frames = align_recording(path, session_start, behaviour_id, max_offset_s=max_offset)
    except ValueError as err:
        raise click.ClickException(str(err)) from err

    click.echo(f"Aligned {n_frames} frames to behaviour session {behaviour_id} (offset {offset_s:.3f} s).")


def align_recording(
    h5_path: str,
    session_start: str | datetime,
    behaviour_session: str,
    max_offset_s: float = DEFAULT_MAX_OFFSET_S,
) -> tuple[float, int]:
    """Write `/timestamps_aligned` to an HDF5 recording, as seconds from the behaviour session start.

    Args:
        h5_path (str): Path to the HDF5 recording holding an ISO 8601 `/timestamps` dataset.
        session_start (str | datetime): Behaviour session start time, ISO 8601.
        behaviour_session (str): Behaviour session identifier.
        max_offset_s (float, optional): Largest allowed absolute offset of the first frame from the session start,
            in seconds. Defaults to 600.

    Returns:
        tuple[float, int]: Offset of the first frame from the session start in seconds, and number of frames.

    Raises:
        ValueError: If the offset exceeds `max_offset_s`. Nothing is written in that case.
    """
    if isinstance(session_start, str):
        session_start = datetime.fromisoformat(session_start)

    with h5py.File(h5_path, "r") as f:
        timestamps = f["/timestamps"][:]

    frame_times = [
        datetime.fromisoformat(ts.decode("utf-8") if isinstance(ts, bytes) else str(ts)) for ts in timestamps
    ]

    # If the frame timestamps and session start have different timezone awareness, make them both naive.
    if frame_times and (frame_times[0].tzinfo is None) != (session_start.tzinfo is None):
        session_start = session_start.replace(tzinfo=None)
        frame_times = [ts.replace(tzinfo=None) for ts in frame_times]

    aligned = np.array([(ts - session_start).total_seconds() for ts in frame_times], dtype=np.float64)
    offset_s = float(aligned[0])

    if abs(offset_s) > max_offset_s:
        msg = (
            f"First frame is {offset_s:.3f} s from the behaviour session start, exceeding the maximum offset of"
            f" {max_offset_s:.3f} s."
        )
        raise ValueError(msg)

    io.write_timestamps_aligned(
        h5_path,
        aligned,
        {
            "session_start_time": session_start.isoformat(),
            "behaviour_session": behaviour_session,
            "offset_s": offset_s,
        },
    )

    return offset_s, len(aligned)


def read_session_start(behaviour_json_path: str) -> str:
    """Read the session start time from a visiomode behaviour session JSON file.

    Args:
        behaviour_json_path (str): Path to the behaviour session JSON file.

    Returns:
        str: The file's `timestamp` value, ISO 8601.
    """
    with Path(behaviour_json_path).open(encoding="utf-8") as fp:
        return str(json.load(fp)["timestamp"])
