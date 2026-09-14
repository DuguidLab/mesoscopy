import json
from datetime import UTC
from datetime import datetime
from datetime import timedelta

import h5py as h5
import numpy as np
import pytest
from click.testing import CliRunner

from mesoscopy import align
from mesoscopy import io

SESSION_START = datetime(2024, 1, 1, 14, 0, 0)
N_FRAMES = 300
FRAME_INTERVAL_S = 0.04


@pytest.fixture
def recording_h5(tmp_path):
    """Preprocessed HDF5 file with ISO 8601 byte-string timestamps starting 2 s after SESSION_START."""
    tmpfile = tmp_path / "recording_preprocessed.h5"
    timestamps = [
        (SESSION_START + timedelta(seconds=2 + i * FRAME_INTERVAL_S)).isoformat().encode("utf-8")
        for i in range(N_FRAMES)
    ]
    with h5.File(str(tmpfile), "w") as f:
        f.create_dataset("/F", data=np.random.rand(N_FRAMES, 4, 4))
        f.create_dataset("/timestamps", data=timestamps, dtype="S26")
    return str(tmpfile)


@pytest.fixture
def behaviour_json(tmp_path):
    """Visiomode-style behaviour session JSON."""
    tmpfile = tmp_path / "sub-MM228_ses-20240101_behaviour-gonogo.json"
    tmpfile.write_text(json.dumps({"timestamp": SESSION_START.isoformat(), "trials": []}), encoding="utf-8")
    return str(tmpfile)


def read_aligned(path):
    with h5.File(path, "r") as f:
        return f["/timestamps_aligned"][:], dict(f["/timestamps_aligned"].attrs)


# ---------------------------------------------------------------------------------------------------------------------
# align.align_recording
# ---------------------------------------------------------------------------------------------------------------------


def test_align_recording_offset_and_length(recording_h5):
    offset_s, n_frames = align.align_recording(recording_h5, SESSION_START.isoformat(), "behaviour")

    aligned, _ = read_aligned(recording_h5)
    assert offset_s == pytest.approx(2.0)
    assert n_frames == N_FRAMES
    assert aligned.shape == (N_FRAMES,)
    assert aligned.dtype == np.float64
    assert aligned[0] == pytest.approx(offset_s)
    assert aligned[-1] == pytest.approx(2 + (N_FRAMES - 1) * FRAME_INTERVAL_S)


def test_align_recording_writes_attrs(recording_h5):
    align.align_recording(recording_h5, SESSION_START.isoformat(), "behaviour")

    _, attrs = read_aligned(recording_h5)
    assert attrs["session_start_time"] == SESSION_START.isoformat()
    assert attrs["behaviour_session"] == "behaviour"
    assert attrs["offset_s"] == pytest.approx(2.0)


def test_align_recording_accepts_negative_offset(recording_h5):
    session_start = SESSION_START + timedelta(seconds=10)
    offset_s, _ = align.align_recording(recording_h5, session_start.isoformat(), "behaviour")

    aligned, attrs = read_aligned(recording_h5)
    assert offset_s == pytest.approx(-8.0)
    assert aligned[0] == pytest.approx(-8.0)
    assert attrs["offset_s"] == pytest.approx(-8.0)


def test_align_recording_accepts_datetime_session_start(recording_h5):
    offset_s, _ = align.align_recording(recording_h5, SESSION_START, "behaviour")
    assert offset_s == pytest.approx(2.0)


def test_align_recording_over_max_offset_raises_and_writes_nothing(recording_h5):
    with pytest.raises(ValueError, match="maximum offset"):
        align.align_recording(recording_h5, SESSION_START.isoformat(), "behaviour", max_offset_s=1.0)

    with h5.File(recording_h5, "r") as f:
        assert "/timestamps_aligned" not in f


def test_align_recording_over_max_offset_leaves_existing_dataset(recording_h5):
    align.align_recording(recording_h5, SESSION_START.isoformat(), "first")
    before = read_aligned(recording_h5)

    with pytest.raises(ValueError, match="maximum offset"):
        align.align_recording(recording_h5, SESSION_START.isoformat(), "second", max_offset_s=1.0)

    after = read_aligned(recording_h5)
    np.testing.assert_array_equal(after[0], before[0])
    assert after[1]["behaviour_session"] == "first"


def test_align_recording_rerun_overwrites(recording_h5):
    align.align_recording(recording_h5, SESSION_START.isoformat(), "first")
    session_start = SESSION_START + timedelta(seconds=1)
    offset_s, _ = align.align_recording(recording_h5, session_start.isoformat(), "second")

    aligned, attrs = read_aligned(recording_h5)
    assert offset_s == pytest.approx(1.0)
    assert aligned[0] == pytest.approx(1.0)
    assert attrs["behaviour_session"] == "second"
    assert attrs["session_start_time"] == session_start.isoformat()


def test_align_recording_naive_frames_aware_session_start(recording_h5):
    session_start = SESSION_START.replace(tzinfo=UTC)
    offset_s, _ = align.align_recording(recording_h5, session_start.isoformat(), "behaviour")
    assert offset_s == pytest.approx(2.0)


# ---------------------------------------------------------------------------------------------------------------------
# align.read_session_start
# ---------------------------------------------------------------------------------------------------------------------


def test_read_session_start(behaviour_json):
    assert align.read_session_start(behaviour_json) == SESSION_START.isoformat()


# ---------------------------------------------------------------------------------------------------------------------
# io.read_timestamps_aligned / io.write_timestamps_aligned
# ---------------------------------------------------------------------------------------------------------------------


def test_read_timestamps_aligned_absent_returns_none(recording_h5):
    assert io.read_timestamps_aligned(recording_h5) is None


def test_read_timestamps_aligned_nwb_returns_none(preproc_nwb):
    assert io.read_timestamps_aligned(str(preproc_nwb)) is None


def test_write_and_read_timestamps_aligned(recording_h5):
    attrs = {"session_start_time": "2024-01-01T14:00:00", "behaviour_session": "ses", "offset_s": -1.5}
    io.write_timestamps_aligned(recording_h5, np.arange(N_FRAMES) - 1.5, attrs)

    result = io.read_timestamps_aligned(recording_h5)
    assert result is not None
    aligned, read_attrs = result
    np.testing.assert_allclose(aligned, np.arange(N_FRAMES) - 1.5)
    assert read_attrs == attrs
    assert isinstance(read_attrs["session_start_time"], str)
    assert isinstance(read_attrs["offset_s"], float)


# ---------------------------------------------------------------------------------------------------------------------
# mesoscopy align
# ---------------------------------------------------------------------------------------------------------------------


def test_align_cmd_session_start(recording_h5):
    result = CliRunner().invoke(
        align.align_cmd,
        [recording_h5, "--session-start", SESSION_START.isoformat(), "--behaviour-id", "ses"],
    )

    assert result.exit_code == 0, result.output
    assert "300 frames" in result.output
    assert "2.000 s" in result.output
    _, attrs = read_aligned(recording_h5)
    assert attrs["behaviour_session"] == "ses"


def test_align_cmd_behaviour_json_defaults_id_to_stem(recording_h5, behaviour_json):
    result = CliRunner().invoke(align.align_cmd, [recording_h5, "--behaviour-json", behaviour_json])

    assert result.exit_code == 0, result.output
    _, attrs = read_aligned(recording_h5)
    assert attrs["behaviour_session"] == "sub-MM228_ses-20240101_behaviour-gonogo"
    assert attrs["session_start_time"] == SESSION_START.isoformat()


def test_align_cmd_behaviour_id_overrides_stem(recording_h5, behaviour_json):
    result = CliRunner().invoke(
        align.align_cmd, [recording_h5, "--behaviour-json", behaviour_json, "--behaviour-id", "custom"]
    )

    assert result.exit_code == 0, result.output
    _, attrs = read_aligned(recording_h5)
    assert attrs["behaviour_session"] == "custom"


def test_align_cmd_requires_one_source(recording_h5, behaviour_json):
    result = CliRunner().invoke(align.align_cmd, [recording_h5])
    assert result.exit_code != 0
    assert "Exactly one" in result.output

    result = CliRunner().invoke(
        align.align_cmd,
        [recording_h5, "--session-start", SESSION_START.isoformat(), "--behaviour-json", behaviour_json],
    )
    assert result.exit_code != 0
    assert "Exactly one" in result.output


def test_align_cmd_session_start_requires_behaviour_id(recording_h5):
    result = CliRunner().invoke(align.align_cmd, [recording_h5, "--session-start", SESSION_START.isoformat()])
    assert result.exit_code != 0
    assert "--behaviour-id" in result.output


def test_align_cmd_max_offset_exceeded_fails(recording_h5):
    result = CliRunner().invoke(
        align.align_cmd,
        [recording_h5, "--session-start", SESSION_START.isoformat(), "--behaviour-id", "ses", "--max-offset", "1"],
    )

    assert result.exit_code != 0
    assert "maximum offset" in result.output
    with h5.File(recording_h5, "r") as f:
        assert "/timestamps_aligned" not in f
