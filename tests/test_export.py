import os
import pathlib
from datetime import datetime
from datetime import timedelta

import h5py as h5
import numpy as np
import pytest
from click.testing import CliRunner
from pynwb import NWBHDF5IO
from pynwb.ophys import ImageSeries

import mesoscopy
import mesoscopy.export as export
import mesoscopy.export.nwb as nwb_export


def test_nwb_sharing_export(preproc_nwb, output_dir):
    """Test the nwb_sharing_export function."""
    out_path = os.path.join(output_dir, "test_export.nwb")
    exported_path = nwb_export.export_standalone(preproc_nwb, out_path)
    assert exported_path.endswith("_export.nwb")
    assert exported_path != preproc_nwb

    # Clean up the created file
    if pathlib.Path(exported_path).exists():
        pathlib.Path(exported_path).unlink()


def test_nwb_sharing_export_no_outpath(preproc_nwb):
    """Test the nwb_sharing_export function without specifying an output path."""
    exported_path = nwb_export.export_standalone(preproc_nwb)
    assert exported_path.endswith("_export.nwb")
    assert exported_path != preproc_nwb

    # Clean up the created file
    if pathlib.Path(exported_path).exists():
        pathlib.Path(exported_path).unlink()


@pytest.fixture
def preproc_h5_iso_timestamps(tmp_path_factory):
    """Create an HDF5 file with preprocessed data and ISO-format byte string timestamps."""
    tmpfile = tmp_path_factory.mktemp("data") / "preproc_iso.h5"
    frames_num = 10
    session_start_time = datetime(2024, 1, 1, 14, 0, 0)
    timestamps = [
        (session_start_time + timedelta(milliseconds=i)).isoformat().encode("utf-8") for i in range(frames_num)
    ]

    with h5.File(str(tmpfile), "w") as f:
        f.create_dataset("/F", data=np.random.rand(frames_num, 10, 10))
        f.create_dataset("/timestamps", data=timestamps, dtype="S26")

    return str(tmpfile), timestamps


@pytest.fixture
def preproc_nwb_iso_timestamps(nwbfile, preproc_h5_iso_timestamps):
    """Create an NWBFile with a DeltaFSeries using ISO-format byte string timestamps."""
    h5_path, timestamps = preproc_h5_iso_timestamps

    with h5.File(h5_path, "r") as f:
        io = NWBHDF5IO(nwbfile, "a")
        nwb = io.read()
        # Pass the raw h5py datasets (rather than materialized numpy arrays) so pynwb keeps the
        # underlying byte-string dtype instead of attempting (and failing) to coerce it to floats.
        deltaF_series = ImageSeries(
            name="DeltaFSeries",
            data=f["/F"],
            timestamps=f["/timestamps"],
            unit="df/f",
            description="dF/F widefield cortical imaging series.",
            comments="This imaging series is corrected for the haemodynamic response.",
        )

        ophys_module = nwb.create_processing_module(name="ophys", description="optical physiology processed data")
        ophys_module.add(deltaF_series)

        io.write(nwb)

    return nwbfile, timestamps


def test_export_timestamps_h5(preproc_h5_iso_timestamps, output_dir):
    """Test export_timestamps with an HDF5 source file."""
    h5_path, timestamps = preproc_h5_iso_timestamps
    session_id = pathlib.Path(h5_path).stem

    exported_path = export.export_timestamps(h5_path, output_dir)

    assert exported_path == os.path.join(output_dir, f"{session_id}_timestamps.txt")
    assert pathlib.Path(exported_path).is_file()

    exported_timestamps = np.loadtxt(exported_path, dtype=str)
    expected_timestamps = [ts.decode("utf-8") for ts in timestamps]
    assert list(exported_timestamps) == expected_timestamps


def test_export_timestamps_nwb(preproc_nwb_iso_timestamps, output_dir):
    """Test export_timestamps with an NWB source file."""
    nwb_path, timestamps = preproc_nwb_iso_timestamps
    session_id = pathlib.Path(nwb_path).stem

    exported_path = export.export_timestamps(nwb_path, output_dir)

    assert exported_path == os.path.join(output_dir, f"{session_id}_timestamps.txt")
    assert pathlib.Path(exported_path).is_file()

    exported_timestamps = np.loadtxt(exported_path, dtype=str)
    expected_timestamps = [ts.decode("utf-8") for ts in timestamps]
    assert list(exported_timestamps) == expected_timestamps


def test_export_timestamps_cmd(preproc_h5_iso_timestamps, output_dir):
    """Check the click export timestamps cmd caller works."""
    h5_path, _ = preproc_h5_iso_timestamps
    session_id = pathlib.Path(h5_path).stem

    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli,
        args=f"export timestamps {h5_path} --out-dir {output_dir}",
    )

    assert result.exit_code == 0
    assert "Exporting timestamps..." in result.output
    assert f"Timestamps exported to {output_dir}" in result.output

    exported_path = os.path.join(output_dir, f"{session_id}_timestamps.txt")
    assert pathlib.Path(exported_path).is_file()


def test_nwb_sharing_export_invalid_path():
    """Test the nwb_sharing_export function with an invalid NWB file path."""
    with pytest.raises(FileNotFoundError):
        nwb_export.export_standalone("invalid_path.nwb")

    # Ensure no file is created
    assert not pathlib.Path("invalid_path_export.nwb").exists()


def test_nwb_sharing_cmd_parity(preproc_nwb, output_dir):
    """Check if click export nwb cmd caller works."""
    runner = CliRunner()
    out_path = os.path.join(output_dir, "test_export.nwb")
    result = runner.invoke(
        mesoscopy.cli,
        args=f"export nwb {preproc_nwb} --out-path {out_path}",
    )
    assert result.exit_code == 0
    assert "Exporting NWB file..." in result.output
    assert "NWB file exported to" in result.output

    exported_path = os.path.join(output_dir, "test_export.nwb")
    assert pathlib.Path(exported_path).is_file()

    # Clean up the created file
    if pathlib.Path(exported_path).exists():
        pathlib.Path(exported_path).unlink()
