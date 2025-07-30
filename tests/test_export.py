import os
import pathlib

import pytest
from click.testing import CliRunner

import mesoscopy
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
