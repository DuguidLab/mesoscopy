import pytest
from click.testing import CliRunner

import os
import mesoscopy
import mesoscopy.export.nwb as nwb_export


def test_nwb_sharing_export(preproc_nwb, output_dir):
    """Test the nwb_sharing_export function."""
    out_path = os.path.join(output_dir, "test_export.nwb")
    exported_path = nwb_export.export_standalone(preproc_nwb, out_path)
    assert exported_path.endswith("_export.nwb")
    assert exported_path != preproc_nwb

    # Clean up the created file
    if os.path.exists(exported_path):
        os.remove(exported_path)


def test_nwb_sharing_export_no_outpath(preproc_nwb):
    """Test the nwb_sharing_export function without specifying an output path."""
    exported_path = nwb_export.export_standalone(preproc_nwb)
    assert exported_path.endswith("_export.nwb")
    assert exported_path != preproc_nwb

    # Clean up the created file
    if os.path.exists(exported_path):
        os.remove(exported_path)


def test_nwb_sharing_export_invalid_path():
    """Test the nwb_sharing_export function with an invalid NWB file path."""
    with pytest.raises(FileNotFoundError):
        nwb_export.export_standalone("invalid_path.nwb")

    # Ensure no file is created
    assert not os.path.exists("invalid_path_export.nwb")


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
    assert os.path.isfile(exported_path)

    # Clean up the created file
    if os.path.exists(exported_path):
        os.remove(exported_path)
