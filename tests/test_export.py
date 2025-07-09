import pytest

import os

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
