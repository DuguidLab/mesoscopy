from importlib import resources

import numpy as np
from click.testing import CliRunner

import mesoscopy
import mesoscopy.register as reg
import mesoscopy.resources as res


def test_landmarks_affine():
    """Test affine transform calculation for defined landmarks"""


def test_load_preprocessed_h5(preproc_h5):
    session_id, deltaf, ts = reg.load_deltaf(preproc_h5)
    assert session_id == "preproc"
    assert deltaf.shape == (300, 40, 40)
    assert ts.shape == (300,)


def test_load_preprocessed_nwb(preproc_nwb):
    session_id, deltaf, ts = reg.load_deltaf(preproc_nwb, nwb=True)
    assert session_id == "session_1234"
    assert deltaf.shape == (300, 40, 40)
    assert ts.shape == (300,)


def test_update_nwb(nwbfile, preproc_h5):
    tform_mock = np.array([[(1, 2, 3), (1, 2, 4)]])
    nwb = reg.update_nwb(nwbfile, preproc_h5, tform_mock)
    assert nwb.processing["ophys"]["CCFRegisteredSeries"].corrected.data
    assert nwb.processing["ophys"]["CCFRegisteredSeries"].original.data
    assert nwb.processing["ophys"]["CCFRegisteredSeries"].xy_translation.data.any()


def test_register_landmarks_cli_default_points_h5(preproc_h5, output_dir):
    mock_recording_landmarks = str(resources.files(res).joinpath("ccf_template_landmarks_140x142.csv"))
    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli,
        args=f"register landmarks {preproc_h5} -r {mock_recording_landmarks} -o {output_dir}",
    )
    assert result.exit_code == 0


def test_mark_landmarks_gui(): ...
