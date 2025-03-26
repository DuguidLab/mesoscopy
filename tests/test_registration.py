import pytest
from click.testing import CliRunner

import numpy as np

import mesoscopy
import mesoscopy.register as register


def test_landmarks_affine():
    """Test affine transform calculation for defined landmarks"""


def test_load_preprocessed_h5(preproc_h5):
    session_id, deltaf, ts = register.load_preprocessed(preproc_h5)
    assert session_id == "preproc"
    assert deltaf.shape == (300, 40, 40)
    assert ts.shape == (300,)


def test_load_preprocessed_nwb(preproc_nwb):
    session_id, deltaf, ts = register.load_preprocessed(preproc_nwb, nwb=True)
    assert session_id == "session_1234"
    assert deltaf.shape == (300, 40, 40)
    assert ts.shape == (300,)


def test_update_nwb(nwbfile, preproc_h5):
    tform_mock = np.array([[(1, 2, 3), (1, 2, 4)]])
    nwb = register.update_nwb(nwbfile, preproc_h5, tform_mock)
    assert nwb.processing["ophys"]["CCFRegisteredSeries"].corrected.data
    assert nwb.processing["ophys"]["CCFRegisteredSeries"].original.data
    assert nwb.processing["ophys"]["CCFRegisteredSeries"].xy_translation.data.any()


def test_register_landmarks_cli(): ...


def test_mark_landmarks_gui(): ...
