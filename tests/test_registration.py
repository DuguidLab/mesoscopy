from importlib import resources

import numpy as np
import pytest
from click.testing import CliRunner
from skimage import transform as trf

import mesoscopy
import mesoscopy.register as reg
import mesoscopy.resources as res
from mesoscopy.register.transform import landmarks_affine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_series():
    """A small (10, 40, 40) float32 DeltaF/F series with a fixed seed."""
    rng = np.random.default_rng(0)
    return rng.random((10, 40, 40), dtype=np.float32)


@pytest.fixture
def landmark_pair():
    """Matching recording / template landmark dicts (pure translation: +2 col, +3 row)."""
    recording = {"bregma": (10.0, 10.0), "lambda": (30.0, 10.0), "midline": (10.0, 25.0)}
    template  = {"bregma": (12.0, 13.0), "lambda": (32.0, 13.0), "midline": (12.0, 28.0)}
    return recording, template


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_landmarks_affine_output_shape(small_series, landmark_pair):
    """Warped series must have the same shape as the input."""
    recording_lm, template_lm = landmark_pair
    warped, _ = landmarks_affine(small_series, recording_lm, template_lm)
    assert warped.shape == small_series.shape


def test_landmarks_affine_returns_projective_transform(small_series, landmark_pair):
    """Second return value must be a skimage ProjectiveTransform."""
    recording_lm, template_lm = landmark_pair
    _, tform = landmarks_affine(small_series, recording_lm, template_lm)
    assert isinstance(tform, trf.ProjectiveTransform)


def test_landmarks_affine_transform_params(small_series, landmark_pair):
    """Returned transform must match the one computed directly from the landmarks."""
    recording_lm, template_lm = landmark_pair
    _, tform = landmarks_affine(small_series, recording_lm, template_lm)

    template  = np.array(list(template_lm.values()),  dtype=np.float32)
    recording = np.array(list(recording_lm.values()), dtype=np.float32)
    expected_tform = trf.estimate_transform("affine", template, recording)

    np.testing.assert_allclose(tform.params, expected_tform.params, atol=1e-6)


def test_landmarks_affine_matches_sequential_reference(small_series, landmark_pair):
    """Parallel output must be numerically identical to the sequential per-frame loop."""
    recording_lm, template_lm = landmark_pair

    template  = np.array(list(template_lm.values()),  dtype=np.float32)
    recording = np.array(list(recording_lm.values()), dtype=np.float32)
    ref_tform = trf.estimate_transform("affine", template, recording)
    reference = np.array([trf.warp(small_series[i], ref_tform, order=3) for i in range(small_series.shape[0])])

    warped, _ = landmarks_affine(small_series, recording_lm, template_lm)

    np.testing.assert_array_equal(warped, reference)


def test_landmarks_affine_crop(landmark_pair):
    """With crop_x / crop_y the warped frames should be cropped to those dimensions."""
    rng = np.random.default_rng(1)
    series = rng.random((5, 40, 40), dtype=np.float32)
    recording_lm, template_lm = landmark_pair
    crop_x, crop_y = 30, 25

    warped, _ = landmarks_affine(series, recording_lm, template_lm, crop_x=crop_x, crop_y=crop_y)

    assert warped.shape == (5, crop_y, crop_x)


def test_landmarks_affine_crop_matches_sequential_reference(landmark_pair):
    """Cropped parallel output must be identical to the sequential cropped loop."""
    rng = np.random.default_rng(2)
    series = rng.random((5, 40, 40), dtype=np.float32)
    recording_lm, template_lm = landmark_pair
    crop_x, crop_y = 30, 25

    template  = np.array(list(template_lm.values()),  dtype=np.float32)
    recording = np.array(list(recording_lm.values()), dtype=np.float32)
    ref_tform = trf.estimate_transform("affine", template, recording)
    reference = np.array([trf.warp(series[i, :crop_y, :crop_x], ref_tform, order=3) for i in range(series.shape[0])])

    warped, _ = landmarks_affine(series, recording_lm, template_lm, crop_x=crop_x, crop_y=crop_y)

    np.testing.assert_array_equal(warped, reference)


def test_landmarks_affine_identity_landmarks(small_series):
    """When recording and template landmarks are identical the warp is a near-identity."""
    landmarks = {"A": (5.0, 5.0), "B": (35.0, 5.0), "C": (5.0, 35.0)}
    warped, _ = landmarks_affine(small_series, landmarks, landmarks)

    # Interior pixels should be reproduced almost exactly (boundary pixels may differ
    # due to the constant-zero padding convention of skimage warp).
    interior = small_series[:, 5:-5, 5:-5]
    warped_interior = warped[:, 5:-5, 5:-5]
    np.testing.assert_allclose(warped_interior, interior, atol=1e-5)


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
