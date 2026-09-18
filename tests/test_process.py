import pathlib
import warnings
from datetime import datetime
from datetime import timedelta
from unittest.mock import patch

import h5py as h5
import numpy as np
import pandas as pd
import pytest
import scipy.ndimage as ndi
import scipy.stats as sst
from click.testing import CliRunner

import mesoscopy
from mesoscopy import io
from mesoscopy.process import metrics as pm
from mesoscopy.process import perievent as pev
from mesoscopy.process import regression as regr
from mesoscopy.process import smooth
from mesoscopy.process import zscore
from mesoscopy.process.region import DEFAULT_EXCLUDE
from mesoscopy.process.region import extract_all_masks
from mesoscopy.process.region import extract_all_regions
from mesoscopy.process.region import extract_mask_activity
from mesoscopy.process.region import extract_region_activity

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def deltaf_series():
    """A small (10, 20, 20) float64 DeltaF/F series with a fixed seed."""
    rng = np.random.default_rng(0)
    return rng.random((10, 20, 20))


@pytest.fixture
def linear_regression_data():
    """Synthetic (time x pixels) data generated from known regression coefficients."""
    rng = np.random.default_rng(42)
    n_samples, n_regressors = 200, 3
    pixel_shape = (4, 5)
    n_pixels = pixel_shape[0] * pixel_shape[1]

    regressors = rng.normal(size=(n_samples, n_regressors))
    true_coefs = rng.normal(size=(n_regressors, n_pixels))

    deltaf_series = (regressors @ true_coefs).reshape(n_samples, *pixel_shape)

    return deltaf_series, regressors, true_coefs, pixel_shape


@pytest.fixture
def regressor_npz(tmp_path_factory):
    """Create an NPZ regressor file matching the (300, 40, 40) preproc_h5 fixture."""
    tmpfile = tmp_path_factory.mktemp("data") / "regressors.npz"
    rng = np.random.default_rng(1)
    regressors = rng.normal(size=(300, 3))
    labels = np.array(["reg_a", "reg_b", "reg_c"])
    np.savez(tmpfile, regressors=regressors, labels=labels)
    return str(tmpfile)


@pytest.fixture
def regressor_h5(tmp_path_factory):
    """Create an HDF5 regressor file matching the (300, 40, 40) preproc_h5 fixture."""
    tmpfile = tmp_path_factory.mktemp("data") / "regressors.h5"
    rng = np.random.default_rng(2)
    regressors = rng.normal(size=(300, 3))
    labels = ["reg_a", "reg_b", "reg_c"]
    with h5.File(str(tmpfile), "w") as f:
        f.create_dataset("regressors", data=regressors)
        f.create_dataset("labels", data=np.array(labels, dtype="S"))
        f.create_dataset("trial_idx", data=np.arange(300))
    return str(tmpfile)


@pytest.fixture
def aligned_preproc_h5(preproc_h5):
    """Copy of the (300, 40, 40) preproc_h5 fixture carrying behaviour-aligned timestamps."""
    path = pathlib.Path(preproc_h5).with_name("preproc_aligned.h5")
    path.write_bytes(pathlib.Path(preproc_h5).read_bytes())
    io.write_timestamps_aligned(
        str(path),
        np.arange(300) * 0.04 + 2.0,
        {"session_start_time": "2024-01-01T14:00:00", "behaviour_session": "ses-aligned", "offset_s": 2.0},
    )
    return str(path)


@pytest.fixture
def aligned_preproc_h5_bytes_timestamps(preproc_h5_bytes_timestamps):
    """Copy of `preproc_h5_bytes_timestamps` carrying behaviour-aligned timestamps."""
    source = pathlib.Path(preproc_h5_bytes_timestamps)
    path = source.with_name("preproc_bytes_aligned.h5")
    path.write_bytes(source.read_bytes())
    io.write_timestamps_aligned(
        str(path),
        np.arange(300) * 0.001 + 2.0,
        {"session_start_time": "2024-01-01T14:00:00", "behaviour_session": "ses-aligned", "offset_s": 2.0},
    )
    return str(path)


@pytest.fixture
def aligned_regressor_npz(tmp_path_factory):
    """NPZ regressor file matching `aligned_preproc_h5`, carrying the behaviour alignment keys."""
    tmpfile = tmp_path_factory.mktemp("data") / "regressors_aligned.npz"
    rng = np.random.default_rng(5)
    np.savez(
        tmpfile,
        regressors=rng.normal(size=(300, 3)),
        labels=np.array(["reg_a", "reg_b", "reg_c"]),
        timestamps=np.arange(300) * 0.04 + 2.0,
        trial_idx=np.arange(300),
        session_start_time="2024-01-01T14:00:00",
        behaviour_session="ses-aligned",
    )
    return str(tmpfile)


@pytest.fixture
def nuisance_regressor_h5(tmp_path_factory):
    """Create an HDF5 nuisance regressor file on its own (coarser) timebase, spanning the same duration as the
    (300, 40, 40) preproc_h5 fixture's timestamps (0 to 0.299 seconds)."""
    tmpfile = tmp_path_factory.mktemp("data") / "nuisance.h5"
    rng = np.random.default_rng(3)
    n_samples = 50
    timestamps = np.linspace(0, 0.3, n_samples)
    with h5.File(str(tmpfile), "w") as f:
        f.create_dataset("motion_a", data=rng.normal(10, 2, size=n_samples))
        f.create_dataset("motion_b", data=rng.normal(-5, 1, size=n_samples))
        f.create_dataset("timestamps", data=timestamps)
    return str(tmpfile)


@pytest.fixture
def nuisance_regressor_npz(tmp_path_factory):
    """Create an NPZ nuisance regressor file on its own (coarser) timebase, spanning the same duration as the
    (300, 40, 40) preproc_h5 fixture's timestamps (0 to 0.299 seconds). Uses different labels from
    `nuisance_regressor_h5` so both fixtures can be combined in the same regression run."""
    tmpfile = tmp_path_factory.mktemp("data") / "nuisance.npz"
    rng = np.random.default_rng(4)
    n_samples = 50
    timestamps = np.linspace(0, 0.3, n_samples)
    np.savez(
        tmpfile,
        pupil_a=rng.normal(10, 2, size=n_samples),
        pupil_b=rng.normal(-5, 1, size=n_samples),
        timestamps=timestamps,
    )
    return str(tmpfile)


@pytest.fixture
def preproc_h5_bytes_timestamps(tmp_path_factory):
    """Preprocessed HDF5 file with byte-string timestamps, matching the real preprocessing pipeline's
    output format. `process regions` decodes timestamps as byte strings, unlike the plain-float
    timestamps used by the shared `preproc_h5` fixture.
    """
    tmpfile = tmp_path_factory.mktemp("data") / "preproc_bytes.h5"
    frames_num = 300
    session_start = datetime(2024, 1, 1, 14, 0, 0)
    timestamps = [(session_start + timedelta(milliseconds=i)).isoformat().encode("utf-8") for i in range(frames_num)]

    with h5.File(str(tmpfile), "w") as f:
        f.create_dataset("/F", data=np.random.rand(frames_num, 40, 40))
        f.create_dataset("/timestamps", data=timestamps, dtype="S26")

    return str(tmpfile)


@pytest.fixture
def mock_left_aba():
    """A 40x40 mock atlas matching the preproc_h5 fixture's spatial dimensions."""
    aba = np.zeros((40, 40), dtype=np.uint16)
    aba[:, :20] = 1
    return aba


@pytest.fixture
def mock_right_aba(mock_left_aba):
    return np.flip(mock_left_aba, axis=1).copy()


@pytest.fixture
def mock_annotations():
    return pd.DataFrame({"id": [1], "acronym": ["REG1"]})


@pytest.fixture
def mask_npy(tmp_path_factory):
    """A boolean 40x40 mask file covering the left half of the preproc_h5 fixture's frames."""
    path = tmp_path_factory.mktemp("masks") / "roi.npy"
    mask = np.zeros((40, 40), dtype=bool)
    mask[:, :20] = True
    np.save(path, mask)
    return str(path)


@pytest.fixture
def labelled_mask_npy(tmp_path_factory):
    """A 40x40 label image holding two labelled regions."""
    path = tmp_path_factory.mktemp("masks") / "labelled.npy"
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[:20] = 1
    mask[20:] = 2
    np.save(path, mask)
    return str(path)


# ---------------------------------------------------------------------------
# smooth.laplace_gaussian
# ---------------------------------------------------------------------------


class TestLaplaceGaussian:
    def test_output_shape(self, deltaf_series):
        result = smooth.laplace_gaussian(deltaf_series)
        assert result.shape == deltaf_series.shape

    def test_matches_scipy_reference(self, deltaf_series):
        result = smooth.laplace_gaussian(deltaf_series, sigma=2)
        expected = ndi.gaussian_laplace(deltaf_series, sigma=(0, 2, 2))
        np.testing.assert_allclose(result, expected)

    def test_default_sigma_matches_scipy_reference(self, deltaf_series):
        result = smooth.laplace_gaussian(deltaf_series)
        expected = ndi.gaussian_laplace(deltaf_series, sigma=(0, 2, 2))
        np.testing.assert_allclose(result, expected)

    def test_custom_sigma_changes_output(self, deltaf_series):
        default = smooth.laplace_gaussian(deltaf_series)
        custom = smooth.laplace_gaussian(deltaf_series, sigma=4)
        assert not np.allclose(default, custom)

    def test_passes_kwargs_to_scipy(self, deltaf_series):
        result = smooth.laplace_gaussian(deltaf_series, sigma=2, mode="nearest")
        expected = ndi.gaussian_laplace(deltaf_series, sigma=(0, 2, 2), mode="nearest")
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# zscore.zscore_deltaf
# ---------------------------------------------------------------------------


class TestZscoreDeltaf:
    def test_output_shape(self, deltaf_series):
        result = zscore.zscore_deltaf(deltaf_series)
        assert result.shape == deltaf_series.shape

    def test_returns_ndarray(self, deltaf_series):
        result = zscore.zscore_deltaf(deltaf_series)
        assert isinstance(result, np.ndarray)

    def test_matches_scipy_reference(self, deltaf_series):
        result = zscore.zscore_deltaf(deltaf_series)
        expected = sst.zscore(deltaf_series, axis=0)
        np.testing.assert_allclose(result, expected)

    def test_zero_mean_along_time_axis(self, deltaf_series):
        result = zscore.zscore_deltaf(deltaf_series)
        np.testing.assert_allclose(result.mean(axis=0), 0, atol=1e-10)

    def test_unit_std_along_time_axis(self, deltaf_series):
        result = zscore.zscore_deltaf(deltaf_series)
        np.testing.assert_allclose(result.std(axis=0), 1, atol=1e-10)


# ---------------------------------------------------------------------------
# regression.ridge_regression
# ---------------------------------------------------------------------------


class TestRidgeRegression:
    def test_output_shapes(self, linear_regression_data):
        deltaf_series, regressors, _, pixel_shape = linear_regression_data
        coefs, r2, mse = regr.ridge_regression(deltaf_series, regressors)
        assert coefs.shape == (regressors.shape[1], *pixel_shape)
        assert r2.shape == pixel_shape
        assert mse.shape == pixel_shape

    def test_recovers_known_coefficients(self, linear_regression_data):
        deltaf_series, regressors, true_coefs, _ = linear_regression_data
        coefs, _, _ = regr.ridge_regression(deltaf_series, regressors)
        np.testing.assert_allclose(coefs.reshape(regressors.shape[1], -1), true_coefs, atol=0.1)

    def test_high_r2_for_noiseless_linear_data(self, linear_regression_data):
        deltaf_series, regressors, _, _ = linear_regression_data
        _, r2, _ = regr.ridge_regression(deltaf_series, regressors)
        assert np.all(r2 > 0.99)

    def test_low_mse_for_noiseless_linear_data(self, linear_regression_data):
        deltaf_series, regressors, _, _ = linear_regression_data
        _, _, mse = regr.ridge_regression(deltaf_series, regressors)
        assert np.all(mse < 0.1)

    def test_handles_nan_values(self, linear_regression_data):
        deltaf_series, regressors, _, _ = linear_regression_data
        deltaf_with_nan = deltaf_series.copy()
        deltaf_with_nan[0, 0, 0] = np.nan
        coefs, r2, mse = regr.ridge_regression(deltaf_with_nan, regressors)
        assert np.isfinite(coefs).all()
        assert np.isfinite(r2).all()
        assert np.isfinite(mse).all()


# ---------------------------------------------------------------------------
# regression.ridge_regression_fast
# ---------------------------------------------------------------------------


class TestRidgeRegressionFast:
    def test_output_shapes(self, linear_regression_data):
        deltaf_series, regressors, _, pixel_shape = linear_regression_data
        coefs, r2, mse = regr.ridge_regression_fast(deltaf_series, regressors)
        assert coefs.shape == (regressors.shape[1], *pixel_shape)
        assert r2.shape == pixel_shape
        assert mse.shape == pixel_shape

    def test_recovers_known_coefficients(self, linear_regression_data):
        deltaf_series, regressors, true_coefs, _ = linear_regression_data
        coefs, _, _ = regr.ridge_regression_fast(deltaf_series, regressors, alpha=1e-6)
        np.testing.assert_allclose(coefs.reshape(regressors.shape[1], -1), true_coefs, atol=0.1)

    def test_matches_naive_implementation(self, linear_regression_data):
        """The vectorised implementation should closely match the per-pixel sklearn implementation."""
        deltaf_series, regressors, _, _ = linear_regression_data
        coefs_naive, r2_naive, mse_naive = regr.ridge_regression(deltaf_series, regressors)
        coefs_fast, r2_fast, mse_fast = regr.ridge_regression_fast(deltaf_series, regressors, alpha=1.0)

        np.testing.assert_allclose(coefs_fast, coefs_naive, atol=1e-2)
        np.testing.assert_allclose(r2_fast, r2_naive, atol=1e-2)
        np.testing.assert_allclose(mse_fast, mse_naive, atol=1e-2)

    def test_alpha_shrinks_coefficients(self, linear_regression_data):
        deltaf_series, regressors, _, _ = linear_regression_data
        coefs_low_alpha, _, _ = regr.ridge_regression_fast(deltaf_series, regressors, alpha=1e-6)
        coefs_high_alpha, _, _ = regr.ridge_regression_fast(deltaf_series, regressors, alpha=1e4)
        assert np.abs(coefs_high_alpha).mean() < np.abs(coefs_low_alpha).mean()

    def test_handles_nan_values(self, linear_regression_data):
        deltaf_series, regressors, _, _ = linear_regression_data
        deltaf_with_nan = deltaf_series.copy()
        deltaf_with_nan[0, 0, 0] = np.nan
        coefs, r2, mse = regr.ridge_regression_fast(deltaf_with_nan, regressors)
        assert np.isfinite(coefs).all()
        assert np.isfinite(r2).all()
        assert np.isfinite(mse).all()

    def test_zero_variance_target_gives_perfect_r2(self):
        """A pixel with a constant (zero-variance) signal is fit exactly by centring, giving r2 == 1."""
        rng = np.random.default_rng(0)
        regressors = rng.normal(size=(50, 2))
        deltaf_series = np.full((50, 1, 1), 3.0)

        _, r2, mse = regr.ridge_regression_fast(deltaf_series, regressors)

        assert r2[0, 0] == 1.0
        assert mse[0, 0] == pytest.approx(0.0, abs=1e-8)

    def test_emits_no_floating_point_warnings(self, linear_regression_data):
        """BLAS raises spurious FP flags on matmul; they must not reach the caller."""
        deltaf_series, regressors, _, _ = linear_regression_data

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            regr.ridge_regression_fast(deltaf_series, regressors)

        assert [str(w.message) for w in caught if "encountered" in str(w.message)] == []


# ---------------------------------------------------------------------------
# regression.elapsed_seconds
# ---------------------------------------------------------------------------


class TestElapsedSeconds:
    def test_numeric_timestamps_normalized_to_start_at_zero(self):
        timestamps = np.array([5.0, 6.0, 8.0])
        result = regr.elapsed_seconds(timestamps)
        np.testing.assert_allclose(result, [0.0, 1.0, 3.0])

    def test_already_zero_started_numeric_timestamps_unchanged(self):
        timestamps = np.array([0.0, 0.5, 1.0])
        result = regr.elapsed_seconds(timestamps)
        np.testing.assert_allclose(result, timestamps)

    def test_parses_iso_byte_string_timestamps(self):
        """Matches the /timestamps format written by the real preprocessing pipeline (dtype 'S25')."""
        timestamps = np.array(
            [b"2024-01-01T14:00:00.000", b"2024-01-01T14:00:00.500", b"2024-01-01T14:00:01.000"], dtype="S25"
        )
        result = regr.elapsed_seconds(timestamps)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_parses_iso_str_timestamps(self):
        timestamps = np.array(["2024-01-01T14:00:00.000", "2024-01-01T14:00:00.500"])
        result = regr.elapsed_seconds(timestamps)
        np.testing.assert_allclose(result, [0.0, 0.5])


# ---------------------------------------------------------------------------
# regression.interpolate_regressors
# ---------------------------------------------------------------------------


class TestInterpolateRegressors:
    def test_output_shape(self):
        regressors = np.arange(20).reshape(10, 2).astype(float)
        source_timestamps = np.linspace(0, 1, 10)
        target_timestamps = np.linspace(0, 1, 25)
        result = regr.interpolate_regressors(regressors, source_timestamps, target_timestamps)
        assert result.shape == (25, 2)

    def test_matches_linear_interpolation(self):
        source_timestamps = np.array([0.0, 1.0, 2.0, 3.0])
        regressors = np.array([[0.0], [10.0], [20.0], [30.0]])
        target_timestamps = np.array([0.5, 1.5, 2.5])
        result = regr.interpolate_regressors(regressors, source_timestamps, target_timestamps)
        np.testing.assert_allclose(result, [[5.0], [15.0], [25.0]])

    def test_clamps_out_of_range_targets(self):
        source_timestamps = np.array([1.0, 2.0, 3.0])
        regressors = np.array([[10.0], [20.0], [30.0]])
        target_timestamps = np.array([-5.0, 10.0])
        result = regr.interpolate_regressors(regressors, source_timestamps, target_timestamps)
        np.testing.assert_allclose(result, [[10.0], [30.0]])


# ---------------------------------------------------------------------------
# regression.append_nuisance_regressors
# ---------------------------------------------------------------------------


class TestAppendNuisanceRegressors:
    def test_output_shape_and_labels(self):
        regressors = np.random.default_rng(0).normal(size=(20, 2))
        nuisance = np.random.default_rng(1).normal(size=(20, 3))
        combined, labels = regr.append_nuisance_regressors(regressors, ["a", "b"], nuisance, ["n1", "n2", "n3"])
        assert combined.shape == (20, 5)
        assert labels == ["a", "b", "n1", "n2", "n3"]

    def test_zscores_nuisance_columns_by_default(self):
        regressors = np.zeros((20, 1))
        nuisance = np.random.default_rng(0).normal(loc=100, scale=10, size=(20, 2))
        combined, _ = regr.append_nuisance_regressors(regressors, ["a"], nuisance, ["n1", "n2"])
        appended = combined[:, 1:]
        np.testing.assert_allclose(appended.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(appended.std(axis=0), 1, atol=1e-10)

    def test_zscore_false_leaves_nuisance_raw(self):
        regressors = np.zeros((20, 1))
        nuisance = np.random.default_rng(0).normal(loc=100, scale=10, size=(20, 2))
        combined, _ = regr.append_nuisance_regressors(regressors, ["a"], nuisance, ["n1", "n2"], zscore=False)
        np.testing.assert_allclose(combined[:, 1:], nuisance)

    def test_leaves_original_regressors_untouched(self):
        regressors = np.random.default_rng(0).normal(size=(20, 1))
        nuisance = np.random.default_rng(1).normal(loc=100, scale=10, size=(20, 1))
        combined, _ = regr.append_nuisance_regressors(regressors, ["a"], nuisance, ["n1"])
        np.testing.assert_allclose(combined[:, :1], regressors)

    def test_raises_on_sample_count_mismatch(self):
        regressors = np.zeros((20, 1))
        nuisance = np.zeros((10, 1))
        with pytest.raises(ValueError, match="samples"):
            regr.append_nuisance_regressors(regressors, ["a"], nuisance, ["n1"])

    def test_raises_on_duplicate_labels(self):
        regressors = np.zeros((20, 1))
        nuisance = np.zeros((20, 1))
        with pytest.raises(ValueError, match="duplicate"):
            regr.append_nuisance_regressors(regressors, ["a"], nuisance, ["a"])


# ---------------------------------------------------------------------------
# regression.check_alignment
# ---------------------------------------------------------------------------


ALIGNED = np.arange(300) * 0.04 + 2.0
RECORDING_ATTRS = {"session_start_time": "2024-01-01T14:00:00", "behaviour_session": "ses", "offset_s": 2.0}
REGRESSOR_ALIGNMENT = {
    "session_start_time": "2024-01-01T14:00:00",
    "behaviour_session": "ses",
    "timestamps": ALIGNED.copy(),
}


class TestCheckAlignment:
    def test_agreeing_inputs_give_no_warnings(self):
        assert regr.check_alignment((ALIGNED, RECORDING_ATTRS), REGRESSOR_ALIGNMENT) == []

    def test_recording_without_alignment_warns(self):
        warnings = regr.check_alignment(None, REGRESSOR_ALIGNMENT)
        assert len(warnings) == 1
        assert "positionally" in warnings[0]

    def test_regressors_without_alignment_warns(self):
        warnings = regr.check_alignment((ALIGNED, RECORDING_ATTRS), None)
        assert len(warnings) == 1
        assert "positionally" in warnings[0]

    def test_session_start_mismatch_raises(self):
        regressors = {**REGRESSOR_ALIGNMENT, "session_start_time": "2024-01-01T15:00:00"}
        with pytest.raises(ValueError, match="session start time"):
            regr.check_alignment((ALIGNED, RECORDING_ATTRS), regressors)

    def test_length_mismatch_raises(self):
        regressors = {**REGRESSOR_ALIGNMENT, "timestamps": ALIGNED[:-1]}
        with pytest.raises(ValueError, match="299 timestamps"):
            regr.check_alignment((ALIGNED, RECORDING_ATTRS), regressors)

    def test_missing_regressor_timestamps_warns(self):
        regressors = {**REGRESSOR_ALIGNMENT, "timestamps": None}
        warnings = regr.check_alignment((ALIGNED, RECORDING_ATTRS), regressors)
        assert len(warnings) == 1
        assert "no timestamps" in warnings[0]

    def test_timestamp_difference_above_tolerance_warns_with_value(self):
        regressors = {**REGRESSOR_ALIGNMENT, "timestamps": ALIGNED + 0.01}
        warnings = regr.check_alignment((ALIGNED, RECORDING_ATTRS), regressors)
        assert len(warnings) == 1
        assert "0.0100 s" in warnings[0]

    def test_timestamp_difference_within_tolerance_is_silent(self):
        regressors = {**REGRESSOR_ALIGNMENT, "timestamps": ALIGNED + 0.004}
        assert regr.check_alignment((ALIGNED, RECORDING_ATTRS), regressors) == []


# ---------------------------------------------------------------------------
# io.read_nuisance_regressors
# ---------------------------------------------------------------------------


class TestReadNuisanceRegressors:
    def test_reads_hdf5(self, nuisance_regressor_h5):
        regressors, labels, timestamps = io.read_nuisance_regressors(nuisance_regressor_h5)
        assert regressors.shape == (50, 2)
        assert labels == ["motion_a", "motion_b"]
        assert timestamps.shape == (50,)

    def test_reads_npz(self, nuisance_regressor_npz):
        regressors, labels, timestamps = io.read_nuisance_regressors(nuisance_regressor_npz)
        assert regressors.shape == (50, 2)
        assert labels == ["pupil_a", "pupil_b"]
        assert timestamps.shape == (50,)

    def test_raises_on_unsupported_format(self, tmp_path):
        path = tmp_path / "nuisance.txt"
        path.touch()
        with pytest.raises(ValueError, match="Unsupported"):
            io.read_nuisance_regressors(str(path))


class TestReadRegressors:
    def test_npz_without_alignment(self, regressor_npz):
        regressors, labels, trial_idx, alignment = io.read_regressors(regressor_npz)
        assert regressors.shape == (300, 3)
        assert list(labels) == ["reg_a", "reg_b", "reg_c"]
        assert trial_idx is None
        assert alignment is None

    def test_npz_with_alignment(self, aligned_regressor_npz):
        _, _, trial_idx, alignment = io.read_regressors(aligned_regressor_npz)
        assert trial_idx.shape == (300,)
        assert alignment is not None
        assert alignment["session_start_time"] == "2024-01-01T14:00:00"
        assert alignment["behaviour_session"] == "ses-aligned"
        assert alignment["timestamps"].shape == (300,)
        assert alignment["timestamps"].dtype == np.float64

    def test_h5_without_alignment(self, regressor_h5):
        regressors, labels, trial_idx, alignment = io.read_regressors(regressor_h5)
        assert regressors.shape == (300, 3)
        assert trial_idx.shape == (300,)
        assert alignment is None

    def test_h5_with_alignment(self, tmp_path):
        path = tmp_path / "regressors_aligned.h5"
        with h5.File(str(path), "w") as f:
            f.create_dataset("regressors", data=np.zeros((10, 2)))
            f.create_dataset("labels", data=np.array(["a", "b"], dtype="S"))
            f.create_dataset("timestamps", data=np.arange(10, dtype=float))
            f.create_dataset("session_start_time", data="2024-01-01T14:00:00")
            f.create_dataset("behaviour_session", data="ses")
        _, _, _, alignment = io.read_regressors(str(path))
        assert alignment == {
            "session_start_time": "2024-01-01T14:00:00",
            "behaviour_session": "ses",
            "timestamps": pytest.approx(np.arange(10, dtype=float)),
        }


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def test_smooth_cmd(preproc_h5, output_dir):
    runner = CliRunner()
    result = runner.invoke(mesoscopy.cli, args=f"process smooth {preproc_h5} -o {output_dir}")
    assert result.exit_code == 0

    outpath = pathlib.Path(output_dir) / "preproc_smoothed.h5"
    assert outpath.is_file()
    assert io.read_h5(str(outpath))["/F"].shape == (300, 40, 40)


def test_smooth_cmd_custom_sigma(preproc_h5, output_dir):
    runner = CliRunner()
    result = runner.invoke(mesoscopy.cli, args=f"process smooth {preproc_h5} -o {output_dir} -s 4")
    assert result.exit_code == 0

    outpath = pathlib.Path(output_dir) / "preproc_smoothed.h5"
    assert outpath.is_file()

    # The option must reach the filter: output should match sigma=4, not the default.
    _, deltaf_series, _ = io.load_deltaf(str(preproc_h5))
    expected = smooth.laplace_gaussian(deltaf_series, sigma=4)
    np.testing.assert_allclose(io.read_h5(str(outpath))["/F"][:], expected)
    assert not np.allclose(expected, smooth.laplace_gaussian(deltaf_series))


def test_zscore_cmd_h5(preproc_h5, output_dir):
    runner = CliRunner()
    result = runner.invoke(mesoscopy.cli, args=f"process zscore {preproc_h5} -o {output_dir}")
    assert result.exit_code == 0

    outpath = pathlib.Path(output_dir) / "preproc_zscored.h5"
    assert outpath.is_file()
    assert io.read_h5(str(outpath))["/F"].shape == (300, 40, 40)


def test_smooth_cmd_propagates_timestamps_aligned(aligned_preproc_h5, output_dir):
    result = CliRunner().invoke(mesoscopy.cli, args=f"process smooth {aligned_preproc_h5} -o {output_dir}")
    assert result.exit_code == 0, result.output

    source = io.read_timestamps_aligned(aligned_preproc_h5)
    copied = io.read_timestamps_aligned(str(pathlib.Path(output_dir) / "preproc_aligned_smoothed.h5"))
    assert copied is not None
    np.testing.assert_array_equal(copied[0], source[0])
    assert copied[1] == source[1]


def test_smooth_cmd_without_timestamps_aligned(preproc_h5, output_dir):
    result = CliRunner().invoke(mesoscopy.cli, args=f"process smooth {preproc_h5} -o {output_dir}")
    assert result.exit_code == 0, result.output
    assert io.read_timestamps_aligned(str(pathlib.Path(output_dir) / "preproc_smoothed.h5")) is None


def test_zscore_cmd_propagates_timestamps_aligned(aligned_preproc_h5, output_dir):
    result = CliRunner().invoke(mesoscopy.cli, args=f"process zscore {aligned_preproc_h5} -o {output_dir}")
    assert result.exit_code == 0, result.output

    source = io.read_timestamps_aligned(aligned_preproc_h5)
    copied = io.read_timestamps_aligned(str(pathlib.Path(output_dir) / "preproc_aligned_zscored.h5"))
    assert copied is not None
    np.testing.assert_array_equal(copied[0], source[0])
    assert copied[1] == source[1]


def test_zscore_cmd_without_timestamps_aligned(preproc_h5, output_dir):
    result = CliRunner().invoke(mesoscopy.cli, args=f"process zscore {preproc_h5} -o {output_dir}")
    assert result.exit_code == 0, result.output
    assert io.read_timestamps_aligned(str(pathlib.Path(output_dir) / "preproc_zscored.h5")) is None


def test_zscore_cmd_nwb(preproc_nwb, output_dir):
    runner = CliRunner()
    result = runner.invoke(mesoscopy.cli, args=f"process zscore {preproc_nwb} -o {output_dir}")
    assert result.exit_code == 0
    assert "Appending to NWB file..." in result.output

    outpath = pathlib.Path(output_dir) / "session_1234_zscored.h5"
    assert outpath.is_file()
    assert io.read_h5(str(outpath))["/F"].shape == (300, 40, 40)


def test_regions_cmd(preproc_h5_bytes_timestamps, output_dir, mock_left_aba, mock_right_aba, mock_annotations):
    runner = CliRunner()
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(mock_left_aba, mock_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=mock_annotations),
    ):
        result = runner.invoke(mesoscopy.cli, args=f"process regions {preproc_h5_bytes_timestamps} -o {output_dir}")
    assert result.exit_code == 0

    outpath = pathlib.Path(output_dir) / "preproc_bytes_regions.csv"
    assert outpath.is_file()

    region_activity = pd.read_csv(outpath)
    assert set(region_activity["region"]) == {"L_REG1", "R_REG1"}


def test_regions_cmd_adds_time_aligned_column(aligned_preproc_h5_bytes_timestamps, output_dir, mask_npy):
    result = CliRunner().invoke(
        mesoscopy.cli, args=f"process regions {aligned_preproc_h5_bytes_timestamps} -o {output_dir} -m {mask_npy}"
    )
    assert result.exit_code == 0, result.output

    region_activity = pd.read_csv(pathlib.Path(output_dir) / "preproc_bytes_aligned_regions.csv")
    columns = list(region_activity.columns)
    assert columns.index("time_aligned") == columns.index("timestamp") + 1
    assert region_activity["time_aligned"].dtype == np.float64

    # Each row's aligned time matches its frame, whichever region it belongs to.
    session_start = datetime(2024, 1, 1, 14, 0, 0)
    elapsed = [(datetime.fromisoformat(ts) - session_start).total_seconds() for ts in region_activity["timestamp"]]
    np.testing.assert_allclose(region_activity["time_aligned"], np.array(elapsed) + 2.0, atol=1e-9)


def test_regions_cmd_without_timestamps_aligned_has_no_column(preproc_h5_bytes_timestamps, output_dir, mask_npy):
    result = CliRunner().invoke(
        mesoscopy.cli, args=f"process regions {preproc_h5_bytes_timestamps} -o {output_dir} -m {mask_npy}"
    )
    assert result.exit_code == 0, result.output

    region_activity = pd.read_csv(pathlib.Path(output_dir) / "preproc_bytes_regions.csv")
    assert "time_aligned" not in region_activity.columns


def test_regions_cmd_mask_only_skips_aba(preproc_h5_bytes_timestamps, output_dir, mask_npy):
    """A supplied mask replaces the ABA extraction, so no atlas is needed at all."""
    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli, args=f"process regions {preproc_h5_bytes_timestamps} -o {output_dir} -m {mask_npy}"
    )
    assert result.exit_code == 0
    assert "Loaded 1 region mask(s): roi" in result.output

    region_activity = pd.read_csv(pathlib.Path(output_dir) / "preproc_bytes_regions.csv")
    assert set(region_activity["region"]) == {"roi"}
    assert len(region_activity) == 300


def test_regions_cmd_mask_values_match_recording(preproc_h5_bytes_timestamps, output_dir, mask_npy):
    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli, args=f"process regions {preproc_h5_bytes_timestamps} -o {output_dir} -m {mask_npy}"
    )
    assert result.exit_code == 0

    with h5.File(preproc_h5_bytes_timestamps, "r") as f:
        expected = f["/F"][:][:, :, :20].mean(axis=(1, 2))

    region_activity = pd.read_csv(pathlib.Path(output_dir) / "preproc_bytes_regions.csv")
    np.testing.assert_allclose(region_activity["F"].to_numpy(), expected)


def test_regions_cmd_labelled_mask_gives_one_region_per_label(
    preproc_h5_bytes_timestamps, output_dir, labelled_mask_npy
):
    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli, args=f"process regions {preproc_h5_bytes_timestamps} -o {output_dir} -m {labelled_mask_npy}"
    )
    assert result.exit_code == 0

    region_activity = pd.read_csv(pathlib.Path(output_dir) / "preproc_bytes_regions.csv")
    assert set(region_activity["region"]) == {"labelled_1", "labelled_2"}


def test_regions_cmd_multiple_masks(preproc_h5_bytes_timestamps, output_dir, mask_npy, labelled_mask_npy):
    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli,
        args=f"process regions {preproc_h5_bytes_timestamps} -o {output_dir} -m {mask_npy} -m {labelled_mask_npy}",
    )
    assert result.exit_code == 0

    region_activity = pd.read_csv(pathlib.Path(output_dir) / "preproc_bytes_regions.csv")
    assert set(region_activity["region"]) == {"roi", "labelled_1", "labelled_2"}


def test_regions_cmd_include_aba_with_mask(
    preproc_h5_bytes_timestamps, output_dir, mask_npy, mock_left_aba, mock_right_aba, mock_annotations
):
    runner = CliRunner()
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(mock_left_aba, mock_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=mock_annotations),
    ):
        result = runner.invoke(
            mesoscopy.cli,
            args=f"process regions {preproc_h5_bytes_timestamps} -o {output_dir} -m {mask_npy} --include-aba",
        )
    assert result.exit_code == 0

    region_activity = pd.read_csv(pathlib.Path(output_dir) / "preproc_bytes_regions.csv")
    assert set(region_activity["region"]) == {"L_REG1", "R_REG1", "roi"}


def test_regions_cmd_duplicate_mask_names_error(preproc_h5_bytes_timestamps, output_dir, mask_npy, tmp_path):
    # A second file with the same stem resolves to the same region name.
    duplicate = tmp_path / "roi.npy"
    np.save(duplicate, np.ones((40, 40), dtype=bool))

    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli,
        args=f"process regions {preproc_h5_bytes_timestamps} -o {output_dir} -m {mask_npy} -m {duplicate}",
    )
    assert result.exit_code != 0
    assert "already defined by an earlier mask file" in result.output
    assert not (pathlib.Path(output_dir) / "preproc_bytes_regions.csv").is_file()


def test_regions_cmd_mask_shape_mismatch_errors(preproc_h5_bytes_timestamps, output_dir, tmp_path):
    path = tmp_path / "small.npy"
    np.save(path, np.ones((20, 20), dtype=bool))

    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli, args=f"process regions {preproc_h5_bytes_timestamps} -o {output_dir} -m {path}"
    )
    assert result.exit_code != 0
    assert "does not match the recording frame shape" in result.output


def test_regions_cmd_unreadable_mask_errors(preproc_h5_bytes_timestamps, output_dir, tmp_path):
    path = tmp_path / "roi.csv"
    path.write_text("not,a,mask\n")

    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli, args=f"process regions {preproc_h5_bytes_timestamps} -o {output_dir} -m {path}"
    )
    assert result.exit_code != 0
    assert "Could not read region masks" in result.output


def test_regression_cmd_npz(preproc_h5, regressor_npz, output_dir):
    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli,
        args=f"process regression {preproc_h5} {regressor_npz} -o {output_dir} --fast",
    )
    assert result.exit_code == 0

    outpath = pathlib.Path(output_dir) / "preproc_regression.npz"
    assert outpath.is_file()

    with np.load(outpath) as f:
        assert f["coefficients"].shape == (3, 40, 40)
        assert f["r2"].shape == (40, 40)
        assert f["mse"].shape == (40, 40)


def test_regression_cmd_h5(preproc_h5, regressor_h5, output_dir):
    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli,
        args=f"process regression {preproc_h5} {regressor_h5} -o {output_dir} --fast --h5",
    )
    assert result.exit_code == 0

    outpath = pathlib.Path(output_dir) / "preproc_regression.h5"
    assert outpath.is_file()

    result_h5 = io.read_h5(str(outpath))
    assert result_h5["/coefficients"].shape == (3, 40, 40)
    assert result_h5["/trial_idx"].shape == (300,)


def test_regression_cmd_alpha(preproc_h5, regressor_npz, output_dir):
    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli,
        args=f"process regression {preproc_h5} {regressor_npz} -o {output_dir} --fast -a 5.0",
    )
    assert result.exit_code == 0


def test_regression_cmd_not_fast(preproc_h5, regressor_npz, output_dir):
    """Check the (slower) naive per-pixel implementation is reachable from the CLI."""
    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli,
        args=f"process regression {preproc_h5} {regressor_npz} -o {output_dir}",
    )
    assert result.exit_code == 0

    outpath = pathlib.Path(output_dir) / "preproc_regression.npz"
    assert outpath.is_file()


def test_regression_cmd_aligned_copies_alignment_to_npz(aligned_preproc_h5, aligned_regressor_npz, output_dir):
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=f"process regression {aligned_preproc_h5} {aligned_regressor_npz} -o {output_dir} --fast",
    )
    assert result.exit_code == 0, result.output
    assert "WARNING" not in result.output

    with np.load(pathlib.Path(output_dir) / "preproc_aligned_regression.npz") as f:
        assert str(f["session_start_time"]) == "2024-01-01T14:00:00"
        assert str(f["behaviour_session"]) == "ses-aligned"
        assert f["trial_idx"].shape == (300,)


def test_regression_cmd_aligned_copies_alignment_to_h5(aligned_preproc_h5, aligned_regressor_npz, output_dir):
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=f"process regression {aligned_preproc_h5} {aligned_regressor_npz} -o {output_dir} --fast --h5",
    )
    assert result.exit_code == 0, result.output

    with h5.File(pathlib.Path(output_dir) / "preproc_aligned_regression.h5", "r") as f:
        assert f.attrs["session_start_time"] == "2024-01-01T14:00:00"
        assert f.attrs["behaviour_session"] == "ses-aligned"


def test_regression_cmd_unaligned_recording_warns_and_continues(preproc_h5, aligned_regressor_npz, output_dir):
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=f"process regression {preproc_h5} {aligned_regressor_npz} -o {output_dir} --fast",
    )
    assert result.exit_code == 0, result.output
    assert "WARNING" in result.output
    assert "positionally" in result.output

    with np.load(pathlib.Path(output_dir) / "preproc_regression.npz") as f:
        assert "session_start_time" not in f
        assert f["trial_idx"].shape == (300,)


def test_regression_cmd_unaligned_regressors_warns_and_continues(aligned_preproc_h5, regressor_npz, output_dir):
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=f"process regression {aligned_preproc_h5} {regressor_npz} -o {output_dir} --fast",
    )
    assert result.exit_code == 0, result.output
    assert "positionally" in result.output

    with np.load(pathlib.Path(output_dir) / "preproc_aligned_regression.npz") as f:
        assert "session_start_time" not in f


def test_regression_cmd_session_start_mismatch_fails(aligned_preproc_h5, tmp_path, output_dir):
    regressor_path = tmp_path / "regressors.npz"
    np.savez(
        regressor_path,
        regressors=np.zeros((300, 2)),
        labels=np.array(["a", "b"]),
        timestamps=np.arange(300) * 0.04 + 2.0,
        session_start_time="2024-01-01T15:00:00",
        behaviour_session="ses-aligned",
    )
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=f"process regression {aligned_preproc_h5} {regressor_path} -o {output_dir} --fast",
    )
    assert result.exit_code != 0
    assert "session start time" in result.output
    assert not (pathlib.Path(output_dir) / "preproc_aligned_regression.npz").is_file()


def test_regression_cmd_length_mismatch_fails(aligned_preproc_h5, tmp_path, output_dir):
    regressor_path = tmp_path / "regressors.npz"
    np.savez(
        regressor_path,
        regressors=np.zeros((299, 2)),
        labels=np.array(["a", "b"]),
        timestamps=np.arange(299) * 0.04 + 2.0,
        session_start_time="2024-01-01T14:00:00",
        behaviour_session="ses-aligned",
    )
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=f"process regression {aligned_preproc_h5} {regressor_path} -o {output_dir} --fast",
    )
    assert result.exit_code != 0
    assert "299 timestamps" in result.output


def test_regression_cmd_timestamp_drift_warns(aligned_preproc_h5, tmp_path, output_dir):
    regressor_path = tmp_path / "regressors.npz"
    np.savez(
        regressor_path,
        regressors=np.random.default_rng(6).normal(size=(300, 2)),
        labels=np.array(["a", "b"]),
        timestamps=np.arange(300) * 0.04 + 2.02,
        session_start_time="2024-01-01T14:00:00",
        behaviour_session="ses-aligned",
    )
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=f"process regression {aligned_preproc_h5} {regressor_path} -o {output_dir} --fast",
    )
    assert result.exit_code == 0, result.output
    assert "differ by up to 0.0200 s" in result.output


def test_regression_cmd_with_nuisance_regressors(preproc_h5, regressor_npz, nuisance_regressor_h5, output_dir):
    """Nuisance regressors from an external file should be interpolated, z-scored, and appended."""
    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli,
        args=f"process regression {preproc_h5} {regressor_npz} -o {output_dir} --fast -n {nuisance_regressor_h5}",
    )
    assert result.exit_code == 0
    assert "Added nuisance regressors: ['motion_a', 'motion_b']" in result.output

    outpath = pathlib.Path(output_dir) / "preproc_regression.npz"
    assert outpath.is_file()

    with np.load(outpath, allow_pickle=True) as f:
        # 3 task regressors (reg_a/b/c) + 2 nuisance regressors (motion_a/b).
        assert f["coefficients"].shape == (5, 40, 40)
        assert list(f["labels"])[-2:] == ["motion_a", "motion_b"]


def test_regression_cmd_with_multiple_nuisance_regressor_files(
    preproc_h5, regressor_npz, nuisance_regressor_h5, nuisance_regressor_npz, output_dir
):
    """Passing -n multiple times should combine nuisance regressors from every file."""
    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli,
        args=(
            f"process regression {preproc_h5} {regressor_npz} -o {output_dir} --fast"
            f" -n {nuisance_regressor_h5} -n {nuisance_regressor_npz}"
        ),
    )
    assert result.exit_code == 0

    outpath = pathlib.Path(output_dir) / "preproc_regression.npz"
    with np.load(outpath, allow_pickle=True) as f:
        # 3 task regressors + 2 nuisance regressors from each of the two files.
        assert f["coefficients"].shape == (7, 40, 40)


def test_regression_cmd_with_nuisance_regressors_and_trial_idx(
    preproc_h5, regressor_h5, nuisance_regressor_h5, output_dir
):
    """Nuisance regressors should be interpolated onto the full recording, then subset by trial_idx to match the
    (already trial_idx-subset) task regressor matrix."""
    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli,
        args=f"process regression {preproc_h5} {regressor_h5} -o {output_dir} --fast --h5 -n {nuisance_regressor_h5}",
    )
    assert result.exit_code == 0

    outpath = pathlib.Path(output_dir) / "preproc_regression.h5"
    result_h5 = io.read_h5(str(outpath))
    assert result_h5["/coefficients"].shape == (5, 40, 40)
    assert result_h5["/trial_idx"].shape == (300,)


def test_regression_cmd_with_nuisance_regressors_iso_timestamps(
    preproc_h5_bytes_timestamps, regressor_npz, nuisance_regressor_h5, output_dir
):
    """Real preprocessed recordings store /timestamps as ISO-8601 byte strings, not the plain floats used by the
    `preproc_h5` fixture -- make sure nuisance regressor alignment handles that format too."""
    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli,
        args=(
            f"process regression {preproc_h5_bytes_timestamps} {regressor_npz} -o {output_dir} --fast"
            f" -n {nuisance_regressor_h5}"
        ),
    )
    assert result.exit_code == 0

    outpath = pathlib.Path(output_dir) / "preproc_bytes_regression.npz"
    assert outpath.is_file()

    with np.load(outpath, allow_pickle=True) as f:
        assert f["coefficients"].shape == (5, 40, 40)


# ---------------------------------------------------------------------------
# region.extract_region_activity / region.extract_all_regions fixtures
# ---------------------------------------------------------------------------

# Mock atlas layout (6×6):
#   Columns 0-2 are the "left" hemisphere; columns 3-5 are zero.
#   right_aba = np.flip(left_aba, axis=1), so regions appear in columns 3-5
#   of the right atlas, mirroring the left — no pixel is non-zero in both.
#
#   left_aba rows:
#     0-1, col 0-2 → region id=1 (REG1)
#     2-3, col 0-2 → region id=2 (REG2)
#     4-5, col 0-2 → region id=3 (FRP1, present in DEFAULT_EXCLUDE)

_ATLAS_H, _ATLAS_W = 6, 6
_N_FRAMES = 10
# Long enough for extract_all_regions to process it as more than one frame block.
_LONG_N_FRAMES = 2500


@pytest.fixture(scope="module")
def region_left_aba():
    aba = np.zeros((_ATLAS_H, _ATLAS_W), dtype=np.uint16)
    aba[0:2, 0:3] = 1
    aba[2:4, 0:3] = 2
    aba[4:6, 0:3] = 3
    return aba


@pytest.fixture(scope="module")
def region_right_aba(region_left_aba):
    return np.flip(region_left_aba, axis=1).copy()


@pytest.fixture(scope="module")
def region_annotations():
    return pd.DataFrame({"id": [1, 2, 3], "acronym": ["REG1", "REG2", "FRP1"]})


@pytest.fixture(scope="module")
def region_annotations_with_empty(region_annotations):
    """Annotations with a region that has no pixels in the mock atlas, as ORBm1 has none in the real one."""
    return pd.concat([region_annotations, pd.DataFrame({"id": [4], "acronym": ["EMPTY"]})], ignore_index=True)


@pytest.fixture(scope="module")
def region_deltaf_series():
    rng = np.random.default_rng(0)
    return rng.random((_N_FRAMES, _ATLAS_H, _ATLAS_W))


# ---------------------------------------------------------------------------
# region.extract_region_activity
# ---------------------------------------------------------------------------


def test_extract_region_activity_left_hemisphere_shape(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        result = extract_region_activity(region_deltaf_series, "REG1", "left")
    assert result.shape == (_N_FRAMES,)


def test_extract_region_activity_right_hemisphere_shape(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        result = extract_region_activity(region_deltaf_series, "REG1", "right")
    assert result.shape == (_N_FRAMES,)


def test_extract_region_activity_both_hemispheres_shape(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        result = extract_region_activity(region_deltaf_series, "REG1", "both")
    assert result.shape == (_N_FRAMES,)


def test_extract_region_activity_left_hemisphere_values(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    # Region 1 occupies rows 0-1, cols 0-2 in the left atlas.
    expected = region_deltaf_series[:, 0:2, 0:3].mean(axis=(1, 2))
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        result = extract_region_activity(region_deltaf_series, "REG1", "left")
    np.testing.assert_allclose(result, expected)


def test_extract_region_activity_right_hemisphere_values(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    # right_aba = flip(left_aba): region 1 lands in rows 0-1, cols 3-5.
    expected = region_deltaf_series[:, 0:2, 3:6].mean(axis=(1, 2))
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        result = extract_region_activity(region_deltaf_series, "REG1", "right")
    np.testing.assert_allclose(result, expected)


def test_extract_region_activity_both_hemispheres_values(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    # left + right: region 1 covers rows 0-1 across all 6 columns.
    expected = region_deltaf_series[:, 0:2, :].mean(axis=(1, 2))
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        result = extract_region_activity(region_deltaf_series, "REG1", "both")
    np.testing.assert_allclose(result, expected)


def test_extract_region_activity_different_regions_produce_different_results(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        reg1 = extract_region_activity(region_deltaf_series, "REG1", "left")
        reg2 = extract_region_activity(region_deltaf_series, "REG2", "left")
    assert not np.allclose(reg1, reg2), "Different regions should produce different activity signals"


def test_extract_region_activity_invalid_hemisphere_raises(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        with pytest.raises(ValueError, match="hemisphere"):
            extract_region_activity(region_deltaf_series, "REG1", "bilateral")


def test_extract_region_activity_unknown_acronym_raises(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    """An unrecognised acronym is a ValueError naming it, not an IndexError from the id lookup."""
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        with pytest.raises(ValueError, match="REG3 not recognised"):
            extract_region_activity(region_deltaf_series, "REG3", "left")


def test_extract_region_activity_unknown_acronym_suggests_close_matches(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        with pytest.raises(ValueError, match="Closest matches") as excinfo:
            extract_region_activity(region_deltaf_series, "REG3", "left")
    assert "REG1" in str(excinfo.value)


def test_extract_region_activity_close_match_suggestion_ignores_case(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    """Lookup stays case-sensitive, but the suggestion ignores case."""
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        with pytest.raises(ValueError, match="Closest matches") as excinfo:
            extract_region_activity(region_deltaf_series, "reg1", "left")
    assert "REG1" in str(excinfo.value)


def test_extract_region_activity_unknown_acronym_without_close_match(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    """Nothing similar in the atlas means no misleading suggestion."""
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        with pytest.raises(ValueError, match="not recognised") as excinfo:
            extract_region_activity(region_deltaf_series, "zzzz", "left")
    assert "Closest matches" not in str(excinfo.value)


def test_extract_region_activity_hemisphere_argument_case_insensitive(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        lower = extract_region_activity(region_deltaf_series, "REG1", "left")
        upper = extract_region_activity(region_deltaf_series, "REG1", "LEFT")
    np.testing.assert_allclose(lower, upper)


def test_extract_region_activity_returns_plain_ndarray(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        result = extract_region_activity(region_deltaf_series, "REG1", "left")
    assert not isinstance(result, np.ma.MaskedArray)


def test_extract_region_activity_ignores_nan_pixels(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    series = region_deltaf_series.copy()
    series[:, 0, 0] = np.nan
    expected = np.nanmean(series[:, 0:2, 0:3].reshape(_N_FRAMES, -1), axis=1)

    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        result = extract_region_activity(series, "REG1", "left")

    # A MaskedArray result swallows both assertions below, so check the type before comparing values.
    assert not isinstance(result, np.ma.MaskedArray)
    assert not np.isnan(result).any()
    np.testing.assert_allclose(result, expected)


def test_extract_region_activity_all_nan_frame_is_nan(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    series = region_deltaf_series.copy()
    series[0, 0:2, 0:3] = np.nan

    with (
        warnings.catch_warnings(),
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        result = extract_region_activity(series, "REG1", "left")

    assert np.isnan(result[0])
    assert not np.isnan(result[1:]).any()


def test_extract_region_activity_empty_region_is_nan(
    region_left_aba, region_right_aba, region_annotations_with_empty, region_deltaf_series
):
    """A region with no pixels in the atlas gives NaN rather than raising."""
    with (
        warnings.catch_warnings(),
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations_with_empty),
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        result = extract_region_activity(region_deltaf_series, "EMPTY", "left")

    assert result.shape == (_N_FRAMES,)
    assert np.isnan(result).all()


# ---------------------------------------------------------------------------
# region.extract_all_regions
# ---------------------------------------------------------------------------


def test_extract_all_regions_returns_dict(region_left_aba, region_right_aba, region_annotations, region_deltaf_series):
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        result = extract_all_regions(region_deltaf_series)
    assert isinstance(result, dict)


def test_extract_all_regions_keys_have_hemisphere_prefix(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        result = extract_all_regions(region_deltaf_series)
    for key in result:
        assert key.startswith("L_") or key.startswith("R_"), f"Unexpected key format: {key}"


def test_extract_all_regions_both_hemispheres_present_for_each_region(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        result = extract_all_regions(region_deltaf_series)
    for region in ("REG1", "REG2"):
        assert f"L_{region}" in result
        assert f"R_{region}" in result


def test_extract_all_regions_values_have_correct_shape(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        result = extract_all_regions(region_deltaf_series)
    for key, value in result.items():
        assert value.shape == (_N_FRAMES,), f"Wrong shape for {key}: {value.shape}"


def test_extract_all_regions_default_exclude_filters_regions(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    # FRP1 is in DEFAULT_EXCLUDE; it should be absent from the result.
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        result = extract_all_regions(region_deltaf_series)
    assert "L_FRP1" not in result
    assert "R_FRP1" not in result


def test_extract_all_regions_ignore_default_exclude_includes_all_regions(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        result = extract_all_regions(region_deltaf_series, ignore_default_exclude=True)
    assert "L_FRP1" in result
    assert "R_FRP1" in result


def test_extract_all_regions_custom_exclude_removes_region(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        result = extract_all_regions(region_deltaf_series, exclude=["REG1"])
    assert "L_REG1" not in result
    assert "R_REG1" not in result
    assert "L_REG2" in result
    assert "R_REG2" in result


def test_extract_all_regions_custom_exclude_combined_with_default(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        result = extract_all_regions(region_deltaf_series, exclude=["REG1"])
    assert "L_FRP1" not in result
    assert "L_REG1" not in result
    assert "L_REG2" in result


def test_extract_all_regions_does_not_mutate_default_exclude(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    original = list(DEFAULT_EXCLUDE)
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        extract_all_regions(region_deltaf_series, exclude=["REG1"])
    assert original == DEFAULT_EXCLUDE, "extract_all_regions must not mutate DEFAULT_EXCLUDE"


def test_extract_all_regions_values_match_extract_region_activity(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    """extract_all_regions results should match extract_region_activity for each region."""
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        all_regions = extract_all_regions(region_deltaf_series, ignore_default_exclude=True)
        single_left = extract_region_activity(region_deltaf_series, "REG2", "left")
        single_right = extract_region_activity(region_deltaf_series, "REG2", "right")
    np.testing.assert_allclose(all_regions["L_REG2"], single_left)
    np.testing.assert_allclose(all_regions["R_REG2"], single_right)


def test_extract_all_regions_values_match_direct_mean(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    """Region 1 occupies rows 0-1, cols 0-2 in the left atlas."""
    expected = region_deltaf_series[:, 0:2, 0:3].mean(axis=(1, 2))
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        result = extract_all_regions(region_deltaf_series, ignore_default_exclude=True)
    np.testing.assert_allclose(result["L_REG1"], expected)


def test_extract_all_regions_ignores_nan_pixels(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    series = region_deltaf_series.copy()
    series[:, 0, 0] = np.nan
    expected = np.nanmean(series[:, 0:2, 0:3].reshape(_N_FRAMES, -1), axis=1)

    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        result = extract_all_regions(series, ignore_default_exclude=True)

    assert not np.isnan(result["L_REG1"]).any()
    np.testing.assert_allclose(result["L_REG1"], expected)


def test_extract_all_regions_nan_does_not_leak_into_other_regions(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    """Pixel (0, 0) belongs to left REG1 only, so no other region's trace may change."""
    series = region_deltaf_series.copy()
    series[:, 0, 0] = np.nan

    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        clean = extract_all_regions(region_deltaf_series, ignore_default_exclude=True)
        result = extract_all_regions(series, ignore_default_exclude=True)

    for key in ("L_REG2", "L_FRP1", "R_REG1", "R_REG2", "R_FRP1"):
        np.testing.assert_allclose(result[key], clean[key], err_msg=f"{key} was affected by a NaN outside it")


def test_extract_all_regions_all_nan_region_frame_is_nan(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    series = region_deltaf_series.copy()
    series[0, 0:2, 0:3] = np.nan

    with (
        warnings.catch_warnings(),
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        result = extract_all_regions(series, ignore_default_exclude=True)

    assert np.isnan(result["L_REG1"][0])
    assert not np.isnan(result["L_REG1"][1:]).any()
    assert not np.isnan(result["L_REG2"]).any()


def test_extract_all_regions_empty_region_is_nan(
    region_left_aba, region_right_aba, region_annotations_with_empty, region_deltaf_series
):
    """A region with no pixels in the atlas gives NaN rather than nulling its neighbours."""
    with (
        warnings.catch_warnings(),
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations_with_empty),
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        result = extract_all_regions(region_deltaf_series, ignore_default_exclude=True)

    assert np.isnan(result["L_EMPTY"]).all()
    assert not np.isnan(result["L_REG1"]).any()


def test_extract_all_regions_matches_extract_region_activity_with_nans(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    """The two ABA paths must agree once NaNs are in play, including on an all-NaN frame."""
    series = region_deltaf_series.copy()
    series[:, 0, 0] = np.nan
    series[2, 2:4, 0:3] = np.nan

    with (
        warnings.catch_warnings(),
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        all_regions = extract_all_regions(series, ignore_default_exclude=True)
        for region in ("REG1", "REG2"):
            for hemisphere, prefix in (("left", "L"), ("right", "R")):
                single = extract_region_activity(series, region, hemisphere)
                np.testing.assert_allclose(all_regions[f"{prefix}_{region}"], single)


def test_extract_all_regions_matches_extract_mask_activity_with_nans(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    """The ABA and custom-mask paths must agree on what a NaN pixel means."""
    series = region_deltaf_series.copy()
    series[:, 0, 0] = np.nan

    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        result = extract_all_regions(series, ignore_default_exclude=True)

    np.testing.assert_allclose(result["L_REG1"], extract_mask_activity(series, region_left_aba == 1))


def test_extract_all_regions_handles_nans_across_frame_blocks(region_left_aba, region_right_aba, region_annotations):
    """Frames are processed in blocks, so NaN counts must stay per-frame across a block boundary."""
    rng = np.random.default_rng(1)
    series = rng.random((_LONG_N_FRAMES, _ATLAS_H, _ATLAS_W))
    series[::3, 0, 0] = np.nan
    expected = np.nanmean(series[:, 0:2, 0:3].reshape(_LONG_N_FRAMES, -1), axis=1)

    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        result = extract_all_regions(series, ignore_default_exclude=True)

    np.testing.assert_allclose(result["L_REG1"], expected)


# ---------------------------------------------------------------------------
# region.extract_mask_activity / region.extract_all_masks
# ---------------------------------------------------------------------------


@pytest.fixture
def custom_mask():
    """A 6x6 boolean mask covering rows 0-1, cols 0-2 -- the same pixels as REG1 in the left mock atlas."""
    mask = np.zeros((_ATLAS_H, _ATLAS_W), dtype=bool)
    mask[0:2, 0:3] = True
    return mask


@pytest.fixture
def custom_masks(custom_mask):
    """Two non-overlapping boolean masks, as returned by io.read_mask."""
    second = np.zeros((_ATLAS_H, _ATLAS_W), dtype=bool)
    second[4:6, 3:6] = True
    return {"roi_1": custom_mask, "roi_2": second}


def test_extract_mask_activity_shape(region_deltaf_series, custom_mask):
    result = extract_mask_activity(region_deltaf_series, custom_mask)
    assert result.shape == (_N_FRAMES,)


def test_extract_mask_activity_values(region_deltaf_series, custom_mask):
    expected = region_deltaf_series[:, 0:2, 0:3].mean(axis=(1, 2))
    result = extract_mask_activity(region_deltaf_series, custom_mask)
    np.testing.assert_allclose(result, expected)


def test_extract_mask_activity_accepts_non_boolean_mask(region_deltaf_series, custom_mask):
    result = extract_mask_activity(region_deltaf_series, custom_mask.astype(np.uint8))
    np.testing.assert_allclose(result, extract_mask_activity(region_deltaf_series, custom_mask))


def test_extract_mask_activity_is_not_mirrored_across_hemispheres(region_deltaf_series, custom_mask):
    """Unlike the ABA path, a custom mask is used as drawn -- the mirrored pixels must not contribute."""
    mirrored = extract_mask_activity(region_deltaf_series, np.flip(custom_mask, axis=1))
    result = extract_mask_activity(region_deltaf_series, custom_mask)
    assert not np.allclose(result, mirrored)


def test_extract_mask_activity_matches_aba_extraction(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    """A mask drawn over an ABA region should give the same trace as extracting that region."""
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations),
    ):
        aba = extract_region_activity(region_deltaf_series, "REG2", "left")
    result = extract_mask_activity(region_deltaf_series, region_left_aba == 2)
    np.testing.assert_allclose(result, aba)


def test_extract_mask_activity_ignores_nan_pixels(region_deltaf_series, custom_mask):
    series = region_deltaf_series.copy()
    series[:, 0, 0] = np.nan
    expected = np.nanmean(series[:, 0:2, 0:3].reshape(_N_FRAMES, -1), axis=1)

    result = extract_mask_activity(series, custom_mask)

    assert not np.isnan(result).any()
    np.testing.assert_allclose(result, expected)


def test_extract_mask_activity_all_nan_frame_is_nan(region_deltaf_series, custom_mask):
    series = region_deltaf_series.copy()
    series[0, 0:2, 0:3] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = extract_mask_activity(series, custom_mask)

    assert np.isnan(result[0])
    assert not np.isnan(result[1:]).any()


def test_extract_mask_activity_shape_mismatch_raises(region_deltaf_series):
    with pytest.raises(ValueError, match="does not match the recording frame shape"):
        extract_mask_activity(region_deltaf_series, np.ones((3, 3), dtype=bool))


def test_extract_mask_activity_empty_mask_raises(region_deltaf_series):
    with pytest.raises(ValueError, match="does not select any pixels"):
        extract_mask_activity(region_deltaf_series, np.zeros((_ATLAS_H, _ATLAS_W), dtype=bool))


def test_extract_all_masks_returns_dict_keyed_by_mask_name(region_deltaf_series, custom_masks):
    result = extract_all_masks(region_deltaf_series, custom_masks)
    assert isinstance(result, dict)
    assert list(result) == ["roi_1", "roi_2"]
    assert all(trace.shape == (_N_FRAMES,) for trace in result.values())


def test_extract_all_masks_values_match_extract_mask_activity(region_deltaf_series, custom_masks):
    result = extract_all_masks(region_deltaf_series, custom_masks)
    for name, mask in custom_masks.items():
        np.testing.assert_allclose(result[name], extract_mask_activity(region_deltaf_series, mask))


def test_extract_all_masks_as_dataframe(region_deltaf_series, custom_masks):
    df = extract_all_masks(region_deltaf_series, custom_masks, as_dataframe=True)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["region", "time_idx", "F"]
    assert set(df["region"]) == {"roi_1", "roi_2"}
    assert len(df) == len(custom_masks) * _N_FRAMES


def test_extract_all_masks_error_names_the_offending_mask(region_deltaf_series, custom_mask):
    masks = {"good": custom_mask, "bad": np.zeros((_ATLAS_H, _ATLAS_W), dtype=bool)}
    with pytest.raises(ValueError, match="Mask bad does not select any pixels"):
        extract_all_masks(region_deltaf_series, masks)


def test_extract_all_masks_empty_input_returns_empty_dict(region_deltaf_series):
    assert extract_all_masks(region_deltaf_series, {}) == {}


# ---------------------------------------------------------------------------
# Peri-event
# ---------------------------------------------------------------------------

PERIEVENT_FREQ_HZ = 0.5
PERIEVENT_REGIONS = ["L_MOp", "R_MOp"]


def _perievent_signal(t):
    """Known sinusoid, one cycle every two seconds."""
    return np.sin(2 * np.pi * PERIEVENT_FREQ_HZ * t)


@pytest.fixture(scope="module")
def perievent_time():
    """Jittered ~25 Hz sample times from 2 s to ~42 s after behaviour start."""
    rng = np.random.default_rng(3)
    return 2.0 + np.cumsum(0.04 + rng.uniform(-0.005, 0.005, size=1000))


@pytest.fixture(scope="module")
def perievent_h5(tmp_path_factory, perievent_time):
    """(1000, 4, 4) recording of the sinusoid offset by pixel index, with /timestamps_aligned."""
    path = tmp_path_factory.mktemp("data") / "ses-01_zscored.h5"
    frames = _perievent_signal(perievent_time)[:, None, None] + np.arange(16).reshape(4, 4)[None]
    session_start = datetime(2024, 1, 1, 14, 0, 0)
    timestamps = [(session_start + timedelta(seconds=t)).isoformat().encode("utf-8") for t in perievent_time]
    with h5.File(str(path), "w") as f:
        f.create_dataset("/F", data=frames)
        f.create_dataset("/timestamps", data=timestamps, dtype="S26")
    io.write_timestamps_aligned(
        str(path),
        perievent_time,
        {"session_start_time": session_start.isoformat(), "behaviour_session": "ses-01", "offset_s": 2.0},
    )
    return str(path)


@pytest.fixture(scope="module")
def perievent_regions_csv(tmp_path_factory, perievent_time):
    """Long-format regions CSV of the sinusoid, one region offset by 1."""
    path = tmp_path_factory.mktemp("data") / "ses-01_regions.csv"
    session_start = datetime(2024, 1, 1, 14, 0, 0)
    frames = [
        pd.DataFrame(
            {
                "region": region,
                "timestamp": [(session_start + timedelta(seconds=t)).isoformat() for t in perievent_time],
                "time_aligned": perievent_time,
                "F": _perievent_signal(perievent_time) + offset,
            }
        )
        for offset, region in enumerate(PERIEVENT_REGIONS)
    ]
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="module")
def perievent_trials_csv(tmp_path_factory):
    """Six trials: one near each end of the recording, all four outcomes, one negative response time."""
    path = tmp_path_factory.mktemp("data") / "ses-01_trials.csv"
    pd.DataFrame(
        {
            "start_time": [0.5, 5.0, 10.0, 15.0, 20.0, 40.0],
            "cue_onset": [1.0, 5.5, 10.5, 15.5, 20.5, 40.5],
            "stop_time": [3.0, 8.0, 13.0, 18.0, 23.0, 41.5],
            "response_time": [0.7, 1.2, np.nan, -1.0, 0.8, 0.5],
            "sdt_type": ["hit", "hit", "miss", "false_alarm", "correct_rejection", "hit"],
        }
    ).to_csv(path, index=False)
    return str(path)


class TestEventTimes:
    @pytest.mark.parametrize(
        ("event", "expected_times", "expected_index"),
        [
            ("cue_onset", [1.0, 5.5, 10.5, 15.5, 20.5, 40.5], [0, 1, 2, 3, 4, 5]),
            ("trial_start", [0.5, 5.0, 10.0, 15.0, 20.0, 40.0], [0, 1, 2, 3, 4, 5]),
            ("response", [1.2, 6.2, 20.8, 40.5], [0, 1, 4, 5]),
            ("reward", [3.0, 8.0, 23.0, 41.5], [0, 1, 4, 5]),
        ],
    )
    def test_times_and_drops(self, perievent_trials_csv, event, expected_times, expected_index):
        times, trial_index = pev.event_times(pd.read_csv(perievent_trials_csv), event)
        np.testing.assert_allclose(times, expected_times)
        np.testing.assert_array_equal(trial_index, expected_index)

    def test_unknown_event_raises(self, perievent_trials_csv):
        with pytest.raises(ValueError, match="Unknown event"):
            pev.event_times(pd.read_csv(perievent_trials_csv), "lick")


class TestWindowGrid:
    def test_inclusive_of_post(self):
        grid = pev.window_grid(1.0, 3.0, 25.0)
        assert len(grid) == 101
        assert grid[0] == pytest.approx(-1.0)
        assert grid[-1] == pytest.approx(3.0)
        np.testing.assert_allclose(np.diff(grid), 0.04)

    def test_non_integer_span_stops_within_post(self):
        grid = pev.window_grid(0.5, 1.03, 25.0)
        assert grid[-1] <= 1.03 + 1e-9
        assert grid[-1] > 1.03 - 0.04

    @pytest.mark.parametrize(("pre", "post", "fs"), [(1.0, 3.0, 25.0), (0.5, 2.0, 30.0), (0.1, 0.7, 30.0)])
    def test_event_and_ends_are_exact(self, pre, post, fs):
        grid = pev.window_grid(pre, post, fs)
        assert grid[0] == -pre
        assert 0.0 in grid
        assert grid[-1] == post


class TestExtract:
    @pytest.fixture
    def grid(self):
        return pev.window_grid(1.0, 3.0, 25.0)

    def test_interp_recovers_sinusoid(self, perievent_time, grid):
        data = _perievent_signal(perievent_time)[:, None]
        events = np.array([5.5, 20.5])
        traces, kept = pev.extract(data, perievent_time, events, grid, method="interp", dtype=np.float64)
        assert kept.all()
        assert traces.shape == (2, len(grid), 1)
        expected = _perievent_signal(events[:, None] + grid[None, :])
        np.testing.assert_allclose(traces[:, :, 0], expected, atol=5e-3)

    def test_nearest_returns_recorded_samples(self, perievent_time, grid):
        data = _perievent_signal(perievent_time)[:, None]
        traces, kept = pev.extract(data, perievent_time, np.array([10.5]), grid, method="nearest", dtype=np.float64)
        assert kept.all()
        recorded = set(data[:, 0].tolist())
        assert all(value in recorded for value in traces[0, :, 0])
        # Each sample is the closest frame to its grid time.
        idx = np.abs(perievent_time[None, :] - (10.5 + grid)[:, None]).argmin(axis=1)
        np.testing.assert_array_equal(traces[0, :, 0], data[idx, 0])

    def test_edge_trials_dropped(self, perievent_time, grid):
        data = _perievent_signal(perievent_time)[:, None]
        events = np.array([1.0, 5.5, 40.5])
        traces, kept = pev.extract(data, perievent_time, events, grid)
        np.testing.assert_array_equal(kept, [False, True, False])
        assert traces.shape == (1, len(grid), 1)
        assert traces.dtype == np.float32

    def test_preserves_frame_shape(self, perievent_time, grid):
        data = np.broadcast_to(_perievent_signal(perievent_time)[:, None, None], (len(perievent_time), 3, 5))
        traces, _ = pev.extract(data, perievent_time, np.array([5.5]), grid)
        assert traces.shape == (1, len(grid), 3, 5)

    def test_unknown_method_raises(self, perievent_time, grid):
        with pytest.raises(ValueError, match="Unknown method"):
            pev.extract(np.zeros((len(perievent_time), 1)), perievent_time, np.array([5.5]), grid, method="cubic")


class TestApplyBaseline:
    def test_window_mean_is_zero(self):
        grid = pev.window_grid(1.0, 3.0, 25.0)
        rng = np.random.default_rng(4)
        traces = rng.normal(size=(3, len(grid), 2, 2)) + 5.0
        corrected = pev.apply_baseline(traces, grid, -1.0, 0.0)
        window = (grid >= -1.0) & (grid < 0.0)
        np.testing.assert_allclose(corrected[:, window].mean(axis=1), 0.0, atol=1e-12)
        assert corrected.shape == traces.shape
        assert corrected.dtype == traces.dtype

    def test_empty_window_raises(self):
        grid = pev.window_grid(1.0, 3.0, 25.0)
        with pytest.raises(ValueError, match="baseline window"):
            pev.apply_baseline(np.zeros((1, len(grid))), grid, 5.0, 6.0)


def test_perievent_cmd_h5(perievent_h5, perievent_trials_csv, output_dir):
    result = CliRunner().invoke(
        mesoscopy.cli, args=f"process peri-event {perievent_h5} {perievent_trials_csv} -o {output_dir}"
    )
    assert result.exit_code == 0, result.output
    assert "Kept 4 trials, dropped 2." in result.output

    outpath = pathlib.Path(output_dir) / "ses-01_zscored_event-cueonset_perievent.h5"
    assert outpath.is_file()
    with h5.File(str(outpath), "r") as f:
        traces = f["/traces"]
        assert traces.shape == (4, 101, 4, 4)
        assert traces.dtype == np.float32
        assert traces.chunks == (1, 101, 4, 4)
        assert f["/time"].dtype == np.float64
        np.testing.assert_allclose(f["/time"][:], pev.window_grid(1.0, 3.0, 25.0))
        assert f["/trial_index"].dtype.kind == "i"
        np.testing.assert_array_equal(f["/trial_index"][:], [1, 2, 3, 4])
        assert f["/event_time"].dtype == np.float64
        np.testing.assert_allclose(f["/event_time"][:], [5.5, 10.5, 15.5, 20.5])

        attrs = traces.attrs
        assert attrs["event"] == "cue_onset"
        assert attrs["pre"] == 1.0
        assert attrs["post"] == 3.0
        assert attrs["fs"] == 25.0
        assert attrs["method"] == "interp"
        assert len(attrs["baseline"]) == 0
        assert attrs["session_start_time"] == "2024-01-01T14:00:00"
        assert attrs["behaviour_session"] == "ses-01"
        assert attrs["trials_path"] == perievent_trials_csv

        # Pixel offsets survive; the sinusoid is recovered at the event.
        trace = traces[0]
        np.testing.assert_allclose(trace[:, 0, 0] + 5, trace[:, 1, 1], atol=1e-5)
        np.testing.assert_allclose(trace[25, 0, 0], _perievent_signal(5.5), atol=5e-3)


def test_perievent_cmd_h5_baseline_and_nearest(perievent_h5, perievent_trials_csv, output_dir):
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=(
            f"process peri-event {perievent_h5} {perievent_trials_csv} -o {output_dir} --event reward"
            " --method nearest --baseline -0.5 0"
        ),
    )
    assert result.exit_code == 0, result.output

    with h5.File(str(pathlib.Path(output_dir) / "ses-01_zscored_event-reward_perievent.h5"), "r") as f:
        np.testing.assert_array_equal(f["/trial_index"][:], [1, 4])
        np.testing.assert_allclose(f["/traces"].attrs["baseline"], [-0.5, 0.0])
        assert f["/traces"].attrs["method"] == "nearest"
        grid = f["/time"][:]
        window = (grid >= -0.5) & (grid < 0)
        np.testing.assert_allclose(f["/traces"][:, window].mean(axis=1), 0.0, atol=1e-5)


def test_perievent_cmd_h5_without_timestamps_aligned(preproc_h5, perievent_trials_csv, output_dir):
    result = CliRunner().invoke(
        mesoscopy.cli, args=f"process peri-event {preproc_h5} {perievent_trials_csv} -o {output_dir}"
    )
    assert result.exit_code != 0
    assert "timestamps_aligned" in result.output


def test_perievent_cmd_csv(perievent_regions_csv, perievent_trials_csv, output_dir):
    result = CliRunner().invoke(
        mesoscopy.cli, args=f"process peri-event {perievent_regions_csv} {perievent_trials_csv} -o {output_dir}"
    )
    assert result.exit_code == 0, result.output

    outpath = pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_perievent.csv"
    assert outpath.is_file()
    out = pd.read_csv(outpath)
    assert list(out.columns) == [
        "trial_index",
        "event_time",
        "time",
        "region",
        "F",
        "start_time",
        "cue_onset",
        "stop_time",
        "response_time",
        "sdt_type",
    ]
    assert len(out) == 4 * 101 * len(PERIEVENT_REGIONS)
    assert sorted(out["trial_index"].unique()) == [1, 2, 3, 4]
    assert list(out["region"].unique()) == PERIEVENT_REGIONS
    per_trial = out.drop_duplicates(subset="trial_index").set_index("trial_index")
    assert per_trial["sdt_type"].tolist() == ["hit", "miss", "false_alarm", "correct_rejection"]
    np.testing.assert_allclose(per_trial["cue_onset"], per_trial["event_time"])

    trial = out[(out["trial_index"] == 1) & (out["region"] == "R_MOp")]
    assert trial["event_time"].unique().tolist() == [5.5]
    np.testing.assert_allclose(trial["time"], pev.window_grid(1.0, 3.0, 25.0))
    np.testing.assert_allclose(trial["F"], _perievent_signal(5.5 + trial["time"].to_numpy()) + 1, atol=5e-3)


def test_perievent_cmd_csv_trials_with_index_column(perievent_regions_csv, perievent_trials_csv, output_dir, tmp_path):
    indexed = tmp_path / "ses-01_trials.csv"
    pd.read_csv(perievent_trials_csv).to_csv(indexed)  # leading "Unnamed: 0" column, as pandas writes by default
    result = CliRunner().invoke(
        mesoscopy.cli, args=f"process peri-event {perievent_regions_csv} {indexed} -o {output_dir}"
    )
    assert result.exit_code == 0, result.output

    out = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_perievent.csv")
    assert not any(column.startswith("Unnamed") for column in out.columns)
    assert out.drop_duplicates(subset="trial_index")["sdt_type"].tolist() == [
        "hit",
        "miss",
        "false_alarm",
        "correct_rejection",
    ]


def test_perievent_cmd_csv_without_time_aligned(perievent_regions_csv, perievent_trials_csv, output_dir, tmp_path):
    unaligned = tmp_path / "ses-02_regions.csv"
    pd.read_csv(perievent_regions_csv).drop(columns="time_aligned").to_csv(unaligned, index=False)
    result = CliRunner().invoke(
        mesoscopy.cli, args=f"process peri-event {unaligned} {perievent_trials_csv} -o {output_dir}"
    )
    assert result.exit_code != 0
    assert "time_aligned" in result.output


@pytest.mark.parametrize(
    ("event", "name"),
    [("cue_onset", "cueonset"), ("trial_start", "trialstart"), ("response", "response"), ("reward", "reward")],
)
def test_perievent_cmd_output_filename(perievent_regions_csv, perievent_trials_csv, output_dir, event, name):
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=f"process peri-event {perievent_regions_csv} {perievent_trials_csv} -o {output_dir} --event {event}",
    )
    assert result.exit_code == 0, result.output
    assert (pathlib.Path(output_dir) / f"ses-01_regions_event-{name}_perievent.csv").is_file()


# ---------------------------------------------------------------------------
# region extraction at template scale
# ---------------------------------------------------------------------------


def _upsampled_series(series, factor):
    return np.repeat(np.repeat(series, factor, axis=1), factor, axis=2)


def test_extract_all_regions_matches_at_template_scale():
    """A 2x registered series has every pixel duplicated, so region means are identical to 1x."""
    rng = np.random.default_rng(0)
    series = rng.random((5, *mesoscopy.resources.atlas_shape()), dtype=np.float32)

    native = extract_all_regions(series)
    scaled = extract_all_regions(_upsampled_series(series, 2))

    assert native.keys() == scaled.keys()
    for region in native:
        np.testing.assert_allclose(scaled[region], native[region], rtol=1e-5)


def test_extract_region_activity_matches_at_template_scale():
    rng = np.random.default_rng(1)
    series = rng.random((5, *mesoscopy.resources.atlas_shape()), dtype=np.float32)

    for hemisphere in ("left", "right", "both"):
        native = extract_region_activity(series, "MOp1", hemisphere)
        scaled = extract_region_activity(_upsampled_series(series, 3), "MOp1", hemisphere)
        np.testing.assert_allclose(scaled, native, rtol=1e-5)


def test_extract_all_regions_at_non_integer_template_scale():
    """The atlas is resampled to whatever frame shape the registration produced."""
    shape = mesoscopy.resources.template_shape(1.5)
    series = np.ones((3, *shape), dtype=np.float32)

    activity = extract_all_regions(series)

    assert all(np.allclose(trace, 1.0) for trace in activity.values())


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

METRICS_GRID = pev.window_grid(1.0, 3.0, 25.0)


def _metrics_traces():
    """Four trials on the 25 Hz grid, with alternating +-0.01 samples in the pre-event window.

    Trial 0 rises linearly from 0 at t=0.2 to 1 at t=1.0, falls back to 0 at t=2.0, then stays flat. Trial 1 is
    flat after the event. Trial 2 ramps to 1 at t=3.0, and so never decays. Trial 3 is all NaN.
    """
    t = METRICS_GRID
    noise = np.where(np.arange(len(t)) % 2 == 0, 0.01, -0.01) * (t < 0)
    response = np.interp(t, [0.2, 1.0, 2.0], [0.0, 1.0, 0.0]) * (t >= 0.2)
    ramp = np.interp(t, [0.0, 3.0], [0.0, 1.0]) * (t >= 0)
    return np.stack([response + noise, noise, ramp + noise, np.full_like(t, np.nan)])


class TestWindowMask:
    def test_inclusive_and_exclusive_end(self):
        t = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        np.testing.assert_array_equal(pm.window_mask(t, 0.0, 1.0), [False, False, True, True, True])
        np.testing.assert_array_equal(
            pm.window_mask(t, -1.0, 0.0, inclusive_end=False), [True, True, False, False, False]
        )

    def test_empty_window_raises(self):
        with pytest.raises(ValueError, match="No samples"):
            pm.window_mask(METRICS_GRID, 5.0, 6.0)


class TestBaselineStats:
    def test_mean_and_sd(self):
        traces = np.array([[1.0, 3.0, 5.0, 100.0], [2.0, 2.0, 2.0, 100.0]])
        t = np.array([-1.0, -0.5, -0.25, 0.0])
        mean, sd = pm.baseline_stats(traces, t, -1.0, 0.0)
        np.testing.assert_allclose(mean, [3.0, 2.0])
        np.testing.assert_allclose(sd, [2.0, 0.0])

    def test_ignores_nan(self):
        traces = np.array([[1.0, np.nan, 5.0, 100.0]])
        t = np.array([-1.0, -0.5, -0.25, 0.0])
        mean, _ = pm.baseline_stats(traces, t, -1.0, 0.0)
        np.testing.assert_allclose(mean, [3.0])


class TestPeak:
    def test_signed_maximum(self):
        amplitude, peak_time, index = pm.peak(_metrics_traces(), METRICS_GRID, 0.0, 3.0)
        np.testing.assert_allclose(amplitude[:3], [1.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(peak_time[:3], [1.0, 0.0, 3.0])
        np.testing.assert_array_equal(index[:3], [50, 25, 100])

    def test_nan_trial(self):
        amplitude, peak_time, index = pm.peak(_metrics_traces(), METRICS_GRID, 0.0, 3.0)
        assert np.isnan(amplitude[3])
        assert np.isnan(peak_time[3])
        assert index[3] == -1

    def test_negative_response_keeps_sign(self):
        traces = -_metrics_traces()[:1]
        amplitude, peak_time, _ = pm.peak(traces, METRICS_GRID, 0.0, 3.0)
        np.testing.assert_allclose(amplitude, [0.0], atol=1e-12)
        np.testing.assert_allclose(peak_time, [0.0])


class TestAuc:
    def test_triangle(self):
        area = pm.auc(_metrics_traces(), METRICS_GRID, 0.0, 3.0)
        np.testing.assert_allclose(area[:3], [0.9, 0.0, 1.5], atol=1e-12)
        assert np.isnan(area[3])

    def test_window(self):
        area = pm.auc(_metrics_traces(), METRICS_GRID, 0.0, 1.0)
        np.testing.assert_allclose(area[0], 0.4, atol=1e-12)


class TestOnsetTime:
    def test_threshold_crossing(self):
        onset = pm.onset_time(_metrics_traces(), METRICS_GRID, np.full(4, 0.02), 0.0, 3.0)
        np.testing.assert_allclose(onset[:3], [0.24, np.nan, 0.08])
        assert np.isnan(onset[3])

    def test_nan_threshold_gives_nan(self):
        onset = pm.onset_time(_metrics_traces(), METRICS_GRID, np.array([np.nan, 0.02, 0.02, 0.02]), 0.0, 3.0)
        assert np.isnan(onset[0])

    def test_min_samples_rejects_blips(self):
        t = METRICS_GRID
        trace = np.zeros((1, len(t)))
        trace[0, 30:32] = 1.0  # two samples at t=0.2, 0.24
        trace[0, 40:43] = 1.0  # three samples from t=0.6
        np.testing.assert_allclose(pm.onset_time(trace, t, np.array([0.5]), 0.0, 3.0, min_samples=3), [0.6])
        np.testing.assert_allclose(pm.onset_time(trace, t, np.array([0.5]), 0.0, 3.0, min_samples=1), [0.2])
        assert np.isnan(pm.onset_time(trace, t, np.array([0.5]), 0.0, 3.0, min_samples=4))[0]

    def test_window_shorter_than_min_samples(self):
        onset = pm.onset_time(_metrics_traces(), METRICS_GRID, np.full(4, 0.02), 0.0, 0.04, min_samples=3)
        assert np.all(np.isnan(onset))

    def test_min_samples_below_one_raises(self):
        with pytest.raises(ValueError, match="min_samples"):
            pm.onset_time(_metrics_traces(), METRICS_GRID, np.full(4, 0.02), 0.0, 3.0, min_samples=0)


class TestDecayTime:
    def test_half_decay(self):
        traces = _metrics_traces()
        amplitude, _, index = pm.peak(traces, METRICS_GRID, 0.0, 3.0)
        decay = pm.decay_time(traces, METRICS_GRID, index, amplitude, 3.0)
        np.testing.assert_allclose(decay[0], 0.52)
        assert np.all(np.isnan(decay[1:]))  # flat, never decays, NaN

    def test_fraction(self):
        traces = _metrics_traces()
        amplitude, _, index = pm.peak(traces, METRICS_GRID, 0.0, 3.0)
        decay = pm.decay_time(traces, METRICS_GRID, index, amplitude, 3.0, fraction=0.1)
        np.testing.assert_allclose(decay[0], 0.92)

    def test_window_end_limits_search(self):
        traces = _metrics_traces()
        amplitude, _, index = pm.peak(traces, METRICS_GRID, 0.0, 1.2)
        assert np.isnan(pm.decay_time(traces, METRICS_GRID, index, amplitude, 1.2)[0])

    @pytest.mark.parametrize("fraction", [0.0, 1.0, 1.5])
    def test_fraction_out_of_range_raises(self, fraction):
        with pytest.raises(ValueError, match="fraction"):
            pm.decay_time(_metrics_traces(), METRICS_GRID, np.zeros(4, dtype=np.int64), np.ones(4), 3.0, fraction)


class TestSmooth:
    def test_window_one_is_identity(self):
        traces = _metrics_traces()
        out = pm.smooth(traces, 1)
        np.testing.assert_array_equal(out, traces)
        assert out is not traces

    def test_moving_average_and_edges(self):
        trace = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        np.testing.assert_allclose(pm.smooth(trace, 3), [[1.5, 2.0, 3.0, 4.0, 4.5]])
        np.testing.assert_allclose(pm.smooth(trace, 5), [[2.0, 2.5, 3.0, 3.5, 4.0]])

    def test_ignores_nan(self):
        trace = np.array([[1.0, np.nan, 3.0, np.nan, np.nan]])
        out = pm.smooth(trace, 3)
        np.testing.assert_allclose(out[0, :3], [1.0, 2.0, 3.0])
        assert out[0, 3] == 3.0
        assert np.isnan(out[0, 4])

    @pytest.mark.parametrize("window", [0, 2, 4, -1])
    def test_even_or_non_positive_raises(self, window):
        with pytest.raises(ValueError, match="odd"):
            pm.smooth(_metrics_traces(), window)


class TestExtrapolatedOnset:
    def test_linear_rise(self):
        traces = _metrics_traces()
        amplitude, _, index = pm.peak(traces, METRICS_GRID, 0.0, 3.0)
        onset = pm.extrapolated_onset(traces, METRICS_GRID, index, amplitude, 0.0)
        assert onset[0] == pytest.approx(0.2, abs=1e-3)
        assert np.isnan(onset[1])
        assert onset[2] == pytest.approx(0.0, abs=1e-3)
        assert np.isnan(onset[3])

    def test_fits_last_rise_before_peak(self):
        t = METRICS_GRID
        # An early bump to 0.6 that falls back, then the real rise from t=1.0 to the peak at t=2.0.
        trace = np.interp(t, [0.0, 0.2, 0.4, 1.0, 2.0], [0.0, 0.6, 0.0, 0.0, 1.0])[None]
        amplitude, _, index = pm.peak(trace, t, 0.0, 3.0)
        onset = pm.extrapolated_onset(trace, t, index, amplitude, 0.0)
        assert onset[0] == pytest.approx(1.0, abs=1e-9)

    def test_too_few_rise_samples(self):
        t = METRICS_GRID
        trace = np.where(t >= 1.0, 1.0, 0.0)[None]  # a step: no samples between 0.2 and 0.8
        amplitude, _, index = pm.peak(trace, t, 0.0, 3.0)
        assert np.isnan(pm.extrapolated_onset(trace, t, index, amplitude, 0.0))[0]

    @pytest.mark.parametrize(("low", "high"), [(0.0, 0.8), (0.2, 1.0), (0.8, 0.2), (0.5, 0.5)])
    def test_bad_range_raises(self, low, high):
        with pytest.raises(ValueError, match="fractions"):
            pm.extrapolated_onset(
                _metrics_traces(), METRICS_GRID, np.zeros(4, dtype=np.int64), np.ones(4), 0.0, low, high
            )


class TestOffsetTime:
    def test_return_to_threshold(self):
        traces = _metrics_traces()
        amplitude, _, index = pm.peak(traces, METRICS_GRID, 0.0, 3.0)
        offset = pm.offset_time(traces, METRICS_GRID, np.full(4, 0.02), index, 3.0)
        assert offset[0] == pytest.approx(2.0)
        assert np.all(np.isnan(offset[1:]))  # never above, never returns, NaN

    def test_min_samples_rejects_dips(self):
        t = METRICS_GRID
        trace = np.ones((1, len(t)))
        trace[0, 40:42] = 0.0  # two samples from t=0.6
        trace[0, 60:] = 0.0  # from t=1.4 onward
        index = np.array([30])
        np.testing.assert_allclose(pm.offset_time(trace, t, np.array([0.5]), index, 3.0, min_samples=3), [1.4])
        np.testing.assert_allclose(pm.offset_time(trace, t, np.array([0.5]), index, 3.0, min_samples=1), [0.6])

    def test_window_end_limits_search(self):
        traces = _metrics_traces()
        amplitude, _, index = pm.peak(traces, METRICS_GRID, 0.0, 1.5)
        assert np.isnan(pm.offset_time(traces, METRICS_GRID, np.full(4, 0.02), index, 1.5)[0])

    def test_nan_threshold(self):
        traces = _metrics_traces()
        amplitude, _, index = pm.peak(traces, METRICS_GRID, 0.0, 3.0)
        assert np.all(np.isnan(pm.offset_time(traces, METRICS_GRID, np.full(4, np.nan), index, 3.0)))

    def test_min_samples_below_one_raises(self):
        with pytest.raises(ValueError, match="min_samples"):
            pm.offset_time(_metrics_traces(), METRICS_GRID, np.zeros(4), np.zeros(4, dtype=np.int64), 3.0, 0)


class TestTraceCorrelation:
    def test_identical_and_opposite(self):
        trace = _metrics_traces()[0]
        assert pm.trace_correlation(np.stack([trace, trace]), METRICS_GRID, 0.0, 3.0) == pytest.approx(1.0)
        assert pm.trace_correlation(np.stack([trace, -trace]), METRICS_GRID, 0.0, 3.0) == pytest.approx(-1.0)

    def test_mean_over_pairs(self):
        trace = _metrics_traces()[0]
        correlation = pm.trace_correlation(np.stack([trace, trace, -trace]), METRICS_GRID, 0.0, 3.0)
        assert correlation == pytest.approx((1.0 - 1.0 - 1.0) / 3)

    def test_ignores_nan_trials(self):
        trace = _metrics_traces()[0]
        nan = np.full_like(trace, np.nan)
        assert pm.trace_correlation(np.stack([trace, trace, nan]), METRICS_GRID, 0.0, 3.0) == pytest.approx(1.0)

    def test_single_trial_is_nan(self):
        assert np.isnan(pm.trace_correlation(_metrics_traces()[:1], METRICS_GRID, 0.0, 3.0))


class TestTrialMetrics:
    def test_columns_and_values(self):
        metrics = pm.trial_metrics(_metrics_traces(), METRICS_GRID, baseline=(-1.0, 0.0), response=(0.0, 3.0))
        assert list(metrics.columns) == [
            "baseline_mean",
            "baseline_sd",
            "onset_time",
            "peak_time",
            "amplitude",
            "auc",
            "decay_time",
            "offset_time",
            "duration",
        ]
        assert len(metrics) == 4
        row = metrics.iloc[0]
        assert row["baseline_mean"] == pytest.approx(0.01 / 25)
        assert row["baseline_sd"] == pytest.approx(0.01, abs=1e-3)
        assert row["onset_time"] == pytest.approx(0.24)
        assert row["peak_time"] == pytest.approx(1.0)
        assert row["amplitude"] == pytest.approx(1.0, abs=1e-3)
        assert row["auc"] == pytest.approx(0.9, abs=2e-3)
        assert row["decay_time"] == pytest.approx(0.52)
        assert row["offset_time"] == pytest.approx(2.0)
        assert row["duration"] == pytest.approx(1.76)

        flat = metrics.iloc[1]
        assert np.isnan(flat["onset_time"])
        assert flat["amplitude"] < 0
        assert np.isnan(flat["decay_time"])
        assert np.isnan(flat["offset_time"])
        assert np.isnan(flat["duration"])

        ramp = metrics.iloc[2]
        assert ramp["onset_time"] == pytest.approx(0.08)
        assert np.isnan(ramp["offset_time"])  # never returns to threshold

        assert metrics.iloc[3].isna().all()

    def test_onset_from_peak_fraction(self):
        metrics = pm.trial_metrics(
            _metrics_traces(), METRICS_GRID, (-1.0, 0.0), (0.0, 3.0), onset="peak", onset_fraction=0.53
        )
        # Rises through 0.53 of the amplitude between t=0.6 (0.5) and t=0.64 (0.55).
        assert metrics["onset_time"].iloc[0] == pytest.approx(0.64)
        assert np.isnan(metrics["onset_time"].iloc[1])  # amplitude is not positive
        # Offset is the return to the same fraction: falls through 0.53 between t=1.44 (0.56) and t=1.48 (0.52).
        assert metrics["offset_time"].iloc[0] == pytest.approx(1.48)
        assert metrics["duration"].iloc[0] == pytest.approx(1.48 - 0.64)

    def test_onset_extrapolate(self):
        metrics = pm.trial_metrics(
            _metrics_traces(),
            METRICS_GRID,
            (-1.0, 0.0),
            (0.0, 3.0),
            onset="extrapolate",
            extrapolate_range=(0.25, 0.75),
        )
        # The rise is a line from (0.2, 0) to (1.0, 1), so extrapolation recovers 0.2; the offset is the return
        # to 0.25 of the amplitude, between t=1.72 (0.28) and t=1.76 (0.24).
        assert metrics["onset_time"].iloc[0] == pytest.approx(0.2, abs=1e-3)
        assert metrics["offset_time"].iloc[0] == pytest.approx(1.76)
        assert np.isnan(metrics["onset_time"].iloc[1])
        assert metrics["onset_time"].iloc[2] == pytest.approx(0.0, abs=2e-3)

    def test_smoothing_ignores_spike(self):
        traces = _metrics_traces()[:1].copy()
        traces[0, 35] = 2.0  # one-sample spike at t=0.4, above the true peak
        raw = pm.trial_metrics(traces, METRICS_GRID, (-1.0, 0.0), (0.0, 3.0))
        smoothed = pm.trial_metrics(traces, METRICS_GRID, (-1.0, 0.0), (0.0, 3.0), smoothing=5)
        assert raw["peak_time"].iloc[0] == pytest.approx(0.4)
        assert raw["decay_time"].iloc[0] == pytest.approx(0.04)
        assert smoothed["peak_time"].iloc[0] == pytest.approx(1.0)
        assert smoothed["amplitude"].iloc[0] == pytest.approx(0.94, abs=0.01)  # the averaged triangle tip
        assert smoothed["decay_time"].iloc[0] == pytest.approx(0.52, abs=0.05)
        # AUC and baseline SD come from the raw trace.
        assert smoothed["auc"].iloc[0] == raw["auc"].iloc[0]
        assert smoothed["baseline_sd"].iloc[0] == raw["baseline_sd"].iloc[0]

    def test_onset_sd_multiple(self):
        loose = pm.trial_metrics(_metrics_traces(), METRICS_GRID, (-1.0, 0.0), (0.0, 3.0), onset_sd=1.0)
        strict = pm.trial_metrics(_metrics_traces(), METRICS_GRID, (-1.0, 0.0), (0.0, 3.0), onset_sd=30.0)
        assert loose["onset_time"].iloc[0] == pytest.approx(0.24)
        assert strict["onset_time"].iloc[0] == pytest.approx(0.48)

    def test_unknown_onset_raises(self):
        with pytest.raises(ValueError, match="onset method"):
            pm.trial_metrics(_metrics_traces(), METRICS_GRID, (-1.0, 0.0), (0.0, 3.0), onset="slope")


class TestSessionMetrics:
    def test_summary(self):
        metrics = pd.DataFrame(
            {
                "onset_time": [0.2, 0.4, np.nan],
                "peak_time": [1.0, 1.0, 1.0],
                "amplitude": [1.0, 3.0, 5.0],
                "auc": [0.0, 0.0, 0.0],
                "decay_time": [np.nan, np.nan, np.nan],
                "offset_time": [1.0, np.nan, np.nan],
                "duration": [0.8, np.nan, np.nan],
            }
        )
        summary = pm.session_metrics(metrics, correlation=0.5)
        assert summary["n_trials"] == 3
        assert summary["trace_correlation"] == 0.5
        assert summary["onset_time_mean"] == pytest.approx(0.3)
        assert summary["onset_time_sd"] == pytest.approx(np.std([0.2, 0.4], ddof=1))
        assert summary["onset_time_cv"] == pytest.approx(np.std([0.2, 0.4], ddof=1) / 0.3)
        assert summary["peak_time_sd"] == 0.0
        assert summary["amplitude_mean"] == 3.0
        assert np.isnan(summary["auc_cv"])
        assert np.isnan(summary["decay_time_mean"])
        assert summary["onset_n"] == 2
        assert summary["decay_n"] == 0
        assert summary["offset_n"] == 1
        assert summary["duration_mean"] == 0.8
        assert list(summary)[:2] == ["n_trials", "trace_correlation"]
        assert all(f"{name}_{stat}" in summary for name in pm.METRICS for stat in ("mean", "sd", "cv"))

    def test_single_trial_has_no_sd(self):
        metrics = pd.DataFrame({name: [1.0] for name in pm.METRICS})
        summary = pm.session_metrics(metrics, correlation=np.nan)
        assert summary["amplitude_mean"] == 1.0
        assert np.isnan(summary["amplitude_sd"])


@pytest.fixture(scope="module")
def metrics_perievent_csv(tmp_path_factory):
    """Long-format peri-event CSV of the first three synthetic trials, for two regions with the second doubled."""
    path = tmp_path_factory.mktemp("data") / "ses-01_regions_event-cueonset_perievent.csv"
    traces = _metrics_traces()[:3]
    frames = []
    for factor, region in enumerate(PERIEVENT_REGIONS, start=1):
        for trial, trace in zip([1, 2, 4], traces, strict=True):
            frames.append(
                pd.DataFrame(
                    {
                        "trial_index": trial,
                        "event_time": 5.0 * trial + 0.5,
                        "time": METRICS_GRID,
                        "region": region,
                        "F": factor * trace,
                    }
                )
            )
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return str(path)


class TestMetricsTables:
    def test_tables(self, metrics_perievent_csv):
        per_trial, per_session = pm.metrics_tables(pd.read_csv(metrics_perievent_csv))
        assert list(per_trial.columns[:3]) == ["trial_index", "event_time", "region"]
        assert len(per_trial) == 3 * len(PERIEVENT_REGIONS)
        assert per_session["region"].tolist() == PERIEVENT_REGIONS
        assert per_session["n_trials"].tolist() == [3, 3]
        first = per_trial.iloc[0]
        assert first["onset_time"] == pytest.approx(0.24)
        assert first["amplitude"] == pytest.approx(1.0, abs=1e-3)

    def test_joins_trials(self, metrics_perievent_csv, perievent_trials_csv):
        per_trial, _ = pm.metrics_tables(pd.read_csv(metrics_perievent_csv), trials=pd.read_csv(perievent_trials_csv))
        assert per_trial[per_trial["region"] == "L_MOp"]["sdt_type"].tolist() == ["hit", "miss", "correct_rejection"]

    def test_carries_trial_columns_through(self, metrics_perievent_csv, perievent_trials_csv):
        perievent = pm.join_trials(pd.read_csv(metrics_perievent_csv), pd.read_csv(perievent_trials_csv))
        per_trial, per_session = pm.metrics_tables(perievent)
        plain, _ = pm.metrics_tables(pd.read_csv(metrics_perievent_csv))
        assert list(per_trial.columns) == [
            *plain.columns,
            "start_time",
            "cue_onset",
            "stop_time",
            "response_time",
            "sdt_type",
        ]
        pd.testing.assert_frame_equal(per_trial[plain.columns], plain)
        assert per_trial[per_trial["region"] == "R_MOp"]["sdt_type"].tolist() == ["hit", "miss", "correct_rejection"]
        assert "sdt_type" not in per_session.columns

    def test_varying_column_raises(self, metrics_perievent_csv):
        perievent = pd.read_csv(metrics_perievent_csv)
        perievent["per_sample"] = np.arange(len(perievent))
        perievent["constant"] = 1
        with pytest.raises(ValueError, match="per_sample vary within a trial"):
            pm.metrics_tables(perievent)

    def test_trials_skips_columns_already_present(self, metrics_perievent_csv, perievent_trials_csv):
        trials = pd.read_csv(perievent_trials_csv)
        perievent = pm.join_trials(pd.read_csv(metrics_perievent_csv), trials)
        with_trials, _ = pm.metrics_tables(perievent, trials=trials)
        without, _ = pm.metrics_tables(perievent)
        pd.testing.assert_frame_equal(with_trials, without)

    def test_trials_clashing_column_gets_suffix(self, metrics_perievent_csv, perievent_trials_csv):
        trials = pd.read_csv(perievent_trials_csv)
        trials["duration"] = 9.0
        per_trial, _ = pm.metrics_tables(pd.read_csv(metrics_perievent_csv), trials=trials)
        assert "duration_trial" in per_trial.columns
        assert per_trial["duration_trial"].eq(9.0).all()
        assert not per_trial["duration"].eq(9.0).any()

    def test_options_pass_through(self, metrics_perievent_csv):
        per_trial, _ = pm.metrics_tables(
            pd.read_csv(metrics_perievent_csv),
            baseline=(-0.5, 0.0),
            response=(0.0, 1.2),
            onset="peak",
            onset_fraction=0.53,
        )
        assert per_trial["onset_time"].iloc[0] == pytest.approx(0.64)
        assert per_trial["peak_time"].max() <= 1.2

    def test_missing_columns_raises(self, metrics_perievent_csv):
        with pytest.raises(ValueError, match="F, region"):
            pm.metrics_tables(pd.read_csv(metrics_perievent_csv).drop(columns=["region", "F"]))

    def test_empty_raises(self, metrics_perievent_csv):
        with pytest.raises(ValueError, match="no rows"):
            pm.metrics_tables(pd.read_csv(metrics_perievent_csv).iloc[:0])


def test_metrics_cmd(metrics_perievent_csv, output_dir):
    result = CliRunner().invoke(mesoscopy.cli, args=f"process metrics {metrics_perievent_csv} -o {output_dir}")
    assert result.exit_code == 0, result.output

    trial_path = pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_metrics.csv"
    session_path = pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_metrics-session.csv"
    assert trial_path.is_file()
    assert session_path.is_file()
    assert "Saved per-trial metrics" in result.output

    trial = pd.read_csv(trial_path)
    assert list(trial.columns) == [
        "trial_index",
        "event_time",
        "region",
        "baseline_mean",
        "baseline_sd",
        "onset_time",
        "peak_time",
        "amplitude",
        "auc",
        "decay_time",
        "offset_time",
        "duration",
    ]
    assert len(trial) == 3 * len(PERIEVENT_REGIONS)
    assert list(trial["region"].unique()) == PERIEVENT_REGIONS
    assert trial["trial_index"].tolist() == [1, 2, 4] * len(PERIEVENT_REGIONS)
    np.testing.assert_allclose(trial["event_time"], [5.5, 10.5, 20.5] * len(PERIEVENT_REGIONS))

    first = trial[(trial["region"] == "L_MOp") & (trial["trial_index"] == 1)].iloc[0]
    assert first["onset_time"] == pytest.approx(0.24)
    assert first["peak_time"] == pytest.approx(1.0)
    assert first["amplitude"] == pytest.approx(1.0, abs=1e-3)
    assert first["auc"] == pytest.approx(0.9, abs=2e-3)
    assert first["decay_time"] == pytest.approx(0.52)
    assert first["offset_time"] == pytest.approx(2.0)
    assert first["duration"] == pytest.approx(1.76)
    doubled = trial[(trial["region"] == "R_MOp") & (trial["trial_index"] == 1)].iloc[0]
    assert doubled["amplitude"] == pytest.approx(2 * first["amplitude"])
    assert doubled["auc"] == pytest.approx(2 * first["auc"])
    assert doubled["onset_time"] == pytest.approx(first["onset_time"])

    ramp = trial[(trial["region"] == "L_MOp") & (trial["trial_index"] == 4)].iloc[0]
    assert ramp["onset_time"] == pytest.approx(0.08)
    assert trial["peak_time"].iloc[1] == 0.0

    session = pd.read_csv(session_path)
    assert list(session.columns[:3]) == ["region", "n_trials", "trace_correlation"]
    assert session["region"].tolist() == PERIEVENT_REGIONS
    assert session["n_trials"].tolist() == [3, 3]
    assert session["onset_n"].tolist() == [2, 2]
    assert session["decay_n"].tolist() == [1, 1]
    assert session["offset_n"].tolist() == [1, 1]
    assert session["duration_mean"].tolist() == pytest.approx([1.76, 1.76])
    assert session["amplitude_mean"].iloc[1] == pytest.approx(2 * session["amplitude_mean"].iloc[0])
    assert session["amplitude_cv"].iloc[1] == pytest.approx(session["amplitude_cv"].iloc[0])
    assert session["trace_correlation"].iloc[0] == pytest.approx(session["trace_correlation"].iloc[1])


def test_metrics_cmd_options(metrics_perievent_csv, output_dir):
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=(
            f"process metrics {metrics_perievent_csv} -o {output_dir} --baseline -0.5 0 --response 0 1.2"
            " --onset peak --onset-fraction 0.53 --onset-min-samples 1 --decay-fraction 0.9"
        ),
    )
    assert result.exit_code == 0, result.output

    trial = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_metrics.csv")
    first = trial[(trial["region"] == "L_MOp") & (trial["trial_index"] == 1)].iloc[0]
    assert first["onset_time"] == pytest.approx(0.64)
    assert first["peak_time"] == pytest.approx(1.0)
    assert first["auc"] == pytest.approx(0.4 + 0.2 - 0.02, abs=2e-3)
    assert first["decay_time"] == pytest.approx(0.12)


def test_metrics_cmd_extrapolate_and_smooth(metrics_perievent_csv, output_dir):
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=(
            f"process metrics {metrics_perievent_csv} -o {output_dir} --onset extrapolate"
            " --extrapolate-range 0.3 0.7 --smooth 3"
        ),
    )
    assert result.exit_code == 0, result.output

    trial = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_metrics.csv")
    first = trial[(trial["region"] == "L_MOp") & (trial["trial_index"] == 1)].iloc[0]
    assert first["onset_time"] == pytest.approx(0.2, abs=0.02)
    assert first["peak_time"] == pytest.approx(1.0)
    assert first["offset_time"] == pytest.approx(1.72, abs=0.05)  # return to 0.3 of the amplitude


@pytest.mark.parametrize(("option", "hint"), [("--smooth 4", "odd"), ("--extrapolate-range 0.8 0.2", "LOW < HIGH")])
def test_metrics_cmd_rejects_bad_options(metrics_perievent_csv, output_dir, option, hint):
    result = CliRunner().invoke(mesoscopy.cli, args=f"process metrics {metrics_perievent_csv} -o {output_dir} {option}")
    assert result.exit_code == 2
    assert hint in result.output


def test_metrics_cmd_joins_trials(metrics_perievent_csv, perievent_trials_csv, output_dir):
    result = CliRunner().invoke(
        mesoscopy.cli, args=f"process metrics {metrics_perievent_csv} -o {output_dir} -t {perievent_trials_csv}"
    )
    assert result.exit_code == 0, result.output

    trial = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_metrics.csv")
    assert "sdt_type" in trial.columns
    assert trial.columns.get_loc("sdt_type") > trial.columns.get_loc("duration")
    joined = trial[trial["region"] == "L_MOp"].set_index("trial_index")
    assert joined["sdt_type"].tolist() == ["hit", "miss", "correct_rejection"]
    np.testing.assert_allclose(joined["cue_onset"], [5.5, 10.5, 20.5])


def test_metrics_cmd_joins_trials_with_index_column(metrics_perievent_csv, perievent_trials_csv, output_dir, tmp_path):
    indexed = tmp_path / "ses-01_trials.csv"
    pd.read_csv(perievent_trials_csv).to_csv(indexed)  # leading "Unnamed: 0" column, as pandas writes by default
    result = CliRunner().invoke(
        mesoscopy.cli, args=f"process metrics {metrics_perievent_csv} -o {output_dir} -t {indexed}"
    )
    assert result.exit_code == 0, result.output

    trial = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_metrics.csv")
    assert not any(column.startswith("Unnamed") for column in trial.columns)
    assert trial[trial["region"] == "L_MOp"]["sdt_type"].tolist() == ["hit", "miss", "correct_rejection"]


def test_perievent_cmd_with_metrics(perievent_regions_csv, perievent_trials_csv, output_dir):
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=(
            f"process peri-event {perievent_regions_csv} {perievent_trials_csv} -o {output_dir} --with-metrics"
            " --baseline -0.5 0 --smooth 3 --onset peak"
        ),
    )
    assert result.exit_code == 0, result.output
    assert "Saved peri-event windows" in result.output
    assert "Saved per-trial metrics" in result.output

    windows = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_perievent.csv")
    trial = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_metrics.csv")
    session = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_metrics-session.csv")
    assert trial["trial_index"].tolist() == [1, 2, 3, 4] * len(PERIEVENT_REGIONS)
    assert trial["sdt_type"].tolist()[:4] == ["hit", "miss", "false_alarm", "correct_rejection"]
    assert session["region"].tolist() == PERIEVENT_REGIONS
    # The peri-event --baseline is the metrics baseline, so the baseline mean is already zero.
    np.testing.assert_allclose(trial["baseline_mean"], 0.0, atol=1e-6)

    # Same result as running `process metrics` on the written windows with the same options.
    expected, _ = pm.metrics_tables(
        windows, baseline=(-0.5, 0.0), smoothing=3, onset="peak", trials=pd.read_csv(perievent_trials_csv)
    )
    pd.testing.assert_frame_equal(trial, expected, check_dtype=False)


def test_perievent_cmd_with_metrics_rejects_h5(perievent_h5, perievent_trials_csv, output_dir):
    result = CliRunner().invoke(
        mesoscopy.cli, args=f"process peri-event {perievent_h5} {perievent_trials_csv} -o {output_dir} --with-metrics"
    )
    assert result.exit_code == 2
    assert "_regions.csv" in result.output
    assert not list(pathlib.Path(output_dir).glob("*_perievent.h5"))


def test_perievent_cmd_with_metrics_validates_options(perievent_regions_csv, perievent_trials_csv, output_dir):
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=f"process peri-event {perievent_regions_csv} {perievent_trials_csv} -o {output_dir} --with-metrics --smooth 2",
    )
    assert result.exit_code == 2
    assert "odd" in result.output


def test_perievent_cmd_metric_options_need_with_metrics(perievent_regions_csv, perievent_trials_csv, output_dir):
    """Metric options are accepted without --with-metrics and simply have no effect."""
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=f"process peri-event {perievent_regions_csv} {perievent_trials_csv} -o {output_dir} --smooth 2",
    )
    assert result.exit_code == 0, result.output
    assert not list(pathlib.Path(output_dir).glob("*_metrics*.csv"))


def test_metrics_cmd_missing_columns(metrics_perievent_csv, output_dir, tmp_path):
    broken = tmp_path / "ses-02_regions_event-cueonset_perievent.csv"
    pd.read_csv(metrics_perievent_csv).drop(columns=["region", "F"]).to_csv(broken, index=False)
    result = CliRunner().invoke(mesoscopy.cli, args=f"process metrics {broken} -o {output_dir}")
    assert result.exit_code != 0
    assert "F, region" in result.output


def test_perievent_cmd_with_metrics_no_trials_kept(perievent_regions_csv, perievent_trials_csv, output_dir, tmp_path):
    late = tmp_path / "ses-01_trials.csv"
    trials = pd.read_csv(perievent_trials_csv)
    trials[["start_time", "cue_onset", "stop_time"]] += 100.0  # every window runs past the recording
    trials.to_csv(late, index=False)
    result = CliRunner().invoke(
        mesoscopy.cli, args=f"process peri-event {perievent_regions_csv} {late} -o {output_dir} --with-metrics"
    )
    assert result.exit_code == 0, result.output
    assert "Kept 0 trials" in result.output
    assert "skipping metrics" in result.output

    windows = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_perievent.csv")
    assert windows.empty
    assert list(windows.columns[:5]) == ["trial_index", "event_time", "time", "region", "F"]
    assert "sdt_type" in windows.columns
    assert not list(pathlib.Path(output_dir).glob("*_metrics*.csv"))


def test_metrics_cmd_empty_input(metrics_perievent_csv, output_dir, tmp_path):
    empty = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    pd.read_csv(metrics_perievent_csv).iloc[:0].to_csv(empty, index=False)
    result = CliRunner().invoke(mesoscopy.cli, args=f"process metrics {empty} -o {output_dir}")
    assert result.exit_code == 1
    assert str(empty) in result.output
    assert "no rows" in result.output
    assert not list(pathlib.Path(output_dir).glob("*_metrics*.csv"))
