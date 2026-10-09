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
from sklearn.feature_selection import mutual_info_regression

import mesoscopy
from mesoscopy import io
from mesoscopy.process import connectivity as pc
from mesoscopy.process import decoding as pdc
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
            ("response", [1.7, 6.7, 21.3, 41.0], [0, 1, 4, 5]),
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


def _decay_time_loop(traces, time, peak_index, amplitude, end, fraction=0.5):
    """The per-trial loop `metrics.decay_time` replaced."""
    last = int(np.flatnonzero(time <= end)[-1])
    decay = np.full(traces.shape[0], np.nan)
    for i in np.flatnonzero((amplitude > 0) & (peak_index >= 0)):
        after = traces[i, peak_index[i] + 1 : last + 1]
        below = np.flatnonzero(after <= fraction * amplitude[i])
        if below.size:
            decay[i] = time[peak_index[i] + 1 + below[0]] - time[peak_index[i]]
    return decay


def _offset_time_loop(traces, time, threshold, peak_index, end, min_samples=3):
    """The per-trial loop `metrics.offset_time` replaced."""
    last = int(np.flatnonzero(time <= end)[-1])
    offset = np.full(traces.shape[0], np.nan)
    for i in np.flatnonzero(peak_index >= 0):
        if not traces[i, peak_index[i]] > threshold[i]:
            continue
        after = traces[i, peak_index[i] + 1 : last + 1]
        if after.size < min_samples:
            continue
        runs = np.lib.stride_tricks.sliding_window_view(after <= threshold[i], min_samples).all(axis=-1)
        if runs.any():
            offset[i] = time[peak_index[i] + 1 + int(np.argmax(runs))]
    return offset


def _extrapolated_onset_loop(traces, time, peak_index, amplitude, start, low=0.2, high=0.8):
    """The per-trial loop `metrics.extrapolated_onset` replaced, fitting with `np.polyfit`."""
    first = int(np.flatnonzero(time >= start)[0])
    onset = np.full(traces.shape[0], np.nan)
    for i in np.flatnonzero((amplitude > 0) & (peak_index >= 0)):
        rise = traces[i, first : peak_index[i] + 1]
        rise_time = time[first : peak_index[i] + 1]
        below = np.flatnonzero(rise < low * amplitude[i])
        begin = below[-1] + 1 if below.size else 0
        above = np.flatnonzero(rise[begin:] > high * amplitude[i])
        stop = begin + above[0] if above.size else len(rise)
        if stop - begin < 2:
            continue
        slope, intercept = np.polyfit(rise_time[begin:stop], rise[begin:stop], 1)
        if slope > 0:
            onset[i] = -intercept / slope
    return onset


def _loop_inputs(seed):
    """Noisy responses of random size, latency and width with NaN gaps, and peaks from `metrics.peak`.

    Rows 0-4 are all NaN, 5-9 negative, 10-14 have a random peak index anywhere in the window and 15-19 a NaN
    amplitude.
    """
    rng = np.random.default_rng(seed)
    t = METRICS_GRID
    n_trials = 300
    latency = rng.uniform(0.0, 1.5, (n_trials, 1))
    width = rng.uniform(0.1, 0.8, (n_trials, 1))
    traces = rng.uniform(0.2, 2.0, (n_trials, 1)) * np.exp(-0.5 * ((t - latency - width) / width) ** 2)
    traces += rng.uniform(0.01, 0.3, (n_trials, 1)) * rng.standard_normal(traces.shape)
    traces[rng.random(traces.shape) < 0.03] = np.nan
    traces[:5] = np.nan
    traces[5:10] *= -1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        amplitude, _, index = pm.peak(traces, t, 0.0, 3.0)
    index[10:15] = rng.integers(0, len(t), 5)
    amplitude[15:20] = np.nan
    return traces, t, index, amplitude


class TestMatchesLoops:
    """The array versions of the post-peak and rise searches give the per-trial loops' results."""

    @pytest.mark.parametrize("seed", range(5))
    def test_decay_time(self, seed):
        traces, t, index, amplitude = _loop_inputs(seed)
        found = {}
        for end, fraction in [(3.0, 0.5), (1.2, 0.1), (2.0, 0.9), (-1.0, 0.5)]:
            expected = _decay_time_loop(traces, t, index, amplitude, end, fraction)
            np.testing.assert_array_equal(pm.decay_time(traces, t, index, amplitude, end, fraction), expected)
            found[end, fraction] = np.isfinite(expected).sum()
        # Most traces decay over the whole window; many have no decay to 10% before 1.2 s, and none before -1 s.
        assert found[3.0, 0.5] > 250
        assert 20 < found[1.2, 0.1] < 250
        assert found[-1.0, 0.5] == 0

    @pytest.mark.parametrize("seed", range(5))
    def test_offset_time(self, seed):
        traces, t, index, amplitude = _loop_inputs(seed)
        threshold = 0.2 * amplitude
        threshold[20:25] = np.nan
        for end, min_samples in [(3.0, 3), (1.5, 1), (3.0, 7), (0.1, 3), (3.0, len(t) + 1)]:
            expected = _offset_time_loop(traces, t, threshold, index, end, min_samples)
            np.testing.assert_array_equal(pm.offset_time(traces, t, threshold, index, end, min_samples), expected)
        assert 100 < np.isfinite(_offset_time_loop(traces, t, threshold, index, 3.0)).sum() < len(traces) - 20

    @pytest.mark.parametrize("seed", range(5))
    def test_extrapolated_onset(self, seed):
        traces, t, index, amplitude = _loop_inputs(seed)
        for start, low, high in [(0.0, 0.2, 0.8), (0.5, 0.1, 0.5), (-0.5, 0.3, 0.9)]:
            expected = _extrapolated_onset_loop(traces, t, index, amplitude, start, low, high)
            np.testing.assert_allclose(
                pm.extrapolated_onset(traces, t, index, amplitude, start, low, high), expected, rtol=1e-9, atol=1e-9
            )
        assert 100 < np.isfinite(_extrapolated_onset_loop(traces, t, index, amplitude, 0.0)).sum() < len(traces) - 20

    def test_no_trials(self):
        t = METRICS_GRID
        traces, index, values = np.empty((0, len(t))), np.empty(0, dtype=np.int64), np.empty(0)
        assert pm.decay_time(traces, t, index, values, 3.0).shape == (0,)
        assert pm.offset_time(traces, t, values, index, 3.0).shape == (0,)
        assert pm.extrapolated_onset(traces, t, index, values, 0.0).shape == (0,)


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
        tables = pm.metrics_tables(pd.read_csv(metrics_perievent_csv))
        per_trial, per_session = tables.per_trial, tables.per_session
        assert list(per_trial.columns[:3]) == ["trial_index", "event_time", "region"]
        assert len(per_trial) == 3 * len(PERIEVENT_REGIONS)
        assert per_session["region"].tolist() == PERIEVENT_REGIONS
        assert per_session["group"].tolist() == ["all", "all"]
        assert per_session["n_trials"].tolist() == [3, 3]
        first = per_trial.iloc[0]
        assert first["onset_time"] == pytest.approx(0.24)
        assert first["amplitude"] == pytest.approx(1.0, abs=1e-3)

    def test_joins_trials(self, metrics_perievent_csv, perievent_trials_csv):
        per_trial = pm.metrics_tables(
            pd.read_csv(metrics_perievent_csv), trials=pd.read_csv(perievent_trials_csv)
        ).per_trial
        assert per_trial[per_trial["region"] == "L_MOp"]["sdt_type"].tolist() == ["hit", "miss", "correct_rejection"]

    def test_carries_trial_columns_through(self, metrics_perievent_csv, perievent_trials_csv):
        perievent = pm.join_trials(pd.read_csv(metrics_perievent_csv), pd.read_csv(perievent_trials_csv))
        tables = pm.metrics_tables(perievent)
        per_trial, per_session = tables.per_trial, tables.per_session
        plain = pm.metrics_tables(pd.read_csv(metrics_perievent_csv)).per_trial
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
        with_trials = pm.metrics_tables(perievent, trials=trials).per_trial
        without = pm.metrics_tables(perievent).per_trial
        pd.testing.assert_frame_equal(with_trials, without)

    def test_trials_clashing_column_gets_suffix(self, metrics_perievent_csv, perievent_trials_csv):
        trials = pd.read_csv(perievent_trials_csv)
        trials["duration"] = 9.0
        per_trial = pm.metrics_tables(pd.read_csv(metrics_perievent_csv), trials=trials).per_trial
        assert "duration_trial" in per_trial.columns
        assert per_trial["duration_trial"].eq(9.0).all()
        assert not per_trial["duration"].eq(9.0).any()

    def test_options_pass_through(self, metrics_perievent_csv):
        per_trial = pm.metrics_tables(
            pd.read_csv(metrics_perievent_csv),
            baseline=(-0.5, 0.0),
            response=(0.0, 1.2),
            onset="peak",
            onset_fraction=0.53,
        ).per_trial
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
    # The bootstrap runs by default; three trials are below the minimum, so the mean traces file has a header only.
    boot = _read_boot(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_metrics-boot.csv")
    assert boot["n_trials"].tolist() == [3, 3]
    assert pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_traces-boot.csv").empty

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
    assert list(session.columns[:4]) == ["region", "group", "n_trials", "trace_correlation"]
    assert session["region"].tolist() == PERIEVENT_REGIONS
    # Without sdt_type there are no trial groups.
    assert session["group"].tolist() == ["all", "all"]
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


@pytest.mark.parametrize(
    ("option", "hint"),
    [
        ("--smooth 4", "odd"),
        ("--extrapolate-range 0.8 0.2", "LOW < HIGH"),
        ("--min-trials 0", "--min-trials"),
        ("--bootstrap -1", "--bootstrap"),
        ("--ci 100", "--ci"),
    ],
)
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


def test_metrics_cmd_carries_trial_columns(metrics_perievent_csv, perievent_trials_csv, output_dir, tmp_path):
    trials = pd.read_csv(perievent_trials_csv)
    trials["correction"] = [True, False, False, True, False, False]
    trials["response"] = np.nan  # all-empty column, as visiomode writes for no-response sessions
    joined = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    pm.join_trials(pd.read_csv(metrics_perievent_csv), trials).to_csv(joined, index=False)
    result = CliRunner().invoke(mesoscopy.cli, args=f"process metrics {joined} -o {output_dir}")
    assert result.exit_code == 0, result.output

    trial = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_metrics.csv")
    assert list(trial.columns[-7:]) == [
        "start_time",
        "cue_onset",
        "stop_time",
        "response_time",
        "sdt_type",
        "correction",
        "response",
    ]
    assert trial.columns.get_loc("start_time") > trial.columns.get_loc("duration")
    joined_trial = trial[trial["region"] == "L_MOp"].set_index("trial_index")
    assert joined_trial["sdt_type"].tolist() == ["hit", "miss", "correct_rejection"]
    assert joined_trial["correction"].tolist() == [False, False, False]
    assert joined_trial["response"].isna().all()
    np.testing.assert_allclose(joined_trial["cue_onset"], [5.5, 10.5, 20.5])
    session = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_metrics-session.csv")
    assert "sdt_type" not in session.columns


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
    assert session[session["group"] == "all"]["region"].tolist() == PERIEVENT_REGIONS
    assert session[session["group"] == "resp-push"]["n_trials"].tolist() == [2, 2]
    # The peri-event --baseline is the metrics baseline, so the baseline mean is already zero.
    np.testing.assert_allclose(trial["baseline_mean"], 0.0, atol=1e-6)

    # Same result as running `process metrics` on the written windows with the same options.
    expected = pm.metrics_tables(
        windows, baseline=(-0.5, 0.0), smoothing=3, onset="peak", trials=pd.read_csv(perievent_trials_csv)
    ).per_trial
    pd.testing.assert_frame_equal(trial, expected, check_dtype=False)


def test_perievent_cmd_with_metrics_min_trials(perievent_regions_csv, perievent_trials_csv, output_dir):
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=(
            f"process peri-event {perievent_regions_csv} {perievent_trials_csv} -o {output_dir} --with-metrics"
            " --min-trials 2"
        ),
    )
    assert result.exit_code == 0, result.output
    session = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_metrics-session.csv")
    push = session[session["group"] == "resp-push"]
    # The two lever pushes meet the lowered minimum.
    assert push["n_trials"].tolist() == [2, 2]
    assert push["amplitude_mean"].notna().all()


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


# ---------------------------------------------------------------------------
# reliability metrics
# ---------------------------------------------------------------------------

GONOGO_TYPES = ["hit"] * 12 + ["miss"] * 6 + ["false_alarm"] * 12 + ["correct_rejection"] * 6
GONOGO_RT = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4] * 2 + [np.nan] * 6 + [0.5, 0.7, 0.9, 1.1, 1.3, 1.5] * 2 + [np.nan] * 6


def _gonogo_info(event="cue_onset"):
    """One row per trial of a go/no-go session with a 5 s ITI; `response_time` is seconds from the cue."""
    n_trials = len(GONOGO_TYPES)
    cue = 10.0 * np.arange(n_trials) + 5.0
    response_time = np.asarray(GONOGO_RT)
    event_time = {"cue_onset": cue, "trial_start": cue - 5.0, "response": cue + response_time}[event]
    keep = ~np.isnan(event_time)
    return pd.DataFrame(
        {
            "trial_index": np.arange(n_trials),
            "event_time": event_time,
            "start_time": cue - 5.0,
            "cue_onset": cue,
            "stop_time": np.where(np.isnan(response_time), cue + 2.0, cue + response_time),
            "response_time": response_time,
            "sdt_type": GONOGO_TYPES,
            "protocol": "gonogo",
        }
    )[keep].reset_index(drop=True)


def _gonogo_traces(n_trials, noise=0.1, seed=0):
    """A shared response rising after the event, plus independent Gaussian noise per trial."""
    rng = np.random.default_rng(seed)
    shape = np.interp(METRICS_GRID, [0.0, 0.5, 1.5], [0.0, 1.0, 0.0])
    return shape + noise * rng.standard_normal((n_trials, len(METRICS_GRID)))


def _gonogo_perievent(event="cue_onset"):
    """Long-format peri-event table of the go/no-go session for two regions, the second pure noise."""
    info = _gonogo_info(event)
    traces = _gonogo_traces(len(info))
    noise = np.random.default_rng(1).standard_normal(traces.shape)
    frames = []
    for i, row in info.iterrows():
        for region, values in zip(PERIEVENT_REGIONS, (traces[i], noise[i]), strict=True):
            frames.append(
                pd.DataFrame(
                    {
                        "trial_index": row["trial_index"],
                        "event_time": row["event_time"],
                        "time": METRICS_GRID,
                        "region": region,
                        "F": values,
                    }
                )
            )
    table = pd.concat(frames, ignore_index=True)
    return table.merge(info.drop(columns="event_time"), on="trial_index")


class TestInferEvent:
    @pytest.mark.parametrize("event", ["cue_onset", "trial_start", "response"])
    def test_matches_event(self, event):
        assert pm.infer_event(_gonogo_info(event)) == event

    def test_unknown(self):
        info = _gonogo_info()
        info["event_time"] += 0.3
        assert pm.infer_event(info) is None

    def test_missing_columns(self):
        assert pm.infer_event(_gonogo_info().drop(columns=["cue_onset", "start_time", "stop_time"])) is None


class TestReliabilityEpochs:
    def test_cue_masked_at_response(self):
        start, end, cue, keep = pm.reliability_epochs(_gonogo_info(), "cue_onset", (0.0, 3.0))
        np.testing.assert_allclose(start, 0.0)
        np.testing.assert_allclose(cue, 0.0)
        np.testing.assert_allclose(end[:12], GONOGO_RT[:12])
        # Trials without a response end at the median of the others.
        np.testing.assert_allclose(end[12:18], np.nanmedian(GONOGO_RT))
        assert keep.all()

    def test_trial_start_offsets_by_iti(self):
        _, end, cue, _ = pm.reliability_epochs(_gonogo_info("trial_start"), "trial_start", (0.0, 10.0))
        np.testing.assert_allclose(cue, 5.0)
        np.testing.assert_allclose(end[:12], 5.0 + np.asarray(GONOGO_RT[:12]))

    def test_capped_at_response_window(self):
        _, end, _, _ = pm.reliability_epochs(_gonogo_info(), "cue_onset", (0.0, 0.7))
        assert end.max() == pytest.approx(0.7)
        assert end.max() > 0.7

    def test_unmasked_runs_over_window(self):
        _, end, _, _ = pm.reliability_epochs(_gonogo_info(), "cue_onset", (0.0, 3.0), mask_response=False)
        assert (end > 3.0).all()
        assert (end < 3.0 + 1e-9).all()

    def test_min_rt_drops_fast_trials(self):
        _, _, _, keep = pm.reliability_epochs(_gonogo_info(), "cue_onset", (0.0, 3.0), min_rt=0.65)
        rt = np.asarray(GONOGO_RT)
        np.testing.assert_array_equal(keep, ~(rt < 0.65))

    def test_lever_runs_from_cue(self):
        info = _gonogo_info("response")
        start, end, cue, keep = pm.reliability_epochs(info, "response", (0.0, 3.0))
        np.testing.assert_allclose(start, -info["response_time"])
        np.testing.assert_allclose(cue, -info["response_time"])
        np.testing.assert_allclose(end, 0.0)
        assert keep.all()

    def test_reward_raises(self):
        with pytest.raises(ValueError, match="not defined"):
            pm.reliability_epochs(_gonogo_info(), "reward", (0.0, 3.0))


class TestKeptTrials:
    def test_drops_fast_responses(self):
        rt = np.asarray(GONOGO_RT)
        np.testing.assert_array_equal(pm.kept_trials(_gonogo_info(), 0.65), ~(rt < 0.65))

    def test_without_response_time_keeps_all(self):
        assert pm.kept_trials(_gonogo_info().drop(columns="response_time"), 10.0).all()


class TestEpochTraces:
    def test_masks_outside_epoch(self):
        t = np.array([-0.5, 0.0, 0.5, 1.0])
        epochs = pm.epoch_traces(np.ones((2, 4)), t, np.array([0.0, -0.5]), np.array([1.0, np.nan]))
        np.testing.assert_array_equal(epochs[0], [np.nan, 1.0, 1.0, np.nan])
        assert np.isnan(epochs[1]).all()


class TestAnchoredWindow:
    def test_interpolates_relative_to_anchor(self):
        t = np.arange(-2.0, 1.01, 0.5)
        window = pm.anchored_window(t[None, :].repeat(2, axis=0), t, np.array([0.0, -0.75]), -1.0, 0.0)
        np.testing.assert_allclose(window[0], [-1.0, -0.5])
        np.testing.assert_allclose(window[1], [-1.75, -1.25])

    def test_outside_window_is_nan(self):
        t = np.arange(-2.0, 1.01, 0.5)
        window = pm.anchored_window(np.ones((2, len(t))), t, np.array([-1.5, np.nan]), -1.0, 0.0)
        assert np.isnan(window).all()


class TestWarpEpochs:
    def test_linear_epochs_share_grid(self):
        t = METRICS_GRID
        start, end = np.array([0.0, 0.0]), np.array([1.0, 2.0])
        epochs = pm.epoch_traces(np.stack([t, t / 2]), t, start, end)
        warped = pm.warp_epochs(epochs, t, start, end, 11)
        # Both rise from 0 towards 1 over their own epoch; the last grid point is past the last sample.
        np.testing.assert_allclose(warped[:, :10], np.linspace(0, 1, 11)[None, :10].repeat(2, axis=0), atol=1e-9)
        assert np.isnan(warped[:, -1]).all()

    def test_too_short_is_nan(self):
        t = METRICS_GRID
        epochs = pm.epoch_traces(np.ones((1, len(t))), t, np.array([0.0]), np.array([0.01]))
        assert np.isnan(pm.warp_epochs(epochs, t, np.array([0.0]), np.array([0.01]), 5)).all()


class TestReliabilityMeasures:
    def test_identical_trials(self):
        epochs = _gonogo_traces(20, noise=0.0)
        assert pm.epoch_correlation(epochs) == pytest.approx(1.0)
        assert pm.signal_fraction(epochs) == pytest.approx(1.0)

    def test_pure_noise(self):
        epochs = np.random.default_rng(0).standard_normal((60, 100))
        assert abs(pm.epoch_correlation(epochs)) < 0.02
        assert pm.signal_fraction(epochs) == pytest.approx(1 / 60, abs=0.01)

    def test_epoch_correlation_uses_shared_samples(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
        b = np.array([np.nan, 4.0, 6.0, 8.0, 1.0])
        assert pm.epoch_correlation(np.stack([a, b])) == pytest.approx(1.0)
        assert np.isnan(pm.epoch_correlation(np.stack([a, b]), min_samples=4))

    def test_response_fraction(self):
        baseline = np.tile([-0.1, 0.1, -0.1, 0.1], (4, 1))  # SD about 0.115
        epochs = np.array([[1.0, 1.0], [0.1, 0.1], [0.5, np.nan], [np.nan, np.nan]])
        assert pm.response_fraction(epochs, baseline, 2.0) == pytest.approx(2 / 3)

    def test_variance_quench(self):
        rng = np.random.default_rng(0)
        baseline = rng.standard_normal((200, 25))
        epochs = 0.5 * rng.standard_normal((200, 25))
        assert pm.variance_quench(epochs, baseline) == pytest.approx(0.25, rel=0.1)
        assert np.isnan(pm.variance_quench(epochs, np.zeros((200, 25))))


class TestTrialGroups:
    def test_groups(self):
        groups = pm.trial_groups(_gonogo_info())
        assert list(groups) == [
            "all",
            "sdt-hit",
            "sdt-miss",
            "sdt-false_alarm",
            "sdt-correct_rejection",
            "stim-go",
            "stim-nogo",
            "resp-push",
            "resp-nopush",
        ]
        assert groups["all"].sum() == 36
        assert groups["sdt-hit"].sum() == 12
        assert groups["stim-go"].sum() == 18
        assert groups["stim-nogo"].sum() == 18
        # Hits and false alarms push the lever; misses and correct rejections do not.
        np.testing.assert_array_equal(groups["resp-push"], groups["sdt-hit"] | groups["sdt-false_alarm"])
        np.testing.assert_array_equal(groups["resp-nopush"], groups["sdt-miss"] | groups["sdt-correct_rejection"])
        assert groups["resp-push"].sum() == 24
        assert groups["resp-nopush"].sum() == 12


class TestReliabilityMetrics:
    def test_groups_and_min_trials(self):
        info = _gonogo_info()
        traces = _gonogo_traces(len(info))
        args = (traces, METRICS_GRID, info, "cue_onset", (-1.0, 0.0), (0.0, 3.0))
        summary = pm.reliability_metrics(*args)
        assert list(summary) == list(pm.trial_groups(info))
        assert all(list(values) == list(pm.RELIABILITY_METRICS) for values in summary.values())
        assert summary["all"]["reliability_n"] == 36
        assert summary["sdt-miss"]["reliability_n"] == 6
        assert summary["resp-push"]["reliability_n"] == 24
        assert summary["resp-nopush"]["reliability_n"] == 12
        # Six misses are below the default 10-trial minimum.
        assert np.isnan(summary["sdt-miss"]["epoch_correlation"])
        assert summary["all"]["epoch_correlation"] > 0.5
        assert summary["sdt-hit"]["epoch_correlation"] > 0.5
        assert 0 < summary["all"]["signal_fraction"] <= 1
        lowered = pm.reliability_metrics(*args, min_trials=5)
        assert np.isfinite(lowered["sdt-miss"]["epoch_correlation"])
        assert lowered["all"]["epoch_correlation"] == summary["all"]["epoch_correlation"]

    def test_masking_changes_epochs(self):
        info = _gonogo_info()
        traces = _gonogo_traces(len(info))
        args = (traces, METRICS_GRID, info, "cue_onset", (-1.0, 0.0), (0.0, 3.0))
        masked = pm.reliability_metrics(*args)["all"]
        unmasked = pm.reliability_metrics(*args, pm.ReliabilityOptions(mask_response=False))["all"]
        assert masked["epoch_correlation"] != unmasked["epoch_correlation"]
        # Unmasked over the whole response window, epoch correlation is the trace correlation.
        assert unmasked["epoch_correlation"] == pytest.approx(pm.trace_correlation(traces, METRICS_GRID, 0.0, 3.0))

    def test_min_rt_excludes_trials(self):
        info = _gonogo_info()
        options = pm.ReliabilityOptions(min_rt=0.65)
        summary = pm.reliability_metrics(
            _gonogo_traces(len(info)), METRICS_GRID, info, "cue_onset", (-1.0, 0.0), (0.0, 3.0), options
        )
        assert summary["sdt-hit"]["reliability_n"] == 8
        # Two false alarms are also faster than 0.65 s.
        assert summary["all"]["reliability_n"] == 36 - 6

    def test_time_warp(self):
        info = _gonogo_info()
        summary = pm.reliability_metrics(
            _gonogo_traces(len(info)),
            METRICS_GRID,
            info,
            "cue_onset",
            (-1.0, 0.0),
            (0.0, 3.0),
            pm.ReliabilityOptions(time_warp=True),
        )
        assert np.isfinite(summary["all"]["epoch_correlation"])

    def test_lever_baseline_before_cue(self):
        info = _gonogo_info("response")
        traces = _gonogo_traces(len(info))
        summary = pm.reliability_metrics(
            traces,
            METRICS_GRID,
            info,
            "response",
            (-1.0, 0.0),
            (0.0, 3.0),
            pm.ReliabilityOptions(cue_baseline=(-0.5, 0.0)),
        )["all"]
        assert summary["reliability_n"] == 24
        assert np.isfinite(summary["variance_quench"])
        # With a baseline reaching before the window, every trial's baseline is NaN.
        uncovered = pm.reliability_metrics(
            traces,
            METRICS_GRID,
            info,
            "response",
            (-1.0, 0.0),
            (0.0, 3.0),
            pm.ReliabilityOptions(cue_baseline=(-2.0, 0.0)),
        )["all"]
        assert np.isnan(uncovered["variance_quench"])
        assert np.isnan(uncovered["response_fraction"])
        assert uncovered["epoch_correlation"] == pytest.approx(summary["epoch_correlation"])


class TestMetricsTablesReliability:
    def test_gonogo_session(self):
        per_session = pm.metrics_tables(_gonogo_perievent()).per_session
        groups = list(pm.trial_groups(_gonogo_info()))
        assert per_session["region"].tolist() == [region for region in PERIEVENT_REGIONS for _ in groups]
        assert per_session["group"].tolist() == groups * len(PERIEVENT_REGIONS)
        assert list(per_session.columns[:4]) == ["region", "group", "n_trials", "trace_correlation"]
        assert list(per_session.columns[-5:]) == list(pm.RELIABILITY_METRICS)
        signal, noise = per_session[per_session["group"] == "all"]["epoch_correlation"]
        assert signal > 0.5
        assert abs(noise) < 0.1

    def test_reliability_only_adds_columns(self):
        perievent = _gonogo_perievent()
        with_reliability = pm.metrics_tables(perievent).per_session
        with pytest.warns(UserWarning, match="Skipping reliability"):
            without = pm.metrics_tables(perievent.drop(columns="response_time")).per_session
        pd.testing.assert_frame_equal(with_reliability[without.columns], without)

    def test_group_summaries(self):
        per_session = pm.metrics_tables(_gonogo_perievent()).per_session
        info = _gonogo_info()
        traces = _gonogo_traces(len(info))
        hits = (info["sdt_type"] == "hit").to_numpy()
        metrics = pm.trial_metrics(traces[hits], METRICS_GRID, (-1.0, 0.0), (0.0, 3.0))
        expected = pm.session_metrics(metrics, pm.trace_correlation(traces[hits], METRICS_GRID, 0.0, 3.0))
        row = per_session[(per_session["region"] == PERIEVENT_REGIONS[0]) & (per_session["group"] == "sdt-hit")]
        assert row.iloc[0][list(expected)].tolist() == pytest.approx(list(expected.values()), nan_ok=True)

    def test_min_trials(self):
        perievent = _gonogo_perievent()
        per_session = pm.metrics_tables(perievent).per_session
        misses = per_session[per_session["group"] == "sdt-miss"]
        # Six misses are below the default 10-trial minimum: the counts stay, the estimates are empty.
        assert misses["n_trials"].tolist() == [6, 6]
        assert misses["onset_n"].notna().all()
        assert misses[["trace_correlation", "amplitude_mean", "amplitude_sd", "epoch_correlation"]].isna().all().all()

        lowered = pm.metrics_tables(perievent, min_trials=5).per_session
        assert lowered[lowered["group"] == "sdt-miss"][["amplitude_mean", "epoch_correlation"]].notna().all().all()

        # The all-trials summaries are always taken; their reliability metrics follow the minimum.
        strict = pm.metrics_tables(perievent, min_trials=100).per_session
        everything = strict[strict["group"] == "all"]
        pd.testing.assert_frame_equal(
            everything[["n_trials", "trace_correlation", "amplitude_mean"]],
            per_session[per_session["group"] == "all"][["n_trials", "trace_correlation", "amplitude_mean"]],
        )
        assert everything["epoch_correlation"].isna().all()

    def test_without_sdt_type_only_all_trials(self):
        perievent = _gonogo_perievent()
        grouped = pm.metrics_tables(perievent).per_session
        with (
            pytest.warns(UserWarning, match="Skipping reliability"),
            pytest.warns(UserWarning, match="Skipping trial groups: no sdt_type column"),
        ):
            plain = pm.metrics_tables(perievent.drop(columns="sdt_type")).per_session
        assert plain["region"].tolist() == PERIEVENT_REGIONS
        assert plain["group"].tolist() == ["all", "all"]
        everything = grouped[grouped["group"] == "all"].reset_index(drop=True)
        pd.testing.assert_frame_equal(everything[plain.columns], plain)

    def test_lever_aligned(self):
        per_session = pm.metrics_tables(_gonogo_perievent("response"), baseline=(-1.0, 0.0)).per_session
        by_group = per_session.set_index(["group", "region"])
        assert by_group.loc["all", "reliability_n"].tolist() == [24, 24]
        assert by_group.loc["sdt-miss", "reliability_n"].tolist() == [0, 0]
        # Only trials with a lever push have a lever-aligned window.
        assert by_group.loc["resp-push", "n_trials"].tolist() == [24, 24]
        assert by_group.loc["resp-nopush", "n_trials"].tolist() == [0, 0]

    def test_lever_baseline_outside_window_warns(self):
        with pytest.warns(UserWarning, match="pre-cue baseline outside the window"):
            pm.metrics_tables(_gonogo_perievent("response"), reliability=pm.ReliabilityOptions(cue_baseline=(-2, 0)))

    def test_trials_supply_columns(self):
        perievent = _gonogo_perievent()
        trials = _gonogo_info().drop(columns=["trial_index", "event_time"])
        bare = perievent[["trial_index", "event_time", "time", "region", "F"]]
        joined = pm.metrics_tables(bare, trials=trials).per_session
        carried = pm.metrics_tables(perievent).per_session
        pd.testing.assert_frame_equal(joined, carried)

    @pytest.mark.parametrize(
        ("change", "match"),
        [
            (lambda table: table.drop(columns="response_time"), "no response_time column"),
            (lambda table: table.assign(protocol="2afc"), "only go/no-go"),
            (lambda table: table.assign(event_time=table["event_time"] + 0.3), "aligned event is unknown"),
        ],
    )
    def test_skipped(self, change, match):
        with pytest.warns(UserWarning, match=match):
            per_session = pm.metrics_tables(change(_gonogo_perievent())).per_session
        assert "epoch_correlation" not in per_session.columns

    def test_reward_skipped(self):
        with pytest.warns(UserWarning, match="not defined for reward"):
            per_session = pm.metrics_tables(_gonogo_perievent(), event="reward").per_session
        assert "epoch_correlation" not in per_session.columns
        # Trial groups do not depend on the aligned event.
        assert per_session["group"].unique().tolist() == list(pm.trial_groups(_gonogo_info()))

    def test_groups_skipped_for_other_protocols(self):
        with (
            pytest.warns(UserWarning, match="Skipping reliability metrics: only go/no-go"),
            pytest.warns(UserWarning, match="Skipping trial groups: only go/no-go"),
        ):
            per_session = pm.metrics_tables(_gonogo_perievent().assign(protocol="2afc")).per_session
        assert per_session["group"].tolist() == ["all", "all"]


def test_metrics_cmd_reliability(output_dir, tmp_path):
    path = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    _gonogo_perievent().to_csv(path, index=False)
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=f"process metrics {path} -o {output_dir} --no-mask-response --min-rt 0.3 --response-sd 1"
        " --time-warp --cue-baseline -0.5 0",
    )
    assert result.exit_code == 0, result.output
    session = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_metrics-session.csv")
    expected = pm.metrics_tables(
        pd.read_csv(path),
        reliability=pm.ReliabilityOptions(
            mask_response=False,
            min_rt=0.3,
            response_sd=1.0,
            time_warp=True,
            cue_baseline=(-0.5, 0.0),
        ),
    ).per_session
    pd.testing.assert_frame_equal(session, expected, check_dtype=False)
    assert session[session["group"] == "all"]["reliability_n"].tolist() == [36, 36]


def test_metrics_cmd_min_trials(output_dir, tmp_path):
    path = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    _gonogo_perievent().to_csv(path, index=False)
    result = CliRunner().invoke(mesoscopy.cli, args=f"process metrics {path} -o {output_dir} --min-trials 5")
    assert result.exit_code == 0, result.output
    session = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_metrics-session.csv")
    expected = pm.metrics_tables(pd.read_csv(path), min_trials=5).per_session
    pd.testing.assert_frame_equal(session, expected, check_dtype=False)
    # Six misses meet the lowered minimum.
    assert session[session["group"] == "sdt-miss"]["amplitude_mean"].notna().all()


def test_metrics_cmd_reliability_skipped_warns(metrics_perievent_csv, output_dir):
    result = CliRunner().invoke(mesoscopy.cli, args=f"process metrics {metrics_perievent_csv} -o {output_dir}")
    assert result.exit_code == 0, result.output
    assert "Warning: Skipping trial groups: no sdt_type column" in result.output
    assert "Warning: Skipping reliability metrics: no cue_onset, response_time, sdt_type column(s)" in result.output


def test_perievent_cmd_reliability_options_rejected(perievent_regions_csv, perievent_trials_csv, output_dir):
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=f"process peri-event {perievent_regions_csv} {perievent_trials_csv} -o {output_dir} --min-rt -1",
    )
    assert result.exit_code == 2
    assert "--min-rt" in result.output


class TestBootstrapOptions:
    @pytest.mark.parametrize(
        ("options", "match"), [({"n_resamples": -1}, "n_resamples"), ({"ci": 0}, "ci"), ({"ci": 100}, "ci")]
    )
    def test_rejects_bad_values(self, options, match):
        with pytest.raises(ValueError, match=match):
            pm.BootstrapOptions(**options)


class TestResampleCounts:
    def test_counts_draws_with_replacement(self):
        counts = pm.resample_counts(7, 500, seed=1)
        assert counts.shape == (500, 7)
        np.testing.assert_array_equal(counts.sum(axis=1), 7)
        draws = np.random.default_rng(1).integers(0, 7, size=(500, 7))
        np.testing.assert_array_equal(counts, [np.bincount(row, minlength=7) for row in draws])

    def test_seed(self):
        np.testing.assert_array_equal(pm.resample_counts(10, 50, seed=3), pm.resample_counts(10, 50, seed=3))
        assert not np.array_equal(pm.resample_counts(10, 50, seed=3), pm.resample_counts(10, 50, seed=4))


def _resampled(traces, draws):
    """Mean of the baseline-subtracted traces drawn into each resample, ignoring NaNs."""
    baseline_mean, _ = pm.baseline_stats(traces, METRICS_GRID, -1.0, 0.0)
    return np.nanmean((traces - baseline_mean[:, None])[draws], axis=1)


class TestBootstrapMetrics:
    def test_estimates_from_mean_trace(self):
        traces = _gonogo_traces(36)
        counts = pm.resample_counts(36, 500, seed=0)
        summary, mean, _, _ = pm.bootstrap_metrics(traces, METRICS_GRID, (-1.0, 0.0), (0.0, 3.0), counts)
        assert list(summary) == list(pm.BOOT_COLUMNS)
        baseline_mean, _ = pm.baseline_stats(traces, METRICS_GRID, -1.0, 0.0)
        np.testing.assert_allclose(mean, (traces - baseline_mean[:, None]).mean(axis=0))
        estimates = pm.trial_metrics(mean[None], METRICS_GRID, (-1.0, 0.0), (0.0, 3.0))
        assert [summary[f"boot_{name}"] for name in pm.METRICS] == pytest.approx(
            [estimates[name].iloc[0] for name in pm.METRICS], nan_ok=True
        )

    def test_intervals_from_resampled_means(self):
        traces = _gonogo_traces(36)
        args = (traces, METRICS_GRID, (-1.0, 0.0), (0.0, 3.0), pm.resample_counts(36, 500, seed=0))
        summary, _, low, high = pm.bootstrap_metrics(*args, ci=90, onset="peak", smoothing=3)
        resampled = _resampled(traces, np.random.default_rng(0).integers(0, 36, size=(500, 36)))
        np.testing.assert_allclose(low, np.percentile(resampled, 5, axis=0))
        np.testing.assert_allclose(high, np.percentile(resampled, 95, axis=0))
        values = pm.trial_metrics(resampled, METRICS_GRID, (-1.0, 0.0), (0.0, 3.0), onset="peak", smoothing=3)
        for name in pm.METRICS:
            defined = values[name].dropna()
            assert summary[f"boot_{name}_n"] == len(defined)
            assert summary[f"boot_{name}_ci_low"] == pytest.approx(np.percentile(defined, 5))
            assert summary[f"boot_{name}_ci_high"] == pytest.approx(np.percentile(defined, 95))

    def test_nan_samples_are_ignored(self):
        traces = _gonogo_traces(20)
        traces[3, 40:50] = np.nan
        traces[7, 60:] = np.nan
        args = (traces, METRICS_GRID, (-1.0, 0.0), (0.0, 3.0), pm.resample_counts(20, 300, seed=5))
        summary, mean, low, high = pm.bootstrap_metrics(*args)
        resampled = _resampled(traces, np.random.default_rng(5).integers(0, 20, size=(300, 20)))
        np.testing.assert_allclose(mean, _resampled(traces, np.arange(20)[None])[0])
        np.testing.assert_allclose(low, np.percentile(resampled, 2.5, axis=0))
        np.testing.assert_allclose(high, np.percentile(resampled, 97.5, axis=0))
        assert summary["boot_amplitude_n"] == 300

    def test_auc_interval_is_bootstrap_of_trial_mean(self):
        # The area is linear in the trace, so the mean trace's area is the mean of the per-trial areas.
        traces = _gonogo_traces(36, noise=0.3, seed=2)
        counts = pm.resample_counts(36, 2000, seed=0)
        summary, *_ = pm.bootstrap_metrics(traces, METRICS_GRID, (-1.0, 0.0), (0.0, 3.0), counts)
        areas = pm.trial_metrics(traces, METRICS_GRID, (-1.0, 0.0), (0.0, 3.0))["auc"].to_numpy()
        assert summary["boot_auc"] == pytest.approx(areas.mean())
        low, high = np.percentile(counts @ areas / 36, [2.5, 97.5])
        assert summary["boot_auc_ci_low"] == pytest.approx(low)
        assert summary["boot_auc_ci_high"] == pytest.approx(high)
        assert low < areas.mean() < high

    @pytest.mark.parametrize("ci", [0, 100])
    def test_bad_ci_raises(self, ci):
        with pytest.raises(ValueError, match="ci"):
            pm.bootstrap_metrics(_gonogo_traces(5), METRICS_GRID, (-1.0, 0.0), (0.0, 3.0), np.ones((1, 5)), ci)


class TestMetricsTablesBootstrap:
    def test_tables(self):
        tables = pm.metrics_tables(_gonogo_perievent(), bootstrap=pm.BootstrapOptions(n_resamples=500))
        groups = list(pm.trial_groups(_gonogo_info()))
        boot = tables.boot
        assert list(boot.columns) == ["region", "group", "n_trials", *pm.BOOT_COLUMNS]
        assert boot["group"].tolist() == groups * len(PERIEVENT_REGIONS)
        assert boot["n_trials"].tolist() == tables.per_session["n_trials"].tolist()
        # Six misses and six correct rejections are below the default 10-trial minimum: only the count stays.
        small = boot["group"].isin(["sdt-miss", "sdt-correct_rejection"])
        assert boot.loc[small, list(pm.BOOT_COLUMNS)].isna().all().all()
        assert boot.loc[~small, "boot_amplitude_n"].eq(500).all()
        traces = tables.boot_traces
        assert list(traces.columns) == ["region", "group", "n_trials", "time", "mean", "ci_low", "ci_high"]
        assert len(traces) == len(PERIEVENT_REGIONS) * (len(groups) - 2) * len(METRICS_GRID)
        assert not traces["group"].isin(["sdt-miss", "sdt-correct_rejection"]).any()

    def test_matches_bootstrap_metrics(self):
        options = pm.BootstrapOptions(n_resamples=300, ci=90, seed=7)
        tables = pm.metrics_tables(_gonogo_perievent(), smoothing=3, onset="peak", bootstrap=options)
        info = _gonogo_info()
        hits = (info["sdt_type"] == "hit").to_numpy()
        summary, mean, low, high = pm.bootstrap_metrics(
            _gonogo_traces(len(info))[hits],
            METRICS_GRID,
            (-1.0, 0.0),
            (0.0, 3.0),
            pm.resample_counts(int(hits.sum()), 300, seed=7),
            ci=90,
            smoothing=3,
            onset="peak",
        )
        boot = tables.boot.set_index(["region", "group"]).loc[(PERIEVENT_REGIONS[0], "sdt-hit")]
        assert boot[list(summary)].tolist() == pytest.approx(list(summary.values()))
        band = tables.boot_traces.set_index(["region", "group"]).loc[(PERIEVENT_REGIONS[0], "sdt-hit")]
        np.testing.assert_allclose(band["time"], METRICS_GRID)
        np.testing.assert_allclose(band[["mean", "ci_low", "ci_high"]].to_numpy().T, [mean, low, high])

    def test_regions_share_resamples(self):
        perievent = _gonogo_perievent()
        first = perievent["region"] == PERIEVENT_REGIONS[0]
        perievent.loc[~first, "F"] = 2 * perievent.loc[first, "F"].to_numpy()
        boot = pm.metrics_tables(perievent, bootstrap=pm.BootstrapOptions(n_resamples=300)).boot.dropna()
        left = boot[boot["region"] == PERIEVENT_REGIONS[0]].set_index("group")
        right = boot[boot["region"] == PERIEVENT_REGIONS[1]].set_index("group")
        # With the same resamples, doubling a region doubles every resampled amplitude and leaves onsets unchanged.
        amplitude = ["boot_amplitude", "boot_amplitude_ci_low", "boot_amplitude_ci_high"]
        np.testing.assert_allclose(right[amplitude], 2 * left[amplitude])
        onset = ["boot_onset_time", "boot_onset_time_ci_low", "boot_onset_time_ci_high"]
        np.testing.assert_allclose(right[onset], left[onset])

    def test_seed(self):
        perievent = _gonogo_perievent()
        options = pm.BootstrapOptions(n_resamples=200, seed=1)
        first = pm.metrics_tables(perievent, bootstrap=options).boot
        pd.testing.assert_frame_equal(first, pm.metrics_tables(perievent, bootstrap=options).boot)
        other = pm.metrics_tables(perievent, bootstrap=pm.BootstrapOptions(n_resamples=200, seed=2)).boot
        assert not first["boot_amplitude_ci_low"].equals(other["boot_amplitude_ci_low"])

    def test_skipped(self):
        tables = pm.metrics_tables(_gonogo_perievent(), bootstrap=pm.BootstrapOptions(n_resamples=0))
        assert tables.boot is None
        assert tables.boot_traces is None

    def test_without_trial_groups(self, metrics_perievent_csv):
        perievent = pd.read_csv(metrics_perievent_csv)
        tables = pm.metrics_tables(perievent, min_trials=2, bootstrap=pm.BootstrapOptions(n_resamples=100))
        assert tables.boot["group"].tolist() == ["all", "all"]
        assert tables.boot["boot_amplitude"].notna().all()
        # Three trials are below the default minimum.
        default = pm.metrics_tables(perievent)
        assert default.boot["n_trials"].tolist() == [3, 3]
        assert default.boot[list(pm.BOOT_COLUMNS)].isna().all().all()
        assert default.boot_traces.empty


def _read_boot(path):
    """A `_metrics-boot.csv` with its resample counts read as nullable integers, as written."""
    return pd.read_csv(path, dtype={column: "Int64" for column in pm.BOOT_COLUMNS if column.endswith("_n")})


def test_metrics_cmd_bootstrap(output_dir, tmp_path):
    path = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    _gonogo_perievent().to_csv(path, index=False)
    result = CliRunner().invoke(
        mesoscopy.cli, args=f"process metrics {path} -o {output_dir} --bootstrap 300 --ci 90 --seed 3"
    )
    assert result.exit_code == 0, result.output
    assert "Saved bootstrapped mean-trace metrics" in result.output
    assert "Saved bootstrapped mean traces" in result.output
    stem = pathlib.Path(output_dir) / "ses-01_regions_event-cueonset"
    expected = pm.metrics_tables(pd.read_csv(path), bootstrap=pm.BootstrapOptions(n_resamples=300, ci=90, seed=3))
    pd.testing.assert_frame_equal(_read_boot(f"{stem}_metrics-boot.csv"), expected.boot)
    pd.testing.assert_frame_equal(pd.read_csv(f"{stem}_traces-boot.csv"), expected.boot_traces, check_dtype=False)


def test_metrics_cmd_bootstrap_skipped(metrics_perievent_csv, output_dir):
    result = CliRunner().invoke(
        mesoscopy.cli, args=f"process metrics {metrics_perievent_csv} -o {output_dir} --bootstrap 0"
    )
    assert result.exit_code == 0, result.output
    assert "bootstrapped" not in result.output
    assert not list(pathlib.Path(output_dir).glob("*-boot.csv"))
    assert (pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_metrics-session.csv").is_file()


def test_perievent_cmd_with_metrics_bootstrap(perievent_regions_csv, perievent_trials_csv, output_dir):
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=(
            f"process peri-event {perievent_regions_csv} {perievent_trials_csv} -o {output_dir} --with-metrics"
            " --min-trials 2 --bootstrap 200 --ci 80 --seed 5"
        ),
    )
    assert result.exit_code == 0, result.output
    stem = pathlib.Path(output_dir) / "ses-01_regions_event-cueonset"
    windows = pd.read_csv(f"{stem}_perievent.csv")
    expected = pm.metrics_tables(
        windows,
        trials=pd.read_csv(perievent_trials_csv),
        min_trials=2,
        bootstrap=pm.BootstrapOptions(n_resamples=200, ci=80, seed=5),
    )
    boot = _read_boot(f"{stem}_metrics-boot.csv")
    pd.testing.assert_frame_equal(boot, expected.boot)
    # The two lever pushes meet the lowered minimum.
    assert boot[boot["group"] == "resp-push"]["boot_amplitude"].notna().all()


# ---------------------------------------------------------------------------------------------------------------------
# Connectivity


CONNECTIVITY_REGIONS = ["L_MOp", "R_MOp", "L_SSp-ul"]
DEFAULT_METRICS = [metric for metric in pc.PAIR_METRICS if metric not in pc.TE_METRICS]
DEFAULT_COLUMNS = [column for column in pc.TABLE_COLUMNS if column not in pc.TE_METRICS]
TRACE_COLUMNS = [column for column in DEFAULT_COLUMNS if column not in {"r_residual", "r_trials_avg", "mi_residual"}]
CONNECTIVITY_PAIRS = [("L_MOp", "R_MOp"), ("L_MOp", "L_SSp-ul"), ("R_MOp", "L_SSp-ul")]


def _connectivity_cube(seed=0):
    """Go/no-go traces for three regions: a shared response plus noise, and the first delayed by two samples."""
    n_trials = len(_gonogo_info())
    rng = np.random.default_rng(seed)
    shape = np.interp(METRICS_GRID, [0.0, 0.5, 1.5], [0.0, 1.0, 0.0])
    first, second = (shape + 0.1 * rng.standard_normal((n_trials, len(METRICS_GRID))) for _ in range(2))
    return np.stack([first, second, np.roll(first, 2, axis=1)], axis=-1)


def _connectivity_perievent(cube=None):
    """Long-format cue-aligned peri-event table of the go/no-go session for `CONNECTIVITY_REGIONS`."""
    info = _gonogo_info()
    cube = cube if cube is not None else _connectivity_cube()
    frames = [
        pd.DataFrame(
            {
                "trial_index": row["trial_index"],
                "event_time": row["event_time"],
                "time": METRICS_GRID,
                "region": region,
                "F": cube[i, :, column],
            }
        )
        for i, row in info.iterrows()
        for column, region in enumerate(CONNECTIVITY_REGIONS)
    ]
    return pd.concat(frames, ignore_index=True).merge(info.drop(columns="event_time"), on="trial_index")


def _connectivity_regions(n_frames=500, seed=0, time_aligned=True):
    """Long-format regions table of `CONNECTIVITY_REGIONS` at 40 ms: a shared slow wave plus noise, and the first
    region delayed by two frames as the third."""
    rng = np.random.default_rng(seed)
    seconds = 0.04 * np.arange(n_frames)
    wave = np.sin(2 * np.pi * 0.5 * seconds)
    first, second = (wave + 0.3 * rng.standard_normal(n_frames) for _ in range(2))
    traces = np.stack([first, second, np.roll(first, 2)], axis=-1)
    start = datetime(2024, 1, 1, 14, 0, 0)
    frames = [
        pd.DataFrame(
            {
                "region": region,
                "timestamp": [(start + timedelta(seconds=t)).isoformat() for t in seconds],
                "time_aligned": seconds + 10.0,
                "F": traces[:, column],
            }
        )
        for column, region in enumerate(CONNECTIVITY_REGIONS)
    ]
    table = pd.concat(frames, ignore_index=True)
    return table if time_aligned else table.drop(columns="time_aligned")


def _coupled_traces(n_samples=4000, coupling=0.5, lag=1, seed=0):
    """Two AR(1) traces where the second follows the first `lag` samples back, plus an independent third."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n_samples, 3))
    traces = np.zeros((n_samples, 3))
    for t in range(1, n_samples):
        traces[t, 0] = 0.8 * traces[t - 1, 0] + noise[t, 0]
        traces[t, 1] = 0.8 * traces[t - 1, 1] + coupling * traces[max(t - lag, 0), 0] + noise[t, 1]
        traces[t, 2] = 0.8 * traces[t - 1, 2] + noise[t, 2]
    return traces


def _pairwise_transfer_entropy(epochs, history, lag):
    """Transfer entropy by an explicit regression per directed pair, for checking the covariance form."""
    rows = pc.embedded_samples(epochs, history, lag)
    n_regions = epochs.shape[-1]
    te = np.full((n_regions, n_regions), np.nan)
    for target in range(n_regions):
        past = np.column_stack(
            [np.ones(len(rows)), *[rows[:, n_regions * step + target] for step in range(1, history + 1)]]
        )
        residual = lambda column: column - past @ np.linalg.lstsq(past, column, rcond=None)[0]  # noqa: E731
        target_residual = residual(rows[:, target])
        for source in range(n_regions):
            if source == target:
                continue
            source_residual = residual(rows[:, n_regions * (history + 1) + source])
            rho = np.corrcoef(target_residual, source_residual)[0, 1]
            te[source, target] = -0.5 * np.log2(1 - rho**2)
    return te


def _post_event_samples(end=3.0):
    """Samples of the grid within `[0, end]`."""
    grid = METRICS_GRID
    return int(((grid >= 0) & (grid <= np.nextafter(end, np.inf))).sum())


def _onset_coupled_cube(rho=0.6, n_trials=500, n_samples=40, onset=20, seed=0):
    """Two unit-variance Gaussian regions, independent before sample `onset` and correlated by `rho` from it on."""
    rng = np.random.default_rng(seed)
    first, other = rng.standard_normal((2, n_trials, n_samples))
    coupled = np.arange(n_samples) >= onset
    second = np.where(coupled, rho * first + np.sqrt(1 - rho**2) * other, other)
    return np.stack([first, second], axis=-1)


class TestPerieventTrials:
    def test_time_and_trials(self):
        time, info = pm.perievent_trials(_connectivity_perievent())
        np.testing.assert_array_equal(time, METRICS_GRID)
        assert info["trial_index"].tolist() == list(range(36))
        assert "sdt_type" in info.columns
        assert "F" not in info.columns

    def test_varying_column_raises(self):
        perievent = _connectivity_perievent()
        perievent["bad"] = np.arange(len(perievent))
        with pytest.raises(ValueError, match="bad vary within a trial"):
            pm.perievent_trials(perievent)


class TestPooledSamples:
    def test_drops_incomplete_rows(self):
        epochs = np.arange(24, dtype=float).reshape(2, 4, 3)
        epochs[0, 1, 2] = np.nan
        epochs[1, 3, :] = np.nan
        pooled = pc.pooled_samples(epochs)
        assert pooled.shape == (6, 3)
        assert not np.isnan(pooled).any()

    def test_absent_region_stays_nan(self):
        epochs = np.ones((2, 4, 3))
        epochs[:, :, 1] = np.nan
        pooled = pc.pooled_samples(epochs)
        assert pooled.shape == (8, 3)
        assert np.isnan(pooled[:, 1]).all()
        assert not np.isnan(pooled[:, [0, 2]]).any()


class TestCorrelationMatrix:
    def test_matches_corrcoef(self):
        samples = np.random.default_rng(0).standard_normal((50, 4))
        np.testing.assert_allclose(pc.correlation_matrix(samples), np.corrcoef(samples, rowvar=False))

    def test_constant_column_and_too_few_samples(self):
        samples = np.random.default_rng(0).standard_normal((50, 3))
        samples[:, 1] = 2.0
        matrix = pc.correlation_matrix(samples)
        assert np.isnan(matrix[1]).all()
        assert np.isnan(matrix[:, 1]).all()
        assert matrix[0, 2] == pytest.approx(np.corrcoef(samples[:, 0], samples[:, 2])[0, 1])
        assert np.isnan(pc.correlation_matrix(samples[:1])).all()


class TestResidualEpochs:
    def test_removes_mean_per_sample(self):
        epochs = np.random.default_rng(0).standard_normal((5, 6, 2))
        epochs[0, :2] = np.nan
        residuals = pc.residual_epochs(epochs)
        np.testing.assert_allclose(np.nanmean(residuals, axis=0), 0.0, atol=1e-12)
        assert np.isnan(residuals[0, :2]).all()
        assert not np.isnan(residuals[1:]).any()

    def test_identical_trials_give_zero(self):
        epochs = np.tile(np.arange(6.0)[None, :, None], (4, 1, 3))
        np.testing.assert_allclose(pc.residual_epochs(epochs), 0.0)


class TestTrialCorrelation:
    def test_matches_fisher_mean_of_trials(self):
        epochs = np.random.default_rng(0).standard_normal((6, 10, 3))
        epochs[0, 5:] = np.nan
        epochs[1, 2:] = np.nan
        z = []
        for trial in epochs:
            valid = ~np.isnan(trial).any(axis=1)
            if valid.sum() >= 3:
                with np.errstate(divide="ignore"):
                    z.append(np.arctanh(np.corrcoef(trial[valid], rowvar=False)))
        assert len(z) == 5
        expected = np.tanh(np.mean(z, axis=0))
        off_diagonal = ~np.eye(3, dtype=bool)
        np.testing.assert_allclose(pc.trial_correlation(epochs)[off_diagonal], expected[off_diagonal])

    def test_min_samples(self):
        epochs = np.random.default_rng(0).standard_normal((3, 2, 2))
        assert np.isnan(pc.trial_correlation(epochs)).all()
        assert not np.isnan(pc.trial_correlation(epochs, min_samples=2)[0, 1])


class TestPartialCorrelation:
    def test_matches_regression_residuals(self):
        rng = np.random.default_rng(0)
        samples = rng.standard_normal((200, 4)) @ rng.standard_normal((4, 4))
        partial = pc.partial_correlation(samples)
        for i, j in [(0, 1), (2, 3), (1, 3)]:
            others = [k for k in range(4) if k not in {i, j}]
            design = np.column_stack([np.ones(200), samples[:, others]])
            residuals = [samples[:, k] - design @ np.linalg.lstsq(design, samples[:, k], rcond=None)[0] for k in (i, j)]
            assert partial[i, j] == pytest.approx(np.corrcoef(*residuals)[0, 1])
            assert partial[j, i] == partial[i, j]
        np.testing.assert_allclose(np.diag(partial), 1.0)

    def test_shared_sum_turns_independent_negative(self):
        rng = np.random.default_rng(1)
        x, y = rng.standard_normal((2, 500))
        samples = np.column_stack([x, y, x + y + 0.1 * rng.standard_normal(500)])
        assert abs(np.corrcoef(x, y)[0, 1]) < 0.1
        assert pc.partial_correlation(samples)[0, 1] < -0.9

    def test_nan_column_and_too_few_samples(self):
        samples = np.random.default_rng(0).standard_normal((50, 3))
        samples[:, 1] = np.nan
        partial = pc.partial_correlation(samples)
        assert np.isnan(partial[1]).all()
        assert np.isnan(partial[:, 1]).all()
        # With nothing else to condition on, the partial correlation is the correlation.
        assert partial[0, 2] == pytest.approx(np.corrcoef(samples[:, 0], samples[:, 2])[0, 1])
        assert np.isnan(pc.partial_correlation(samples[:1])).all()


class TestMutualInformation:
    @staticmethod
    def _gaussian(rho, n, scale=(1.0, 1.0), seed=0):
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], n) * np.asarray(scale)

    @pytest.mark.parametrize(
        ("rho", "n", "scale"), [(0.3, 500, (1.0, 1.0)), (0.6, 2000, (0.3, 7.0)), (-0.9, 1000, (5.0, 0.1))]
    )
    def test_matches_sklearn(self, rho, n, scale):
        z = self._gaussian(rho, n, scale)
        expected = mutual_info_regression(z[:, [0]], z[:, 1], n_neighbors=3, random_state=0)[0] / np.log(2)
        assert pc.mutual_information(z[:, 0], z[:, 1]) == pytest.approx(expected, abs=0.005)

    def test_gaussian_analytic(self):
        z = self._gaussian(0.8, 20000)
        assert pc.mutual_information(z[:, 0], z[:, 1]) == pytest.approx(-0.5 * np.log2(1 - 0.8**2), abs=0.05)

    def test_independent_near_zero(self):
        z = self._gaussian(0.0, 2000)
        assert abs(pc.mutual_information(z[:, 0], z[:, 1])) < 0.05

    def test_nonlinear_dependence(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(-1.0, 1.0, 3000)
        y = x**2 + 0.05 * rng.standard_normal(3000)
        assert abs(np.corrcoef(x, y)[0, 1]) < 0.1
        assert pc.mutual_information(x, y) > 1.0

    def test_symmetric_and_affine_invariant(self):
        z = self._gaussian(0.5, 500)
        mi = pc.mutual_information(z[:, 0], z[:, 1])
        assert pc.mutual_information(z[:, 1], z[:, 0]) == pytest.approx(mi)
        # Rescaling moves a few samples across neighbour boundaries, which shifts the estimate by ~1e-4 bits.
        assert pc.mutual_information(3.0 * z[:, 0] + 1.0, -z[:, 1]) == pytest.approx(mi, abs=0.01)

    def test_neighbours(self):
        z = self._gaussian(0.6, 2000)
        assert pc.mutual_information(z[:, 0], z[:, 1], k=10) == pytest.approx(
            pc.mutual_information(z[:, 0], z[:, 1]), abs=0.05
        )

    def test_undefined(self):
        z = self._gaussian(0.5, 50)
        assert np.isnan(pc.mutual_information(z[:3, 0], z[:3, 1]))
        assert np.isnan(pc.mutual_information(np.ones(50), z[:, 1]))
        assert np.isnan(pc.mutual_information(np.where(np.arange(50) == 7, np.nan, z[:, 0]), z[:, 1]))

    def test_ties_are_finite(self):
        rng = np.random.default_rng(0)
        x = np.repeat([0.0, 1.0], 50)
        assert np.isfinite(pc.mutual_information(x, rng.standard_normal(100)))
        assert np.isfinite(pc.mutual_information(x, x))


class TestMutualInformationMatrix:
    def test_matches_pairs(self):
        samples = np.random.default_rng(0).standard_normal((200, 4)) @ np.random.default_rng(1).standard_normal((4, 4))
        matrix = pc.mutual_information_matrix(samples)
        assert np.isnan(np.diag(matrix)).all()
        for i in range(4):
            for j in range(i + 1, 4):
                assert matrix[i, j] == pc.mutual_information(samples[:, i], samples[:, j])
                assert matrix[j, i] == matrix[i, j]

    def test_nan_column(self):
        samples = np.random.default_rng(0).standard_normal((100, 3))
        samples[:, 1] = np.nan
        matrix = pc.mutual_information_matrix(samples, k=5)
        assert np.isnan(matrix[1]).all()
        assert np.isnan(matrix[:, 1]).all()
        assert matrix[0, 2] == pc.mutual_information(samples[:, 0], samples[:, 2], k=5)


class TestLaggedCorrelation:
    def test_delayed_copy_peaks_at_its_lag(self):
        first = np.random.default_rng(0).standard_normal((4, 30))
        epochs = np.stack([first, np.roll(first, 2, axis=1)], axis=-1)
        lagged = pc.lagged_correlation(epochs, 3)
        assert lagged.shape == (7, 2, 2)
        # Region 1 at t + 2 is region 0 at t.
        assert lagged[3 + 2, 0, 1] == pytest.approx(1.0)
        assert lagged[3 - 2, 1, 0] == pytest.approx(1.0)
        assert lagged[3, 0, 0] == pytest.approx(1.0)
        index, peak = pc.peak_lag(lagged)
        assert index[0, 1] == 5
        assert index[1, 0] == 1
        assert peak[0, 1] == pytest.approx(1.0)

    def test_symmetric_in_lag_and_pair(self):
        epochs = np.random.default_rng(0).standard_normal((3, 20, 3))
        epochs[0, 15:] = np.nan
        lagged = pc.lagged_correlation(epochs, 4)
        np.testing.assert_allclose(lagged, np.swapaxes(lagged[::-1], 1, 2))

    def test_zero_lag_is_pooled_correlation(self):
        epochs = np.random.default_rng(0).standard_normal((3, 20, 3))
        epochs[0, 15:] = np.nan
        expected = pc.correlation_matrix(pc.pooled_samples(epochs))
        np.testing.assert_allclose(pc.lagged_correlation(epochs, 2)[2], expected)

    def test_lag_beyond_epoch_is_nan(self):
        epochs = np.random.default_rng(0).standard_normal((2, 3, 2))
        lagged = pc.lagged_correlation(epochs, 3)
        assert np.isnan(lagged[0]).all()
        assert np.isnan(lagged[-1]).all()
        assert not np.isnan(lagged[3]).any()


class TestPeakLag:
    def test_largest_magnitude_first_on_ties(self):
        lagged = np.zeros((3, 2, 2))
        lagged[0, 0, 1], lagged[2, 0, 1], lagged[1, 1, 0] = -0.5, 0.5, 0.9
        index, peak = pc.peak_lag(lagged)
        assert index[0, 1] == 0
        assert peak[0, 1] == -0.5
        assert index[1, 0] == 1
        assert peak[1, 0] == 0.9

    def test_all_nan(self):
        lagged = np.full((3, 2, 2), np.nan)
        lagged[1, 0, 0] = 1.0
        index, peak = pc.peak_lag(lagged)
        assert np.isnan(peak[0, 1])
        assert index[0, 1] == 0
        assert peak[0, 0] == 1.0
        assert index[0, 0] == 1


class TestEpochBounds:
    def test_matches_reliability_epochs(self):
        info = _gonogo_info()
        start, end, keep = pc.epoch_bounds(info, "cue_onset", (0.0, 3.0), min_rt=0.5)
        expected = pm.reliability_epochs(info, "cue_onset", (0.0, 3.0), min_rt=0.5)
        np.testing.assert_array_equal(start, expected[0])
        np.testing.assert_array_equal(end, expected[1])
        np.testing.assert_array_equal(keep, expected[3])
        assert not keep.all()

    @pytest.mark.parametrize(
        ("columns", "event", "reason"),
        [
            (["response_time"], "cue_onset", "no response_time column"),
            (["cue_onset", "response_time"], "cue_onset", "no cue_onset, response_time column"),
            ([], None, "aligned event is unknown"),
            ([], "reward", "not defined for reward-aligned windows"),
        ],
    )
    def test_falls_back_to_response_window(self, columns, event, reason):
        info = _gonogo_info().drop(columns=columns)
        with pytest.warns(UserWarning, match=f"Using the response window as the epoch: .*{reason}"):
            start, end, keep = pc.epoch_bounds(info, event, (0.0, 3.0))
        np.testing.assert_array_equal(start, 0.0)
        assert (end > 3.0).all()
        assert (end < 3.0001).all()
        assert keep.all()

    @pytest.mark.parametrize(("event", "window"), [("cue_onset", (0.0, 3.0)), ("trial_start", (0.0, 10.0))])
    def test_response_pad_extends_epochs(self, event, window):
        info = _gonogo_info(event)
        start, end, keep = pc.epoch_bounds(info, event, window, response_pad=0.05)
        expected = pm.reliability_epochs(info, event, window)
        np.testing.assert_array_equal(start, expected[0])
        np.testing.assert_allclose(end, expected[1] + 0.05)
        np.testing.assert_array_equal(keep, expected[3])

    def test_response_pad_capped_at_response_window(self):
        _, end, _ = pc.epoch_bounds(_gonogo_info(), "cue_onset", (0.0, 1.0), response_pad=0.3)
        np.testing.assert_allclose(end[:6], [0.7, 0.9, 1.0, 1.0, 1.0, 1.0])
        assert end.max() == pytest.approx(1.0)
        assert end.max() > 1.0

    def test_response_pad_leaves_unmasked_epochs(self):
        info = _gonogo_info()
        padded = pc.epoch_bounds(info, "cue_onset", (0.0, 3.0), mask_response=False, response_pad=0.3)
        for actual, expected in zip(padded, pc.epoch_bounds(info, "cue_onset", (0.0, 3.0), False), strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_response_pad_extends_lever_epochs(self):
        info = _gonogo_info("response")
        start, end, _ = pc.epoch_bounds(info, "response", (0.0, 0.1), response_pad=0.3)
        np.testing.assert_allclose(start, -info["response_time"])
        # Lever-aligned epochs are not capped at the response window.
        np.testing.assert_allclose(end, 0.3)

    def test_response_pad_ignored_on_fallback(self):
        info = _gonogo_info().drop(columns="response_time")
        with pytest.warns(UserWarning, match="Using the response window as the epoch"):
            _, end, _ = pc.epoch_bounds(info, "cue_onset", (0.0, 3.0), response_pad=0.3)
        assert (end < 3.0001).all()


class TestConnectivityTable:
    @staticmethod
    def _row(table, pair, group):
        return table[(table["region_a"] == pair[0]) & (table["region_b"] == pair[1]) & (table["group"] == group)].iloc[
            0
        ]

    def test_layout(self):
        table = pc.connectivity_table(_connectivity_perievent())
        groups = list(pm.trial_groups(_gonogo_info()))
        assert list(table.columns) == DEFAULT_COLUMNS
        assert list(zip(table["region_a"], table["region_b"], strict=True)) == [
            p for p in CONNECTIVITY_PAIRS for _ in groups
        ]
        assert table["group"].tolist() == groups * len(CONNECTIVITY_PAIRS)
        assert len(table) == 27

    def test_shared_response_versus_residual(self):
        row = self._row(pc.connectivity_table(_connectivity_perievent()), CONNECTIVITY_PAIRS[0], "all")
        assert row["r"] > 0.8
        assert abs(row["r_residual"]) < 0.15
        assert row["r_trials_avg"] > 0.8
        assert row["mi"] > 0.5
        assert abs(row["mi_residual"]) < 0.1
        assert row["n_trials"] == 36

    def test_lag_of_delayed_copy(self):
        row = self._row(pc.connectivity_table(_connectivity_perievent()), CONNECTIVITY_PAIRS[1], "all")
        assert row["lag"] == pytest.approx(0.08)
        assert row["r_lag"] == pytest.approx(1.0)
        assert row["r"] < row["r_lag"]

    def test_matches_pair_metrics(self):
        perievent = _connectivity_perievent()
        table = pc.connectivity_table(perievent, max_lag=0.2)
        info = _gonogo_info()
        hits = (info["sdt_type"] == "hit").to_numpy()
        start, end, keep = pc.epoch_bounds(info, "cue_onset", (0.0, 3.0))
        cube = pc.epoch_cube(perievent, info["trial_index"].to_numpy(), METRICS_GRID, CONNECTIVITY_REGIONS, start, end)
        assert cube.shape == (36, len(METRICS_GRID), 3)
        expected = pc.pair_metrics(cube[hits & keep], 5)
        rows = table[table["group"] == "sdt-hit"]
        for metric in DEFAULT_METRICS:
            matrix = expected[metric] * (0.04 if metric == "lag" else 1.0)
            assert rows[metric].tolist() == pytest.approx([matrix[0, 1], matrix[0, 2], matrix[1, 2]], nan_ok=True)
        assert rows["n_samples"].tolist() == [expected["n_samples"]] * 3
        assert rows["n_trials"].tolist() == [12] * 3

    def test_n_samples_counts_epoch_samples(self):
        table = pc.connectivity_table(_connectivity_perievent())
        info = _gonogo_info()
        start, end, _, keep = pm.reliability_epochs(info, "cue_onset", (0.0, 3.0))
        inside = ~np.isnan(pm.epoch_traces(np.zeros((len(info), len(METRICS_GRID))), METRICS_GRID, start, end))
        assert table[table["group"] == "all"]["n_samples"].tolist() == [int(inside[keep].sum())] * 3

    def test_min_trials(self):
        table = pc.connectivity_table(_connectivity_perievent())
        misses = table[table["group"] == "sdt-miss"]
        assert misses["n_trials"].tolist() == [6] * 3
        assert misses["n_samples"].tolist() == [0] * 3
        assert misses[DEFAULT_METRICS].isna().all().all()

        lowered = pc.connectivity_table(_connectivity_perievent(), min_trials=5, transfer_entropy=True, te_surrogates=5)
        assert lowered[lowered["group"] == "sdt-miss"][[*DEFAULT_METRICS, *pc.TE_METRICS]].notna().all().all()

        # The all-trials metrics are always taken.
        strict = pc.connectivity_table(_connectivity_perievent(), min_trials=100)
        assert strict[strict["group"] == "all"]["r"].notna().all()
        assert strict[strict["group"] != "all"]["r"].isna().all()

    def test_min_rt_drops_trials(self):
        table = pc.connectivity_table(_connectivity_perievent(), min_rt=0.5)
        # Two hits respond at 0.4 s.
        assert table[table["group"] == "all"]["n_trials"].tolist() == [34] * 3
        assert table[table["group"] == "sdt-hit"]["n_trials"].tolist() == [10] * 3

    def test_no_mask_response_uses_whole_window(self):
        masked = pc.connectivity_table(_connectivity_perievent())
        whole = pc.connectivity_table(_connectivity_perievent(), mask_response=False)
        taken = whole["n_samples"] > 0
        assert (whole["n_samples"][taken] > masked["n_samples"][taken]).all()
        assert whole[whole["group"] == "all"]["n_samples"].tolist() == [36 * _post_event_samples()] * 3

    def test_response_window(self):
        table = pc.connectivity_table(_connectivity_perievent(), response=(0.0, 0.5))
        assert table[table["group"] == "all"]["n_samples"].tolist() == [36 * _post_event_samples(0.5) - 2 * 3] * 3

    def test_response_pad_adds_samples(self):
        padded = pc.connectivity_table(_connectivity_perievent(), response_pad=0.1)
        info = _gonogo_info()
        start, end, keep = pc.epoch_bounds(info, "cue_onset", (0.0, 3.0), response_pad=0.1)
        inside = ~np.isnan(pm.epoch_traces(np.zeros((len(info), len(METRICS_GRID))), METRICS_GRID, start, end))
        assert padded[padded["group"] == "all"]["n_samples"].tolist() == [int(inside[keep].sum())] * 3
        masked = pc.connectivity_table(_connectivity_perievent())
        taken = masked["n_samples"] > 0
        assert (padded["n_samples"][taken] > masked["n_samples"][taken]).all()

    def test_without_groups_warns(self):
        with pytest.warns(UserWarning, match="Skipping trial groups: no sdt_type column"):
            table = pc.connectivity_table(_connectivity_perievent().drop(columns="sdt_type"))
        assert table["group"].tolist() == ["all"] * 3

    def test_without_epoch_columns_warns(self):
        with pytest.warns(UserWarning, match="Using the response window as the epoch: no response_time column"):
            table = pc.connectivity_table(_connectivity_perievent().drop(columns="response_time"))
        assert table[table["group"] == "all"]["n_samples"].tolist() == [36 * _post_event_samples()] * 3

    def test_joins_trials(self):
        perievent = _connectivity_perievent()
        columns = [column for column in perievent.columns if column not in pm.PERIEVENT_COLUMNS]
        trials = perievent.drop_duplicates("trial_index")[columns].reset_index(drop=True)
        joined = pc.connectivity_table(perievent.drop(columns=columns), trials=trials)
        pd.testing.assert_frame_equal(joined, pc.connectivity_table(perievent))

    def test_transfer_entropy_columns(self):
        perievent = _connectivity_perievent()
        table = pc.connectivity_table(perievent, transfer_entropy=True, te_surrogates=20, seed=7)
        rows = table[table["group"] == "all"]
        assert rows[list(pc.TE_METRICS)].notna().all().all()
        delayed = self._row(table, CONNECTIVITY_PAIRS[1], "all")
        assert delayed["te_ab"] > delayed["te_ba"]
        assert delayed["te_ab_z"] > 5
        pd.testing.assert_frame_equal(
            table, pc.connectivity_table(perievent, transfer_entropy=True, te_surrogates=20, seed=7)
        )
        off = pc.connectivity_table(perievent)
        assert not set(pc.TE_METRICS) & set(off.columns)
        bare = pc.connectivity_table(perievent, transfer_entropy=True, te_surrogates=0)
        pd.testing.assert_series_equal(bare["te_ab"], table["te_ab"])
        assert list(bare.columns) == [*DEFAULT_COLUMNS[:-2], "te_ab", "te_ba", "n_trials", "n_samples"]

    def test_without_mutual_info(self):
        perievent = _connectivity_perievent()
        skipped = pc.connectivity_table(perievent, mutual_info=False)
        assert not {"mi", "mi_residual"} & set(skipped.columns)
        others = [column for column in DEFAULT_COLUMNS if column not in {"mi", "mi_residual"}]
        pd.testing.assert_frame_equal(skipped[others], pc.connectivity_table(perievent)[others])

    def test_mi_neighbours(self):
        perievent = _connectivity_perievent()
        default = pc.connectivity_table(perievent)
        wider = pc.connectivity_table(perievent, mi_neighbours=10)
        assert not np.allclose(default["mi"].dropna(), wider["mi"].dropna())
        assert wider[wider["group"] == "all"]["mi"].tolist() == pytest.approx(
            default[default["group"] == "all"]["mi"].tolist(), abs=0.1
        )

    def test_invalid_table(self):
        with pytest.raises(ValueError, match="lacks the F column"):
            pc.connectivity_table(_connectivity_perievent().drop(columns="F"))
        with pytest.raises(ValueError, match="no rows"):
            pc.connectivity_table(_connectivity_perievent().iloc[:0])

    def test_single_region(self):
        perievent = _connectivity_perievent()
        table = pc.connectivity_table(perievent[perievent["region"] == "L_MOp"])
        assert table.empty
        assert list(table.columns) == DEFAULT_COLUMNS


class TestWindowStarts:
    def test_starts(self):
        np.testing.assert_array_equal(pc.window_starts(10, 4, 3), [0, 3, 6])
        np.testing.assert_array_equal(pc.window_starts(10, 10, 1), [0])

    @pytest.mark.parametrize(
        ("window", "step", "match"),
        [(0, 1, "at least one sample"), (2, 0, "at least one sample"), (11, 1, "longer than")],
    )
    def test_invalid(self, window, step, match):
        with pytest.raises(ValueError, match=match):
            pc.window_starts(10, window, step)


class TestRollingMetrics:
    def test_matches_each_window(self):
        cube = _connectivity_cube()
        rolling = pc.rolling_metrics(cube, 12, 2)
        starts = pc.window_starts(cube.shape[1], 12, 2)
        np.testing.assert_array_equal(rolling["start"], starts)
        assert rolling["mi"].shape == (len(starts), 3, 3)
        # The residuals are taken over the whole traces, then windowed.
        residual = pc.residual_epochs(cube)
        for i in (0, len(starts) // 2, len(starts) - 1):
            window = slice(starts[i], starts[i] + 12)
            pooled = pc.pooled_samples(cube[:, window])
            residuals = pc.pooled_samples(residual[:, window])
            np.testing.assert_allclose(rolling["r"][i], pc.correlation_matrix(pooled))
            np.testing.assert_allclose(rolling["r_residual"][i], pc.correlation_matrix(residuals))
            np.testing.assert_allclose(rolling["mi"][i], pc.mutual_information_matrix(pooled), equal_nan=True)
            np.testing.assert_allclose(
                rolling["mi_residual"][i], pc.mutual_information_matrix(residuals), equal_nan=True
            )
        assert (rolling["n_samples"] == 36 * 12).all()

    def test_gaussian_coupling_after_onset(self):
        rho = 0.6
        rolling = pc.rolling_metrics(_onset_coupled_cube(rho), 10, 5)
        before = rolling["start"] + 10 <= 20
        after = rolling["start"] >= 20
        assert before.sum() == 3
        assert after.sum() == 3
        for metric in ("mi", "mi_residual"):
            np.testing.assert_allclose(rolling[metric][before, 0, 1], 0.0, atol=0.05)
            np.testing.assert_allclose(rolling[metric][after, 0, 1], -0.5 * np.log2(1 - rho**2), atol=0.05)
        np.testing.assert_allclose(rolling["r_residual"][after, 0, 1], rho, atol=0.05)

    def test_missing_samples(self):
        cube = _connectivity_cube()
        cube[0, :5, 1] = np.nan
        rolling = pc.rolling_metrics(cube, 5, 5)
        assert rolling["n_samples"][0] == 35 * 5
        assert (rolling["n_samples"][1:] == 36 * 5).all()


class TestRollingTable:
    @staticmethod
    def _rows(table, pair, group):
        return table[(table["region_a"] == pair[0]) & (table["region_b"] == pair[1]) & (table["group"] == group)]

    def test_layout(self):
        table = pc.rolling_table(_connectivity_perievent())
        groups = list(pm.trial_groups(_gonogo_info()))
        # 0.5 s and 0.1 s round to 12 and 2 samples at 25 Hz.
        starts = pc.window_starts(len(METRICS_GRID), 12, 2)
        centres = (METRICS_GRID[starts] + METRICS_GRID[starts + 11]) / 2
        assert list(table.columns) == list(pc.ROLLING_COLUMNS)
        assert len(table) == len(CONNECTIVITY_PAIRS) * len(groups) * len(starts)
        assert list(zip(table["region_a"], table["region_b"], strict=True)) == [
            p for p in CONNECTIVITY_PAIRS for _ in range(len(groups) * len(starts))
        ]
        assert table["group"].tolist() == [g for g in groups for _ in starts] * len(CONNECTIVITY_PAIRS)
        np.testing.assert_allclose(table["time"], np.tile(centres, len(CONNECTIVITY_PAIRS) * len(groups)))
        assert table["time"].iloc[0] == pytest.approx(-0.78)

    def test_matches_rolling_metrics(self):
        table = pc.rolling_table(_connectivity_perievent())
        cube = _connectivity_cube()
        hits = (_gonogo_info()["sdt_type"] == "hit").to_numpy()
        for group, used in (("all", slice(None)), ("sdt-hit", hits)):
            rolling = pc.rolling_metrics(cube[used], 12, 2)
            for (a, b), pair in zip([(0, 1), (0, 2), (1, 2)], CONNECTIVITY_PAIRS, strict=True):
                rows = self._rows(table, pair, group)
                for metric in pc.ROLLING_METRICS:
                    np.testing.assert_allclose(rows[metric], rolling[metric][:, a, b])
                np.testing.assert_array_equal(rows["n_samples"], rolling["n_samples"])
        assert self._rows(table, CONNECTIVITY_PAIRS[0], "sdt-hit")["n_trials"].eq(12).all()

    def test_covers_whole_window(self):
        rows = self._rows(pc.rolling_table(_connectivity_perievent()), CONNECTIVITY_PAIRS[0], "all")
        # Every trial gives every sample, before the cue and after its response.
        assert rows["n_samples"].eq(36 * 12).all()
        assert rows[list(pc.ROLLING_METRICS)].notna().all().all()
        before = rows[rows["time"] < -0.3]
        rising = rows[(rows["time"] >= 0.2) & (rows["time"] <= 0.3)]
        assert (before["r"].abs() < 0.2).all()
        assert (rising["r"] > 0.8).all()
        assert (rising["r_residual"].abs() < 0.2).all()
        assert (rising["mi"] > rising["mi_residual"] + 0.3).all()

    def test_min_trials(self):
        default = pc.rolling_table(_connectivity_perievent())
        assert default[default["group"] == "sdt-miss"][list(pc.ROLLING_METRICS)].isna().all().all()
        strict = pc.rolling_table(_connectivity_perievent(), min_trials=100)
        others = strict[strict["group"] != "all"]
        assert others[list(pc.ROLLING_METRICS)].isna().all().all()
        assert others["n_samples"].eq(0).all()
        assert others[others["group"] == "sdt-miss"]["n_trials"].eq(6).all()
        assert strict[strict["group"] == "all"]["mi"].notna().all()

    def test_min_rt_drops_trials(self):
        table = pc.rolling_table(_connectivity_perievent(), min_rt=0.5)
        rows = table[table["group"] == "all"]
        # Two hits respond at 0.4 s.
        assert rows["n_trials"].eq(34).all()
        assert rows["n_samples"].eq(34 * 12).all()

    def test_window_and_step_round_to_samples(self):
        table = pc.rolling_table(_connectivity_perievent(), window=0.2, step=0.12)
        np.testing.assert_allclose(np.diff(np.unique(table["time"])), 0.12)
        assert table["n_samples"].max() == 36 * 5

    def test_invalid_window(self):
        with pytest.raises(ValueError, match="longer than"):
            pc.rolling_table(_connectivity_perievent(), window=10.0)
        with pytest.raises(ValueError, match="at least one sample"):
            pc.rolling_table(_connectivity_perievent(), step=0.01)

    def test_without_groups_warns(self):
        with pytest.warns(UserWarning, match="Skipping trial groups: no sdt_type column"):
            table = pc.rolling_table(_connectivity_perievent().drop(columns="sdt_type"))
        assert set(table["group"]) == {"all"}

    def test_joins_trials(self):
        perievent = _connectivity_perievent()
        columns = [column for column in perievent.columns if column not in pm.PERIEVENT_COLUMNS]
        trials = perievent.drop_duplicates("trial_index")[columns].reset_index(drop=True)
        joined = pc.rolling_table(perievent.drop(columns=columns), trials=trials)
        pd.testing.assert_frame_equal(joined, pc.rolling_table(perievent))

    def test_single_region(self):
        perievent = _connectivity_perievent()
        table = pc.rolling_table(perievent[perievent["region"] == "L_MOp"])
        assert table.empty
        assert list(table.columns) == list(pc.ROLLING_COLUMNS)


class TestEmbeddedSamples:
    def test_columns_and_within_trial_rows(self):
        cube = np.arange(2 * 6 * 2, dtype=float).reshape(2, 6, 2)
        rows = pc.embedded_samples(cube, history=2, lag=3)
        assert rows.shape == (2 * (6 - 3), 2 * 4)
        first = rows[0]
        np.testing.assert_array_equal(first[:2], cube[0, 3])
        np.testing.assert_array_equal(first[2:4], cube[0, 2])
        np.testing.assert_array_equal(first[4:6], cube[0, 1])
        np.testing.assert_array_equal(first[6:], cube[0, 0])
        np.testing.assert_array_equal(rows[3, :2], cube[1, 3])

    def test_drops_nan_rows_and_takes_sources_elsewhere(self):
        cube = np.ones((1, 5, 2))
        cube[0, 1, 0] = np.nan
        source = np.full((1, 5, 2), 7.0)
        rows = pc.embedded_samples(cube, history=1, lag=1, source=source)
        assert rows.shape == (2, 6)
        np.testing.assert_array_equal(rows[:, 4:], 7.0)

    def test_too_short(self):
        assert pc.embedded_samples(np.ones((3, 2, 2)), history=2, lag=1).shape == (0, 8)


class TestTransferEntropy:
    @pytest.mark.parametrize("history", [1, 2])
    def test_matches_pairwise_regression(self, history):
        cube = _coupled_traces(500).reshape(5, 100, 3)
        cube[:, :10] = np.nan
        for lag in (1, 3):
            rows = pc.embedded_samples(cube, history, lag)
            te = pc.transfer_entropy_from_covariance(np.cov(rows, rowvar=False), history)
            np.testing.assert_allclose(te, _pairwise_transfer_entropy(cube, history, lag), atol=1e-12)
            assert np.isnan(np.diag(te)).all()

    def test_direction_and_analytic_value(self):
        traces = _coupled_traces(20000, coupling=0.5)
        te = pc.transfer_entropy(traces[None], history=1, max_lag=1)
        # x_t = 0.8 x_{t-1} + e, y_t = 0.8 y_{t-1} + 0.5 x_{t-1} + e. The stationary covariance of (x, y) solves
        # the Lyapunov equation, and the partial correlation of y_t and x_{t-1} given y_{t-1} follows from it.
        import scipy.linalg

        a = np.array([[0.8, 0.0], [0.5, 0.8]])
        same = scipy.linalg.solve_discrete_lyapunov(a, np.eye(2))
        back = a @ same  # cov(z_t, z_{t-1})
        r_yx = back[1, 0] / np.sqrt(same[1, 1] * same[0, 0])
        r_yh = back[1, 1] / same[1, 1]
        r_xh = same[1, 0] / np.sqrt(same[1, 1] * same[0, 0])
        partial = (r_yx - r_yh * r_xh) / np.sqrt((1 - r_yh**2) * (1 - r_xh**2))
        expected = -0.5 * np.log2(1 - partial**2)
        assert te[0, 1] == pytest.approx(expected, rel=0.05)
        assert te[1, 0] < 0.01
        assert te[0, 2] < 0.01
        assert te[2, 1] < 0.01

    def test_max_over_lags_finds_delayed_coupling(self):
        traces = _coupled_traces(20000, coupling=0.5, lag=3)
        one = pc.transfer_entropy(traces[None], history=1, max_lag=1)
        three = pc.transfer_entropy(traces[None], history=1, max_lag=3)
        assert three[0, 1] > 2 * one[0, 1]
        assert three[0, 1] > 0.2

    def test_too_few_samples(self):
        te = pc.transfer_entropy(np.ones((1, 3, 2)), history=1, max_lag=1)
        assert np.isnan(te).all()

    def test_circular_matches_direct(self):
        traces = _coupled_traces(5000)
        direct = pc.transfer_entropy(traces[None], history=2, max_lag=4)
        circular = pc.transfer_entropy_circular(traces, history=2, max_lag=4, shifts=np.array([0]))[0]
        np.testing.assert_allclose(circular, direct, atol=2e-3, equal_nan=True)
        assert np.nanmax(np.abs(circular - direct)) < 0.1 * np.nanmax(direct)

    def test_circular_shift_breaks_coupling(self):
        traces = _coupled_traces(5000)
        estimates = pc.transfer_entropy_circular(traces, history=1, max_lag=2, shifts=np.array([0, 1000, 2500]))
        assert estimates[0, 0, 1] > 0.1
        assert estimates[1:, 0, 1].max() < 0.01


class TestCircularCovariances:
    def test_matches_rolled_products(self):
        traces = np.random.default_rng(1).standard_normal((50, 3))
        centred = traces - traces.mean(axis=0)
        at = pc.circular_covariances(traces, [-2, 0, 3])
        for lag, matrix in at.items():
            expected = centred.T @ np.roll(centred, lag, axis=0) / len(traces)
            np.testing.assert_allclose(matrix, expected, atol=1e-12)


class TestSurrogateZ:
    def test_z_score(self):
        value = np.array([[np.nan, 3.0], [1.0, np.nan]])
        surrogates = np.array(
            [[[np.nan, 1.0], [1.0, np.nan]], [[np.nan, 2.0], [1.0, np.nan]], [[np.nan, 3.0], [1.0, np.nan]]]
        )
        z = pc.surrogate_z(value, surrogates)
        assert z[0, 1] == pytest.approx((3.0 - 2.0) / np.std([1.0, 2.0, 3.0]))
        assert np.isnan(z[1, 0])

    def test_too_few_surrogates(self):
        assert np.isnan(pc.surrogate_z(np.ones((2, 2)), np.ones((1, 2, 2)))).all()
        assert np.isnan(pc.surrogate_z(np.ones((2, 2)), np.empty((0, 2, 2)))).all()


class TestEpochTransferEntropy:
    def test_z_separates_coupled_from_independent(self):
        cube = _coupled_traces(6000).reshape(60, 100, 3)
        te, z = pc.epoch_transfer_entropy(cube, history=1, max_lag=2, n_surrogates=50, rng=np.random.default_rng(0))
        assert te[0, 1] > 0.1
        assert z[0, 1] > 10
        assert abs(z[1, 0]) < 4
        assert abs(z[2, 1]) < 4

    def test_without_surrogates(self):
        cube = _coupled_traces(600).reshape(6, 100, 3)
        te, z = pc.epoch_transfer_entropy(cube, 1, 2, 0, np.random.default_rng(0))
        np.testing.assert_allclose(te, pc.transfer_entropy(cube, 1, 2), equal_nan=True)
        assert np.isnan(z).all()


class TestTraceTransferEntropy:
    def test_z_separates_coupled_from_independent(self):
        traces = _coupled_traces(5000)
        te, z = pc.trace_transfer_entropy(traces, history=1, max_lag=2, n_surrogates=50, rng=np.random.default_rng(0))
        assert te[0, 1] > 0.1
        assert z[0, 1] > 10
        assert abs(z[1, 0]) < 4
        np.testing.assert_allclose(te, pc.transfer_entropy_circular(traces, 1, 2, np.array([0]))[0], equal_nan=True)

    def test_short_traces(self):
        te, z = pc.trace_transfer_entropy(np.ones((3, 2)), 1, 2, 10, np.random.default_rng(0))
        assert np.isnan(te).all()
        assert np.isnan(z).all()
        te, z = pc.trace_transfer_entropy(_coupled_traces(6), 1, 2, 10, np.random.default_rng(0))
        assert np.isfinite(te[0, 1])
        assert np.isnan(z).all()


class TestDirectedMetrics:
    def test_orientation(self):
        te = np.array([[np.nan, 1.0], [2.0, np.nan]])
        z = np.array([[np.nan, 3.0], [4.0, np.nan]])
        metrics = pc.directed_metrics(te, z)
        assert set(metrics) == set(pc.TE_METRICS)
        assert metrics["te_ab"][0, 1] == 1.0
        assert metrics["te_ba"][0, 1] == 2.0
        assert metrics["te_ab_z"][0, 1] == 3.0
        assert metrics["te_ba_z"][0, 1] == 4.0


class TestPooledMetrics:
    def test_pair_metrics_extends_it(self):
        cube = _connectivity_cube()
        pooled = pc.pooled_metrics(cube, 5)
        pair = pc.pair_metrics(cube, 5)
        assert set(pooled) == {"r", "mi", "partial_r", "lag", "r_lag", "n_samples"}
        assert set(pair) == {*pc.PAIR_METRICS, "n_samples"} - set(pc.TE_METRICS)
        for key, value in pooled.items():
            np.testing.assert_array_equal(pair[key], value)

    def test_without_mutual_info(self):
        pooled = pc.pooled_metrics(_connectivity_cube(), 5, mutual_info=False)
        assert "mi" not in pooled
        assert not np.isnan(pooled["r"]).any()


class TestPairMetricsTransferEntropy:
    def test_on_residual_epochs_with_trial_shuffles(self):
        cube = _connectivity_cube()
        rng = np.random.default_rng(3)
        pair = pc.pair_metrics(cube, 5, te_surrogates=20, transfer_entropy=True, rng=np.random.default_rng(3))
        te, z = pc.epoch_transfer_entropy(pc.residual_epochs(cube), 1, 5, 20, rng)
        np.testing.assert_allclose(pair["te_ab"], te, equal_nan=True)
        np.testing.assert_allclose(pair["te_ba"], te.T, equal_nan=True)
        np.testing.assert_allclose(pair["te_ab_z"], z, equal_nan=True)
        assert pair["te_ab"][0, 2] > pair["te_ba"][0, 2]  # the third region is the first delayed by two samples

    def test_history_and_disabled(self):
        cube = _connectivity_cube()
        two = pc.pair_metrics(cube, 5, te_history=2, te_surrogates=0, transfer_entropy=True)
        one = pc.pair_metrics(cube, 5, te_surrogates=0, transfer_entropy=True)
        assert not np.allclose(two["te_ab"], one["te_ab"], equal_nan=True)
        off = pc.pair_metrics(cube, 5)
        assert not set(pc.TE_METRICS) & set(off)


class TestTraceSamples:
    def test_traces_names_and_step(self):
        regions = _connectivity_regions()
        traces, names, step = pc.trace_samples(regions)
        assert names == CONNECTIVITY_REGIONS
        assert traces.shape == (500, 3)
        assert step == pytest.approx(0.04)
        for column, region in enumerate(names):
            np.testing.assert_array_equal(traces[:, column], regions[regions["region"] == region]["F"])

    def test_timestamp_only_and_shuffled_rows(self):
        regions = _connectivity_regions(time_aligned=False)
        traces, names, step = pc.trace_samples(regions)
        assert step == pytest.approx(0.04)
        shuffled = regions.sample(frac=1.0, random_state=1)
        shuffled_traces, shuffled_names, _ = pc.trace_samples(shuffled)
        assert shuffled_names == list(shuffled["region"].unique())
        np.testing.assert_array_equal(shuffled_traces[:, [shuffled_names.index(n) for n in names]], traces)

    def test_missing_columns(self):
        regions = _connectivity_regions()
        with pytest.raises(ValueError, match=r"Missing columns: \['F'\]"):
            pc.trace_samples(regions.drop(columns="F"))
        with pytest.raises(ValueError, match=r"Missing columns: \['timestamp'\]"):
            pc.trace_samples(regions.drop(columns=["timestamp", "time_aligned"]))


class TestTraceTable:
    @staticmethod
    def _row(table, pair):
        return table[(table["region_a"] == pair[0]) & (table["region_b"] == pair[1])].iloc[0]

    def test_layout(self):
        table = pc.trace_table(_connectivity_regions())
        assert list(table.columns) == TRACE_COLUMNS
        assert list(zip(table["region_a"], table["region_b"], strict=True)) == CONNECTIVITY_PAIRS
        assert table["group"].tolist() == ["all"] * 3
        assert table["n_trials"].isna().all()
        assert table["n_samples"].tolist() == [500] * 3
        assert table[["r", "mi", "partial_r", "lag", "r_lag"]].notna().all().all()

    def test_matches_direct_metrics(self):
        regions = _connectivity_regions()
        traces, _, _ = pc.trace_samples(regions)
        table = pc.trace_table(regions, max_lag=0.2, mi_neighbours=4, transfer_entropy=True, te_surrogates=20, seed=5)
        expected = pc.pooled_metrics(traces[None], 5, mi_neighbours=4)
        te, z = pc.trace_transfer_entropy(traces, 1, 5, 20, np.random.default_rng(5))
        expected |= pc.directed_metrics(te, z)
        for metric in pc.TE_METRICS:
            assert table[metric].tolist() == pytest.approx(
                [expected[metric][a, b] for a, b in [(0, 1), (0, 2), (1, 2)]]
            )
        assert table["te_ab"].iloc[1] > table["te_ba"].iloc[1]  # the third region is the first delayed by two frames
        assert table["te_ab_z"].iloc[1] > 5
        for metric in ("r", "mi", "partial_r", "r_lag"):
            assert table[metric].tolist() == pytest.approx(
                [expected[metric][a, b] for a, b in [(0, 1), (0, 2), (1, 2)]]
            )
        assert table["r"].tolist() == pytest.approx([np.corrcoef(traces.T)[a, b] for a, b in [(0, 1), (0, 2), (1, 2)]])
        assert table["lag"].tolist() == pytest.approx((expected["lag"] * 0.04)[np.triu_indices(3, 1)])

    def test_lag_of_delayed_copy(self):
        row = self._row(pc.trace_table(_connectivity_regions()), ("L_MOp", "L_SSp-ul"))
        assert row["lag"] == pytest.approx(0.08)
        assert row["r_lag"] > 0.99
        assert row["r"] < row["r_lag"]

    def test_n_samples_counts_finite_frames(self):
        regions = _connectivity_regions()
        regions.loc[regions.index[:3], "F"] = np.nan
        table = pc.trace_table(regions)
        assert table["n_samples"].tolist() == [497] * 3

    def test_without_mutual_info(self):
        table = pc.trace_table(_connectivity_regions(), mutual_info=False)
        assert "mi" not in table.columns
        assert table["r"].notna().all()

    def test_transfer_entropy_options(self):
        regions = _connectivity_regions()
        off = pc.trace_table(regions)
        assert list(off.columns) == TRACE_COLUMNS
        no_surrogates = pc.trace_table(regions, transfer_entropy=True, te_surrogates=0)
        assert no_surrogates[["te_ab", "te_ba"]].notna().all().all()
        assert not {"te_ab_z", "te_ba_z"} & set(no_surrogates.columns)
        seeded = pc.trace_table(regions, transfer_entropy=True, te_surrogates=10, seed=1)
        pd.testing.assert_frame_equal(seeded, pc.trace_table(regions, transfer_entropy=True, te_surrogates=10, seed=1))
        assert not seeded["te_ab_z"].equals(
            pc.trace_table(regions, transfer_entropy=True, te_surrogates=10, seed=2)["te_ab_z"]
        )

    def test_single_region(self):
        regions = _connectivity_regions()
        table = pc.trace_table(regions[regions["region"] == "L_MOp"])
        assert table.empty
        assert list(table.columns) == TRACE_COLUMNS


def test_connectivity_cmd(output_dir, tmp_path):
    path = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    _connectivity_perievent().to_csv(path, index=False)
    result = CliRunner().invoke(mesoscopy.cli, args=f"process connectivity {path} -o {output_dir}")
    assert result.exit_code == 0, result.output
    output = pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_connectivity.csv"
    assert f"Saved connectivity metrics at {output}" in result.output
    table = pd.read_csv(output)
    pd.testing.assert_frame_equal(table, pc.connectivity_table(pd.read_csv(path)), check_dtype=False)
    assert len(table) == 27
    assert not (pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_connectivity-rolling.csv").exists()


def test_connectivity_cmd_options(output_dir, tmp_path):
    path = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    _connectivity_perievent().to_csv(path, index=False)
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=f"process connectivity {path} -o {output_dir} --response 0 1 --min-trials 5 --no-mask-response"
        " --min-rt 0.5 --max-lag 0.2",
    )
    assert result.exit_code == 0, result.output
    table = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_connectivity.csv")
    expected = pc.connectivity_table(
        pd.read_csv(path), response=(0.0, 1.0), mask_response=False, min_rt=0.5, min_trials=5, max_lag=0.2
    )
    pd.testing.assert_frame_equal(table, expected, check_dtype=False)
    assert table[table["group"] == "sdt-miss"]["r"].notna().all()
    assert table["lag"].abs().max() <= 0.2 + 1e-9


def test_connectivity_cmd_response_pad(output_dir, tmp_path):
    path = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    _connectivity_perievent().to_csv(path, index=False)
    result = CliRunner().invoke(mesoscopy.cli, args=f"process connectivity {path} -o {output_dir} --response-pad 0.1")
    assert result.exit_code == 0, result.output
    table = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_connectivity.csv")
    pd.testing.assert_frame_equal(table, pc.connectivity_table(pd.read_csv(path), response_pad=0.1), check_dtype=False)


def test_connectivity_cmd_mutual_info_options(output_dir, tmp_path):
    path = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    _connectivity_perievent().to_csv(path, index=False)
    result = CliRunner().invoke(mesoscopy.cli, args=f"process connectivity {path} -o {output_dir} --mi-neighbours 5")
    assert result.exit_code == 0, result.output
    table = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_connectivity.csv")
    expected = pc.connectivity_table(pd.read_csv(path), mi_neighbours=5)
    pd.testing.assert_frame_equal(table, expected, check_dtype=False)
    assert table[table["group"] == "all"]["mi"].notna().all()

    result = CliRunner().invoke(mesoscopy.cli, args=f"process connectivity {path} -o {output_dir} --no-mi")
    assert result.exit_code == 0, result.output
    table = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_connectivity.csv")
    assert not {"mi", "mi_residual"} & set(table.columns)
    assert table["r"].notna().sum() == expected["r"].notna().sum()


def test_connectivity_cmd_transfer_entropy_options(output_dir, tmp_path):
    path = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    _connectivity_perievent().to_csv(path, index=False)
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=f"process connectivity {path} -o {output_dir} --with-te --te-history 2 --te-surrogates 10 --seed 3",
    )
    assert result.exit_code == 0, result.output
    table = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_connectivity.csv")
    expected = pc.connectivity_table(pd.read_csv(path), transfer_entropy=True, te_history=2, te_surrogates=10, seed=3)
    pd.testing.assert_frame_equal(table, expected, check_dtype=False)
    assert table[table["group"] == "all"][list(pc.TE_METRICS)].notna().all().all()

    result = CliRunner().invoke(mesoscopy.cli, args=f"process connectivity {path} -o {output_dir}")
    assert result.exit_code == 0, result.output
    table = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_connectivity.csv")
    assert list(table.columns) == DEFAULT_COLUMNS


def test_connectivity_cmd_rolling(output_dir, tmp_path):
    path = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    _connectivity_perievent().to_csv(path, index=False)
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=f"process connectivity {path} -o {output_dir} --with-rolling --rolling-window 0.2 --rolling-step 0.12"
        " --min-trials 5 --min-rt 0.5 --mi-neighbours 4 --no-mi",
    )
    assert result.exit_code == 0, result.output
    output = pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_connectivity-rolling.csv"
    assert f"Saved rolling connectivity metrics at {output}" in result.output
    assert (pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_connectivity.csv").exists()
    table = pd.read_csv(output)
    expected = pc.rolling_table(pd.read_csv(path), window=0.2, step=0.12, min_rt=0.5, min_trials=5, mi_neighbours=4)
    pd.testing.assert_frame_equal(table, expected, check_dtype=False)
    # --no-mi applies to the epoch table only.
    assert table[table["group"] == "all"]["mi"].notna().all()


def test_connectivity_cmd_rolling_window_too_long(output_dir, tmp_path):
    path = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    _connectivity_perievent().to_csv(path, index=False)
    result = CliRunner().invoke(
        mesoscopy.cli, args=f"process connectivity {path} -o {output_dir} --with-rolling --rolling-window 10"
    )
    assert result.exit_code == 1
    assert "longer than the 101 samples" in result.output
    assert "Check --rolling-window and --rolling-step." in result.output


def test_connectivity_cmd_rolling_warns_once(metrics_perievent_csv, output_dir):
    result = CliRunner().invoke(
        mesoscopy.cli, args=f"process connectivity {metrics_perievent_csv} -o {output_dir} --with-rolling"
    )
    assert result.exit_code == 0, result.output
    assert result.output.count("Warning: Skipping trial groups: no sdt_type column") == 1
    assert "Saved rolling connectivity metrics" in result.output


def test_connectivity_cmd_joins_trials(output_dir, tmp_path):
    perievent = _connectivity_perievent()
    columns = [column for column in perievent.columns if column not in pm.PERIEVENT_COLUMNS]
    path = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    perievent.drop(columns=columns).to_csv(path, index=False)
    trials_path = tmp_path / "ses-01_trials.csv"
    perievent.drop_duplicates("trial_index")[columns].to_csv(trials_path, index=False)
    result = CliRunner().invoke(mesoscopy.cli, args=f"process connectivity {path} -o {output_dir} -t {trials_path}")
    assert result.exit_code == 0, result.output
    assert "Warning" not in result.output
    table = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_connectivity.csv")
    expected = pc.connectivity_table(pd.read_csv(path), trials=pd.read_csv(trials_path))
    pd.testing.assert_frame_equal(table, expected, check_dtype=False)
    assert table["n_trials"].tolist() == pc.connectivity_table(perievent)["n_trials"].tolist()


def test_connectivity_cmd_warns_without_trials_columns(metrics_perievent_csv, output_dir):
    result = CliRunner().invoke(mesoscopy.cli, args=f"process connectivity {metrics_perievent_csv} -o {output_dir}")
    assert result.exit_code == 0, result.output
    assert "Warning: Using the response window as the epoch: no cue_onset, response_time column(s)" in result.output
    assert "Warning: Skipping trial groups: no sdt_type column" in result.output
    table = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_event-cueonset_connectivity.csv")
    assert table["group"].tolist() == ["all"]
    assert table["n_trials"].tolist() == [3]


def test_connectivity_cmd_missing_columns(output_dir, tmp_path):
    path = tmp_path / "bad_perievent.csv"
    _connectivity_perievent().drop(columns="F").to_csv(path, index=False)
    result = CliRunner().invoke(mesoscopy.cli, args=f"process connectivity {path} -o {output_dir}")
    assert result.exit_code == 1
    assert "lacks the F column(s)" in result.output


def test_connectivity_cmd_empty_input(output_dir, tmp_path):
    path = tmp_path / "empty_perievent.csv"
    pd.DataFrame(columns=list(pm.PERIEVENT_COLUMNS)).to_csv(path, index=False)
    result = CliRunner().invoke(mesoscopy.cli, args=f"process connectivity {path} -o {output_dir}")
    assert result.exit_code == 1
    assert "has no rows" in result.output


def test_connectivity_cmd_regions(output_dir, tmp_path):
    path = tmp_path / "ses-01_regions.csv"
    _connectivity_regions().to_csv(path, index=False)
    result = CliRunner().invoke(mesoscopy.cli, args=f"process connectivity {path} -o {output_dir} --max-lag 0.2")
    assert result.exit_code == 0, result.output
    assert "Loading region traces" in result.output
    output = pathlib.Path(output_dir) / "ses-01_regions_connectivity.csv"
    assert f"Saved connectivity metrics at {output}" in result.output
    table = pd.read_csv(output)
    pd.testing.assert_frame_equal(table, pc.trace_table(pd.read_csv(path), max_lag=0.2), check_dtype=False)
    assert table["group"].tolist() == ["all"] * 3


def test_connectivity_cmd_regions_ignores_trial_options(output_dir, tmp_path):
    path = tmp_path / "ses-01_regions.csv"
    _connectivity_regions().to_csv(path, index=False)
    trials_path = tmp_path / "ses-01_trials.csv"
    _gonogo_info().to_csv(trials_path, index=False)
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=f"process connectivity {path} -o {output_dir} -t {trials_path} --min-trials 1 --no-mask-response --no-mi",
    )
    assert result.exit_code == 0, result.output
    assert "Loading trials" not in result.output
    table = pd.read_csv(pathlib.Path(output_dir) / "ses-01_regions_connectivity.csv")
    assert "mi" not in table.columns
    assert table["group"].tolist() == ["all"] * 3


def test_connectivity_cmd_regions_skips_rolling(output_dir, tmp_path):
    path = tmp_path / "ses-01_regions.csv"
    _connectivity_regions().to_csv(path, index=False)
    result = CliRunner().invoke(mesoscopy.cli, args=f"process connectivity {path} -o {output_dir} --with-rolling")
    assert result.exit_code == 0, result.output
    assert "Warning: --with-rolling takes a peri-event file; the rolling table is skipped." in result.output
    assert (pathlib.Path(output_dir) / "ses-01_regions_connectivity.csv").exists()
    assert not (pathlib.Path(output_dir) / "ses-01_regions_connectivity-rolling.csv").exists()


def test_connectivity_cmd_regions_missing_columns(output_dir, tmp_path):
    path = tmp_path / "ses-01_regions.csv"
    _connectivity_regions().drop(columns="F").to_csv(path, index=False)
    result = CliRunner().invoke(mesoscopy.cli, args=f"process connectivity {path} -o {output_dir}")
    assert result.exit_code != 0
    assert "Missing columns: ['F']" in result.output
    assert "`process regions`" in result.output


def test_connectivity_cmd_creates_output_dir(tmp_path):
    path = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    _connectivity_perievent().to_csv(path, index=False)
    out_dir = tmp_path / "new" / "dir"
    result = CliRunner().invoke(mesoscopy.cli, args=f"process connectivity {path} -o {out_dir}")
    assert result.exit_code == 0, result.output
    assert (out_dir / "ses-01_regions_event-cueonset_connectivity.csv").exists()


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

DECODING_GROUPS = {"stim": ["all", "resp-push", "resp-nopush"], "response": ["all", "stim-go", "stim-nogo"]}


def _decoding_cube(seed=0):
    """Go/no-go traces for three regions: the first follows the stimulus, the second is noise, the third follows the
    lever push."""
    info = _gonogo_info()
    rng = np.random.default_rng(seed)
    # A step, so the epoch mean does not depend on the response time.
    shape = (METRICS_GRID >= 0).astype(float)
    go = np.isin(info["sdt_type"], pm.GO_TYPES)
    push = np.isin(info["sdt_type"], pm.PUSH_TYPES)
    noise = 0.1 * rng.standard_normal((3, len(info), len(METRICS_GRID)))
    return np.stack([go[:, None] * shape + noise[0], noise[1], push[:, None] * shape + noise[2]], axis=-1)


def _decoding_perievent(**kwargs):
    """Long-format cue-aligned peri-event table of `_decoding_cube` for `CONNECTIVITY_REGIONS`."""
    return _connectivity_perievent(_decoding_cube(**kwargs))


def _decoding_features(perievent=None, info=None, per_trial_end=False):
    """Epoch features of the go/no-go session, shape `(36, 3)`."""
    perievent = perievent if perievent is not None else _decoding_perievent()
    info = info if info is not None else _gonogo_info()
    start, end, keep = pc.epoch_bounds(info, "cue_onset", (0.0, 3.0))
    if not per_trial_end:
        end = pdc.median_end(info, end, keep)
    return pdc.epoch_features(perievent, info["trial_index"].to_numpy(), METRICS_GRID, CONNECTIVITY_REGIONS, start, end)


def _ramp_perievent(n_per_class=60):
    """Cue-aligned peri-event table of hits and misses for one region whose trace ramps up identically after the
    cue on every trial, so only the epoch length can tell the two apart. Hit response times are right-skewed."""
    n_trials = 2 * n_per_class
    cue = 10.0 * np.arange(n_trials) + 5.0
    response_time = np.full(n_trials, np.nan)
    response_time[:n_per_class] = 0.3 + 2.0 * np.linspace(0.0, 1.0, n_per_class) ** 2
    info = pd.DataFrame(
        {
            "trial_index": np.arange(n_trials),
            "event_time": cue,
            "start_time": cue - 5.0,
            "cue_onset": cue,
            "stop_time": cue + 3.0,
            "response_time": response_time,
            "sdt_type": ["hit"] * n_per_class + ["miss"] * n_per_class,
            "protocol": "gonogo",
        }
    )
    ramp = np.interp(METRICS_GRID, [0.0, 3.0], [0.0, 1.0]) * (METRICS_GRID >= 0)
    noise = 1e-3 * np.random.default_rng(0).standard_normal((n_trials, len(METRICS_GRID)))
    frames = [
        pd.DataFrame(
            {
                "trial_index": row["trial_index"],
                "event_time": row["event_time"],
                "time": METRICS_GRID,
                "region": "L_MOp",
                "F": ramp + noise[i],
            }
        )
        for i, row in info.iterrows()
    ]
    return pd.concat(frames, ignore_index=True).merge(info.drop(columns="event_time"), on="trial_index")


class TestTrialLabels:
    def test_stim(self):
        y, labelled = pdc.trial_labels(_gonogo_info(), "stim")
        assert y.tolist() == [1] * 18 + [0] * 18
        assert labelled.all()

    def test_response(self):
        y, labelled = pdc.trial_labels(_gonogo_info(), "response")
        assert y.tolist() == [1] * 12 + [0] * 6 + [1] * 12 + [0] * 6
        assert labelled.all()

    def test_unknown_type_is_unlabelled(self):
        info = _gonogo_info()
        info.loc[0, "sdt_type"] = "other"
        y, labelled = pdc.trial_labels(info, "stim")
        assert not labelled[0]
        assert labelled[1:].all()
        assert y[0] == 0

    def test_unknown_label_raises(self):
        with pytest.raises(ValueError, match="Unknown label 'reward'"):
            pdc.trial_labels(_gonogo_info(), "reward")


class TestEpochFeatures:
    def test_mean_over_epoch(self):
        perievent = _decoding_perievent()
        info = _gonogo_info()
        start, end, _ = pc.epoch_bounds(info, "cue_onset", (0.0, 3.0))
        cube = pc.epoch_cube(perievent, info["trial_index"].to_numpy(), METRICS_GRID, CONNECTIVITY_REGIONS, start, end)
        features = _decoding_features(perievent, info, per_trial_end=True)
        assert features.shape == (36, 3)
        np.testing.assert_allclose(features, np.nanmean(cube, axis=1))
        # Go trials respond in the first region, no trial does in the second.
        assert features[:18, 0].min() > features[18:, 0].max()
        assert abs(features[:, 1]).max() < 0.2

    def test_empty_epoch_is_nan(self):
        perievent = _decoding_perievent()
        info = _gonogo_info()
        start = np.zeros(len(info))
        end = np.where(info["trial_index"] == 0, 0.0, 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            features = pdc.epoch_features(
                perievent, info["trial_index"].to_numpy(), METRICS_GRID, CONNECTIVITY_REGIONS, start, end
            )
        assert np.isnan(features[0]).all()
        assert np.isfinite(features[1:]).all()


class TestMedianEnd:
    def test_common_end(self):
        info = _gonogo_info()
        start, end, keep = pc.epoch_bounds(info, "cue_onset", (0.0, 3.0))
        common = pdc.median_end(info, end, keep)
        # Responders end at their own time, the others at the median; the common end is that median.
        assert len(np.unique(end)) > 1
        assert common.tolist() == [pytest.approx(np.median(GONOGO_RT[:12] + GONOGO_RT[18:30]))] * 36
        assert common[0] == pytest.approx(end[12])
        assert start.tolist() == [0.0] * 36

    def test_median_over_kept_trials(self):
        info = _gonogo_info()
        start, end, keep = pc.epoch_bounds(info, "cue_onset", (0.0, 3.0), min_rt=0.5)
        assert keep.sum() == 34
        common = pdc.median_end(info, end, keep)
        assert common[0] == pytest.approx(np.median([rt for rt in GONOGO_RT[:12] + GONOGO_RT[18:30] if rt >= 0.5]))

    def test_clipped_to_window(self):
        info = _gonogo_info()
        _, end, keep = pc.epoch_bounds(info, "cue_onset", (0.0, 0.8))
        common = pdc.median_end(info, end, keep)
        assert common.tolist() == [pytest.approx(np.nextafter(0.8, np.inf))] * 36

    def test_without_response_times(self):
        info = _gonogo_info().drop(columns="response_time")
        end = np.arange(36, dtype=float)
        assert pdc.median_end(info, end, np.ones(36, dtype=bool)) is end

    def test_without_responders(self):
        info = _gonogo_info()
        info["response_time"] = np.nan
        end = np.arange(36, dtype=float)
        assert pdc.median_end(info, end, np.ones(36, dtype=bool)) is end


class TestMakeDecoder:
    def test_logistic(self):
        from sklearn.linear_model import LogisticRegression

        model = pdc.make_decoder("logistic")
        assert isinstance(model[-1], LogisticRegression)
        assert model[-1].class_weight == "balanced"

    def test_lda(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        model = pdc.make_decoder("lda")
        assert isinstance(model[-1], LinearDiscriminantAnalysis)
        assert model[-1].shrinkage == "auto"
        assert model[-1].priors == [0.5, 0.5]

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown decoder 'svm'"):
            pdc.make_decoder("svm")


class TestCrossValidationSplits:
    def test_stratified_partition(self):
        y = np.array([1] * 18 + [0] * 18)
        splits = pdc.cross_validation_splits(y, 5, 2, 42)
        assert len(splits) == 10
        for repeat in range(2):
            tests = [test for _, test in splits[repeat * 5 : (repeat + 1) * 5]]
            assert sorted(np.concatenate(tests).tolist()) == list(range(36))
            for train, test in splits[repeat * 5 : (repeat + 1) * 5]:
                assert set(y[test]) == {0, 1}
                assert not set(train) & set(test)

    def test_seed(self):
        y = np.array([1] * 18 + [0] * 18)
        same = pdc.cross_validation_splits(y, 5, 1, 0)
        again = pdc.cross_validation_splits(y, 5, 1, 0)
        other = pdc.cross_validation_splits(y, 5, 1, 1)
        assert all(np.array_equal(a[1], b[1]) for a, b in zip(same, again, strict=True))
        assert not all(np.array_equal(a[1], b[1]) for a, b in zip(same, other, strict=True))


class TestFitScores:
    @pytest.mark.parametrize("decoder", pdc.DECODERS)
    def test_separable_feature(self, decoder):
        y = np.array([1] * 18 + [0] * 18)
        features = np.where(y == 1, 1.0, 0.0)[:, None] + 0.01 * np.random.default_rng(0).standard_normal((36, 1))
        splits = pdc.cross_validation_splits(y, 5, 2, 42)
        scores = pdc.fit_scores(features, y, splits, decoder, 5)
        assert scores["accuracy"].shape == (10,)
        assert scores["accuracy"].tolist() == [1.0] * 10
        assert scores["balanced_accuracy"].tolist() == [1.0] * 10
        assert scores["auroc"].tolist() == [1.0, 1.0]
        assert (scores["tp"], scores["fn"], scores["fp"], scores["tn"]) == (18, 0, 0, 18)
        assert scores["weights"].shape == (10, 1)
        assert (scores["weights"] > 0).all()

    def test_counts_average_over_repeats(self):
        y = np.array([1] * 18 + [0] * 18)
        features = np.random.default_rng(0).standard_normal((36, 1))
        scores = pdc.fit_scores(features, y, pdc.cross_validation_splits(y, 5, 3, 42), "lda", 5)
        assert scores["tp"] + scores["fn"] == pytest.approx(18)
        assert scores["fp"] + scores["tn"] == pytest.approx(18)
        assert scores["auroc"].shape == (3,)


class TestConfusionCounts:
    def test_counts(self):
        y = np.array([1, 1, 1, 0, 0, 0, 0])
        predicted = np.array([1, 1, 0, 1, 0, 0, 0])
        assert pdc.confusion_counts(y, predicted).tolist() == [2, 1, 1, 3]


class TestDPrime:
    def test_known_rates(self):
        assert pdc.d_prime(8, 2, 3, 7) == pytest.approx(sst.norm.ppf(0.8) - sst.norm.ppf(0.3))

    def test_perfect_is_finite(self):
        assert pdc.d_prime(10, 0, 0, 10) == pytest.approx(2 * sst.norm.ppf(1 - 1 / 20))
        assert pdc.d_prime(0, 10, 10, 0) == pytest.approx(-2 * sst.norm.ppf(1 - 1 / 20))

    def test_missing_class_is_nan(self):
        assert np.isnan(pdc.d_prime(0, 0, 3, 7))
        assert np.isnan(pdc.d_prime(8, 2, 0, 0))


class TestFBeta:
    def test_known_counts(self):
        assert pdc.f_beta(8, 2, 3) == pytest.approx(40 / 51)
        assert pdc.f_beta(8, 2, 3, beta=1.0) == pytest.approx(16 / 21)

    def test_nothing_positive_is_nan(self):
        assert np.isnan(pdc.f_beta(0, 0, 0))


class TestScoreSummary:
    def test_summary(self):
        scores = {
            "accuracy": np.array([0.8, 1.0]),
            "balanced_accuracy": np.array([0.75, 1.0]),
            "auroc": np.array([0.9, 1.0]),
            "tp": 8.0,
            "fn": 2.0,
            "fp": 3.0,
            "tn": 7.0,
        }
        summary = pdc.score_summary(scores)
        assert list(summary) == list(pdc.SCORES)
        assert summary["accuracy"] == pytest.approx(0.9)
        assert summary["accuracy_sd"] == pytest.approx(np.std([0.8, 1.0], ddof=1))
        assert summary["balanced_accuracy"] == pytest.approx(0.875)
        assert summary["auroc"] == pytest.approx(0.95)
        assert summary["d_prime"] == pytest.approx(pdc.d_prime(8, 2, 3, 7))
        assert summary["f2"] == pytest.approx(40 / 51)
        assert (summary["tp"], summary["fn"], summary["fp"], summary["tn"]) == (8, 2, 3, 7)

    def test_single_fold_sd_is_nan(self):
        scores = {name: np.array([1.0]) for name in ("accuracy", "balanced_accuracy", "auroc")}
        summary = pdc.score_summary(scores | {"tp": 1.0, "fn": 0.0, "fp": 0.0, "tn": 1.0})
        assert np.isnan(summary["accuracy_sd"])
        assert np.isnan(summary["balanced_accuracy_sd"])


class TestUsableTrials:
    def test_nan_region_and_trial(self):
        features = np.ones((4, 3))
        features[:, 1] = np.nan
        features[2, 0] = np.nan
        complete, present = pdc.usable_trials(features)
        assert complete.tolist() == [True, True, False, True]
        assert present.tolist() == [True, False, True]

    def test_nothing_present(self):
        complete, present = pdc.usable_trials(np.full((4, 3), np.nan))
        assert not complete.any()
        assert not present.any()


class TestFoldIds:
    def test_matches_splits(self):
        y = np.array([1] * 18 + [0] * 18)
        splits = pdc.cross_validation_splits(y, 5, 2, 42)
        ids = pdc.fold_ids(splits, 5, 36)
        assert ids.shape == (2, 36)
        for i, (train, test) in enumerate(splits):
            assert (ids[i // 5, test] == i % 5).all()
            assert (ids[i // 5, train] != i % 5).all()


class TestRegionDecisions:
    @staticmethod
    def _problems():
        features = _decoding_features()
        y, _ = pdc.trial_labels(_gonogo_info(), "stim")
        splits = pdc.cross_validation_splits(y, 5, 2, 42)
        return features, y, splits, pdc.fold_ids(splits, 5, len(y))

    def test_lda_matches_sklearn(self):
        features, y, splits, fold_of = self._problems()
        decisions = pdc.region_decisions(features, np.broadcast_to(y, (2, 36)), fold_of, "lda")
        assert decisions.shape == (2, 36, 3)
        for region in range(3):
            for i, (train, test) in enumerate(splits):
                model = pdc.make_decoder("lda").fit(features[train][:, [region]], y[train])
                expected = model.decision_function(features[test][:, [region]])
                np.testing.assert_allclose(decisions[i // 5, test, region], expected, atol=1e-10)

    def test_logistic_matches_converged_sklearn(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        features, y, splits, fold_of = self._problems()
        decisions = pdc.region_decisions(features, np.broadcast_to(y, (2, 36)), fold_of, "logistic")
        for region in range(3):
            for i, (train, test) in enumerate(splits):
                model = make_pipeline(
                    StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", tol=1e-12, max_iter=10000)
                ).fit(features[train][:, [region]], y[train])
                expected = model.decision_function(features[test][:, [region]])
                np.testing.assert_allclose(decisions[i // 5, test, region], expected, atol=1e-6)

    @pytest.mark.parametrize("decoder", pdc.DECODERS)
    def test_problems_are_independent(self, decoder):
        features, y, _, fold_of = self._problems()
        shuffled = np.random.default_rng(0).permutation(y)
        labels = np.stack([y, shuffled])
        folds = np.stack([fold_of[0], fold_of[1]])
        together = pdc.region_decisions(features, labels, folds, decoder)
        alone = pdc.region_decisions(features, shuffled[None], folds[1:], decoder)
        np.testing.assert_allclose(together[1], alone[0])
        assert not np.allclose(together[0], together[1])

    @pytest.mark.parametrize("decoder", pdc.DECODERS)
    def test_constant_feature_predicts_nothing(self, decoder):
        _, y, _, fold_of = self._problems()
        decisions = pdc.region_decisions(np.ones((36, 1)), y[None], fold_of[:1], decoder)
        np.testing.assert_allclose(decisions, 0.0, atol=1e-12)

    @pytest.mark.parametrize("decoder", pdc.DECODERS)
    def test_near_constant_feature_predicts_nothing(self, decoder):
        # Constant up to rounding, which StandardScaler treats as constant too.
        _, y, _, fold_of = self._problems()
        features = 1.0 + 1e-17 * np.random.default_rng(0).standard_normal((36, 1))
        decisions = pdc.region_decisions(features, y[None], fold_of[:1], decoder)
        np.testing.assert_allclose(decisions, 0.0, atol=1e-12)

    @pytest.mark.parametrize("decoder", pdc.DECODERS)
    def test_separable_feature(self, decoder):
        _, y, _, fold_of = self._problems()
        features = y[:, None] + 0.01 * np.random.default_rng(0).standard_normal((36, 1))
        decisions = pdc.region_decisions(features, y[None], fold_of[:1], decoder)
        assert np.isfinite(decisions).all()
        assert ((decisions[0, :, 0] > 0) == (y == 1)).all()

    def test_unknown_decoder_raises(self):
        _, y, _, fold_of = self._problems()
        with pytest.raises(ValueError, match="Unknown decoder 'svm'"):
            pdc.region_decisions(np.ones((36, 1)), y[None], fold_of[:1], "svm")


class TestProblemScores:
    def test_matches_sklearn_metrics(self):
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(0)
        y = np.array([1] * 18 + [0] * 18)
        labels = np.stack([y, rng.permutation(y)])
        splits = [pdc.cross_validation_splits(row, 5, 1, i) for i, row in enumerate(labels)]
        fold_of = np.stack([pdc.fold_ids(s, 5, 36)[0] for s in splits])
        decisions = rng.standard_normal((2, 36, 3)) + labels[:, :, None]
        scores = pdc.problem_scores(decisions, labels, fold_of)
        assert scores["accuracy"].shape == (2, 5, 3)
        assert scores["auroc"].shape == (2, 3)
        for problem in range(2):
            for region in range(3):
                predicted = (decisions[problem, :, region] > 0).astype(int)
                truth = labels[problem]
                for fold in range(5):
                    test = fold_of[problem] == fold
                    assert scores["accuracy"][problem, fold, region] == pytest.approx(
                        np.mean(predicted[test] == truth[test])
                    )
                    assert scores["balanced_accuracy"][problem, fold, region] == pytest.approx(
                        balanced_accuracy_score(truth[test], predicted[test])
                    )
                assert scores["auroc"][problem, region] == pytest.approx(
                    roc_auc_score(truth, decisions[problem, :, region])
                )
                counts = [scores[name][problem, region] for name in ("tp", "fn", "fp", "tn")]
                assert counts == pdc.confusion_counts(truth, predicted).tolist()

    def test_ties_use_average_ranks(self):
        from sklearn.metrics import roc_auc_score

        y = np.array([1, 1, 1, 0, 0, 0])
        decisions = np.array([1.0, 0.5, 0.5, 0.5, 0.0, 0.0])[None, :, None]
        fold_of = np.array([[0, 1, 0, 1, 0, 1]])
        scores = pdc.problem_scores(decisions, y[None], fold_of)
        assert scores["auroc"][0, 0] == pytest.approx(roc_auc_score(y, decisions[0, :, 0]))


class TestNullSummary:
    def test_summary(self):
        null_accuracy = np.array([0.5, 0.6, 0.4, 0.5])
        null_balanced = np.array([0.5, 0.7, 0.45, 0.55])
        summary = pdc.null_summary(0.7, null_accuracy, null_balanced)
        assert list(summary) == list(pdc.NULL_SCORES)
        assert summary["shuffle_accuracy"] == pytest.approx(0.5)
        assert summary["shuffle_accuracy_sd"] == pytest.approx(np.std(null_accuracy, ddof=1))
        assert summary["shuffle_balanced_accuracy"] == pytest.approx(0.55)
        assert summary["shuffle_balanced_accuracy_sd"] == pytest.approx(np.std(null_balanced, ddof=1))
        assert summary["shuffle_balanced_accuracy_95"] == pytest.approx(np.percentile(null_balanced, 95))
        # One shuffle ties the observed value.
        assert summary["p_value"] == pytest.approx(2 / 5)

    def test_p_value_bounds(self):
        null = np.full(9, 0.5)
        assert pdc.null_summary(0.9, null, null)["p_value"] == pytest.approx(1 / 10)
        assert pdc.null_summary(0.1, null, null)["p_value"] == pytest.approx(1.0)
        assert np.isnan(pdc.null_summary(float("nan"), null, null)["p_value"])

    def test_single_shuffle_sd_is_nan(self):
        summary = pdc.null_summary(0.5, np.array([0.5]), np.array([0.5]))
        assert np.isnan(summary["shuffle_accuracy_sd"])
        assert np.isnan(summary["shuffle_balanced_accuracy_sd"])


class TestShuffledLabels:
    def test_permutations_with_splits(self):
        y = np.array([1] * 18 + [0] * 18)
        labels, splits = pdc.shuffled_labels(y, 5, 4, np.random.default_rng(0))
        assert labels.shape == (4, 36)
        assert (labels.sum(axis=1) == 18).all()
        assert not (labels == y).all(axis=1).any()
        assert len(splits) == 4
        for row, row_splits in zip(labels, splits, strict=True):
            assert len(row_splits) == 5
            for _, test in row_splits:
                assert set(row[test]) == {0, 1}

    def test_none(self):
        labels, splits = pdc.shuffled_labels(np.array([1, 0, 1, 0]), 2, 0, np.random.default_rng(0))
        assert labels.shape == (0, 4)
        assert splits == []


class TestDecodingTable:
    @staticmethod
    def _row(table, decoder, region, group):
        return table[(table["decoder"] == decoder) & (table["region"] == region) & (table["group"] == group)].iloc[0]

    def test_layout(self):
        tables = pdc.decoding_table(_decoding_perievent(), repeats=2, shuffles=0)
        table = tables.table
        regions = [*CONNECTIVITY_REGIONS, pdc.POPULATION]
        groups = DECODING_GROUPS["stim"]
        assert list(table.columns) == [column for column in pdc.TABLE_COLUMNS if column not in pdc.NULL_SCORES]
        assert table["decoder"].tolist() == [d for d in pdc.DECODERS for _ in regions for _ in groups]
        assert table["region"].tolist() == [r for _ in pdc.DECODERS for r in regions for _ in groups]
        assert table["group"].tolist() == groups * (len(regions) * len(pdc.DECODERS))
        assert table["label"].eq("stim").all()
        weights = tables.weights
        assert list(weights.columns) == list(pdc.WEIGHT_COLUMNS)
        assert weights["decoder"].tolist() == [d for d in pdc.DECODERS for _ in groups for _ in CONNECTIVITY_REGIONS]
        assert weights["group"].tolist() == [g for _ in pdc.DECODERS for g in groups for _ in CONNECTIVITY_REGIONS]
        assert weights["region"].tolist() == CONNECTIVITY_REGIONS * (len(groups) * len(pdc.DECODERS))

    def test_counts(self):
        table = pdc.decoding_table(_decoding_perievent(), repeats=2, shuffles=0).table
        counts = table.groupby("group", sort=False)[["n_trials", "n_class_a", "n_class_b"]].agg(["min", "max"])
        assert counts.loc["all"].tolist() == [36, 36, 18, 18, 18, 18]
        assert counts.loc["resp-push"].tolist() == [24, 24, 12, 12, 12, 12]
        assert counts.loc["resp-nopush"].tolist() == [12, 12, 6, 6, 6, 6]

    @pytest.mark.parametrize("decoder", pdc.DECODERS)
    def test_stimulus_region_decodes(self, decoder):
        table = pdc.decoding_table(_decoding_perievent(), repeats=2, shuffles=0).table
        row = self._row(table, decoder, "L_MOp", "all")
        assert row["balanced_accuracy"] > 0.9
        assert row["accuracy"] > 0.9
        assert row["auroc"] > 0.95
        assert row["d_prime"] > 2
        assert row["f2"] > 0.9
        assert row["tp"] + row["fn"] == pytest.approx(18)
        assert row["fp"] + row["tn"] == pytest.approx(18)
        assert row["balanced_accuracy_sd"] >= 0
        # The stimulus still decodes among the pushed trials, where hits meet false alarms.
        assert self._row(table, decoder, "L_MOp", "resp-push")["balanced_accuracy"] > 0.9
        # The population decoder follows the informative region.
        assert self._row(table, decoder, pdc.POPULATION, "all")["balanced_accuracy"] > 0.9

    @pytest.mark.parametrize("decoder", pdc.DECODERS)
    def test_uninformative_regions_near_chance(self, decoder):
        table = pdc.decoding_table(_decoding_perievent(), repeats=2, shuffles=0).table
        # Cross-validated scores on uninformative features fall at or below chance.
        noise = self._row(table, decoder, "R_MOp", "all")
        assert noise["balanced_accuracy"] < 0.65
        assert noise["auroc"] < 0.7
        assert noise["d_prime"] < 1
        # The push region does not tell the stimuli apart among the pushed trials.
        push = self._row(table, decoder, "L_SSp-ul", "resp-push")
        assert push["balanced_accuracy"] < 0.7

    def test_response_label(self):
        tables = pdc.decoding_table(_decoding_perievent(), label="response", min_trials=5, repeats=2, shuffles=0)
        table = tables.table
        assert table["label"].eq("response").all()
        assert table["group"].unique().tolist() == DECODING_GROUPS["response"]
        counts = table.groupby("group", sort=False)[["n_trials", "n_class_a", "n_class_b"]].first()
        assert counts.loc["all"].tolist() == [36, 24, 12]
        assert counts.loc["stim-go"].tolist() == [18, 12, 6]
        assert counts.loc["stim-nogo"].tolist() == [18, 12, 6]
        for decoder in pdc.DECODERS:
            assert self._row(table, decoder, "L_SSp-ul", "all")["balanced_accuracy"] > 0.9
            assert self._row(table, decoder, "L_SSp-ul", "stim-go")["balanced_accuracy"] > 0.9
            # Every trial of a go group saw the go stimulus.
            assert abs(self._row(table, decoder, "L_MOp", "stim-go")["balanced_accuracy"] - 0.5) < 0.35
        assert tables.weights["label"].eq("response").all()

    def test_min_trials(self):
        tables = pdc.decoding_table(_decoding_perievent(), repeats=2, shuffles=0)
        rows = tables.table[tables.table["group"] == "resp-nopush"]
        assert rows[list(pdc.SCORES)].isna().all().all()
        assert rows["n_trials"].tolist() == [12] * 8
        weights = tables.weights[tables.weights["group"] == "resp-nopush"]
        assert weights[["weight", "weight_sd"]].isna().all().all()

        lowered = pdc.decoding_table(_decoding_perievent(), min_trials=5, repeats=2, shuffles=0)
        rows = lowered.table[lowered.table["group"] == "resp-nopush"]
        assert rows[list(pdc.SCORES)].notna().all().all()
        assert self._row(lowered.table, "lda", "L_MOp", "resp-nopush")["balanced_accuracy"] > 0.8
        assert lowered.weights[lowered.weights["group"] == "resp-nopush"]["weight"].notna().all()

        # Every group, including all trials, needs the trials.
        strict = pdc.decoding_table(_decoding_perievent(), min_trials=20, repeats=2, shuffles=0)
        assert strict.table[list(pdc.SCORES)].isna().all().all()
        assert strict.table["n_trials"].tolist() == tables.table["n_trials"].tolist()

    def test_min_rt_drops_trials(self):
        table = pdc.decoding_table(_decoding_perievent(), min_rt=0.5, repeats=2, shuffles=0).table
        # Two hits respond at 0.4 s.
        assert table[table["group"] == "all"]["n_trials"].tolist() == [34] * 8
        assert table[table["group"] == "all"]["n_class_a"].tolist() == [16] * 8
        assert table[table["group"] == "resp-push"]["n_trials"].tolist() == [22] * 8

    def test_weights_favour_informative_region(self):
        weights = pdc.decoding_table(_decoding_perievent(), repeats=2, shuffles=0).weights
        for decoder in pdc.DECODERS:
            rows = weights[(weights["decoder"] == decoder) & (weights["group"] == "all")].set_index("region")
            assert rows.loc["L_MOp", "weight"] > 0
            assert rows.loc["L_MOp", "weight"] > abs(rows.loc["R_MOp", "weight"])
            assert rows.loc["L_MOp", "weight"] > abs(rows.loc["L_SSp-ul", "weight"])
            assert (rows["weight_sd"] >= 0).all()

    def test_matches_fit_scores(self):
        perievent = _decoding_perievent()
        table = pdc.decoding_table(perievent, repeats=2, shuffles=0).table
        features = _decoding_features(perievent)
        y, _ = pdc.trial_labels(_gonogo_info(), "stim")
        splits = pdc.cross_validation_splits(y, 5, 2, 42)
        for decoder in pdc.DECODERS:
            population = pdc.score_summary(pdc.fit_scores(features, y, splits, decoder, 5))
            row = self._row(table, decoder, pdc.POPULATION, "all")
            for score in pdc.SCORES:
                assert row[score] == pytest.approx(population[score])
        # The one-feature linear discriminant is the same fit in numpy.
        expected = pdc.score_summary(pdc.fit_scores(features[:, [0]], y, splits, "lda", 5))
        row = self._row(table, "lda", "L_MOp", "all")
        for score in pdc.SCORES:
            assert row[score] == pytest.approx(expected[score])

    def test_seed(self):
        first = pdc.decoding_table(_decoding_perievent(), repeats=2, shuffles=0)
        again = pdc.decoding_table(_decoding_perievent(), repeats=2, shuffles=0)
        pd.testing.assert_frame_equal(first.table, again.table)
        pd.testing.assert_frame_equal(first.weights, again.weights)
        other = pdc.decoding_table(_decoding_perievent(), repeats=2, shuffles=0, seed=1)
        assert not first.table["accuracy_sd"].equals(other.table["accuracy_sd"])

    def test_nan_region_left_out(self):
        perievent = _decoding_perievent()
        perievent.loc[perievent["region"] == "R_MOp", "F"] = np.nan
        tables = pdc.decoding_table(perievent, repeats=2, shuffles=0)
        rows = tables.table[tables.table["region"] == "R_MOp"]
        assert rows[list(pdc.SCORES)].isna().all().all()
        assert rows["n_trials"].tolist() == [36, 24, 12] * 2
        assert self._row(tables.table, "lda", pdc.POPULATION, "all")["balanced_accuracy"] > 0.9
        weights = tables.weights[tables.weights["group"] == "all"].set_index("region")
        assert weights["weight"].isna().tolist() == [False, True, False] * 2

    def test_all_nan_is_empty(self):
        perievent = _decoding_perievent()
        perievent["F"] = np.nan
        tables = pdc.decoding_table(perievent, repeats=2, shuffles=5)
        # The columns follow the options, not the data.
        assert list(tables.table.columns) == list(pdc.TABLE_COLUMNS)
        assert tables.table[[*pdc.SCORES, *pdc.NULL_SCORES]].isna().all().all()
        assert tables.table["n_trials"].eq(0).all()
        assert tables.weights["weight"].isna().all()

    def test_nan_trial_dropped(self):
        perievent = _decoding_perievent()
        perievent.loc[(perievent["trial_index"] == 0) & (perievent["region"] == "R_MOp"), "F"] = np.nan
        table = pdc.decoding_table(perievent, repeats=2, shuffles=0).table
        assert table[table["group"] == "all"]["n_trials"].tolist() == [35] * 8
        assert table[table["group"] == "all"]["n_class_a"].tolist() == [17] * 8

    def test_joins_trials(self):
        info = _gonogo_info()
        columns = [column for column in info.columns if column not in {"trial_index", "event_time"}]
        joined = pdc.decoding_table(
            _decoding_perievent().drop(columns=columns), trials=info[columns], repeats=2, shuffles=0
        )
        expected = pdc.decoding_table(_decoding_perievent(), repeats=2, shuffles=0)
        pd.testing.assert_frame_equal(joined.table, expected.table)
        pd.testing.assert_frame_equal(joined.weights, expected.weights)

    def test_without_epoch_columns_uses_response_window(self):
        perievent = _decoding_perievent().drop(columns=["cue_onset", "response_time"])
        with pytest.warns(UserWarning, match="Using the response window as the epoch"):
            table = pdc.decoding_table(perievent, response=(0.0, 1.0), repeats=2, shuffles=0).table
        assert self._row(table, "lda", "L_MOp", "all")["balanced_accuracy"] > 0.9

    def test_missing_sdt_type_raises(self):
        with pytest.raises(ValueError, match="Cannot label the trials: no sdt_type column"):
            pdc.decoding_table(_decoding_perievent().drop(columns=["sdt_type"]))

    @pytest.mark.parametrize(
        ("options", "match"),
        [
            ({"label": "reward"}, "Unknown label 'reward'"),
            ({"folds": 1}, "folds must be at least 2"),
            ({"repeats": 0}, "repeats must be at least 1"),
            ({"min_trials": 3}, r"min_trials must be at least folds \(5\)"),
            ({"shuffles": -1}, "shuffles must be at least 0"),
        ],
    )
    def test_bad_options_raise(self, options, match):
        with pytest.raises(ValueError, match=match):
            pdc.decoding_table(_decoding_perievent(), **options)

    def test_missing_columns_raises(self):
        with pytest.raises(ValueError, match="lacks the F column"):
            pdc.decoding_table(_decoding_perievent().drop(columns="F"))

    def test_per_trial_end(self):
        perievent = _decoding_perievent()
        table = pdc.decoding_table(perievent, per_trial_end=True, repeats=2, shuffles=0).table
        features = _decoding_features(perievent, per_trial_end=True)
        y, _ = pdc.trial_labels(_gonogo_info(), "stim")
        splits = pdc.cross_validation_splits(y, 5, 2, 42)
        expected = pdc.score_summary(pdc.fit_scores(features[:, [0]], y, splits, "lda", 5))
        row = self._row(table, "lda", "L_MOp", "all")
        for score in pdc.SCORES:
            assert row[score] == pytest.approx(expected[score])
        default = pdc.decoding_table(perievent, repeats=2, shuffles=0).table
        assert not table["accuracy"].equals(default["accuracy"])

    def test_common_end_removes_epoch_length(self):
        # With per-trial ends every miss ends at the median response time while the hits spread around it, so a
        # ramp that is identical on every trial decodes the push from the epoch length alone.
        perievent = _ramp_perievent()
        own = pdc.decoding_table(perievent, label="response", per_trial_end=True, repeats=2, shuffles=0).table
        common = pdc.decoding_table(perievent, label="response", repeats=2, shuffles=0).table
        unmasked = pdc.decoding_table(perievent, label="response", mask_response=False, repeats=2, shuffles=0).table
        for decoder in pdc.DECODERS:
            artefact = self._row(own, decoder, "L_MOp", "stim-go")["balanced_accuracy"]
            assert artefact > 0.65
            for table in (common, unmasked):
                row = self._row(table, decoder, "L_MOp", "stim-go")
                assert row["balanced_accuracy"] < 0.6
                assert row["auroc"] < 0.6
                assert artefact - row["balanced_accuracy"] > 0.1

    def test_null(self):
        tables = pdc.decoding_table(_decoding_perievent(), repeats=2, shuffles=20)
        table = tables.table
        assert list(table.columns) == list(pdc.TABLE_COLUMNS)
        for decoder in pdc.DECODERS:
            for region in ("L_MOp", pdc.POPULATION):
                row = self._row(table, decoder, region, "all")
                assert row["p_value"] == pytest.approx(1 / 21)
                assert row["balanced_accuracy"] > row["shuffle_balanced_accuracy_95"]
            noise = self._row(table, decoder, "R_MOp", "all")
            assert noise["p_value"] > 0.1
            assert noise["balanced_accuracy"] < noise["shuffle_balanced_accuracy_95"]
        fitted = table[table["group"] != "resp-nopush"]
        assert fitted["shuffle_balanced_accuracy"].between(0.4, 0.6).all()
        assert fitted["shuffle_accuracy"].between(0.4, 0.6).all()
        assert (fitted["shuffle_balanced_accuracy_95"] >= fitted["shuffle_balanced_accuracy"]).all()
        assert (fitted["shuffle_balanced_accuracy_sd"] > 0).all()
        assert fitted["p_value"].between(1 / 21, 1).all()
        empty = table[table["group"] == "resp-nopush"]
        assert empty[list(pdc.NULL_SCORES)].isna().all().all()

    def test_null_seed(self):
        first = pdc.decoding_table(_decoding_perievent(), repeats=2, shuffles=10).table
        again = pdc.decoding_table(_decoding_perievent(), repeats=2, shuffles=10).table
        pd.testing.assert_frame_equal(first, again)
        other = pdc.decoding_table(_decoding_perievent(), repeats=2, shuffles=10, seed=1).table
        assert not first["shuffle_balanced_accuracy"].equals(other["shuffle_balanced_accuracy"])


class TestWindowFeatures:
    def test_means(self):
        cube = np.random.default_rng(0).standard_normal((4, 10, 2))
        starts = pc.window_starts(10, 4, 3)
        features = pdc.window_features(cube, starts, 4)
        assert starts.tolist() == [0, 3, 6]
        assert features.shape == (4, 3, 2)
        for i, start in enumerate(starts):
            np.testing.assert_allclose(features[:, i], cube[:, start : start + 4].mean(axis=1))


ROLLING_STARTS = pc.window_starts(len(METRICS_GRID), 12, 2)
ROLLING_CENTRES = (METRICS_GRID[ROLLING_STARTS] + METRICS_GRID[ROLLING_STARTS + 11]) / 2


def _rolling_features(perievent=None):
    """Window features of the go/no-go session, shape `(36, n_windows, 3)`."""
    perievent = perievent if perievent is not None else _decoding_perievent()
    info = _gonogo_info()
    unbounded = np.full(len(info), np.inf)
    cube = pc.epoch_cube(
        perievent, info["trial_index"].to_numpy(), METRICS_GRID, CONNECTIVITY_REGIONS, -unbounded, unbounded
    )
    return pdc.window_features(cube, ROLLING_STARTS, 12)


@pytest.fixture(scope="module")
def decoding_rolling_default():
    """Rolling table of the go/no-go session with the default windows, two repeats and no shuffles."""
    return pdc.rolling_table(_decoding_perievent(), repeats=2, shuffles=0)


class TestDecodingRollingTable:
    @staticmethod
    def _rows(table, decoder, region, group):
        rows = table[(table["decoder"] == decoder) & (table["region"] == region) & (table["group"] == group)]
        return rows.set_index("time")

    def test_layout(self, decoding_rolling_default):
        table = decoding_rolling_default
        regions = [*CONNECTIVITY_REGIONS, pdc.POPULATION]
        groups = DECODING_GROUPS["stim"]
        windows = range(len(ROLLING_STARTS))
        assert list(table.columns) == [column for column in pdc.ROLLING_COLUMNS if column not in pdc.NULL_SCORES]
        assert len(table) == len(pdc.DECODERS) * len(regions) * len(groups) * len(windows)
        assert table["decoder"].tolist() == [d for d in pdc.DECODERS for _ in regions for _ in groups for _ in windows]
        assert table["region"].tolist() == [r for _ in pdc.DECODERS for r in regions for _ in groups for _ in windows]
        assert table["group"].tolist() == [g for _ in pdc.DECODERS for _ in regions for g in groups for _ in windows]
        np.testing.assert_allclose(
            table["time"], np.tile(ROLLING_CENTRES, len(pdc.DECODERS) * len(regions) * len(groups))
        )
        assert table["time"].iloc[0] == pytest.approx(-0.78)
        assert table["label"].eq("stim").all()

    def test_counts(self, decoding_rolling_default):
        table = decoding_rolling_default
        counts = table.groupby("group", sort=False)[["n_trials", "n_class_a", "n_class_b"]].agg(["min", "max"])
        assert counts.loc["all"].tolist() == [36, 36, 18, 18, 18, 18]
        assert counts.loc["resp-push"].tolist() == [24, 24, 12, 12, 12, 12]
        assert counts.loc["resp-nopush"].tolist() == [12, 12, 6, 6, 6, 6]
        assert table[table["group"] == "resp-nopush"][list(pdc.SCORES)].isna().all().all()
        assert table[table["group"] != "resp-nopush"][list(pdc.SCORES)].notna().all().all()

    @pytest.mark.parametrize("decoder", pdc.DECODERS)
    def test_time_course(self, decoder, decoding_rolling_default):
        table = decoding_rolling_default
        for region in ("L_MOp", pdc.POPULATION):
            rows = self._rows(table, decoder, region, "all")
            # Windows wholly before the step carry nothing; windows wholly after it separate the stimuli.
            assert rows[rows.index < -0.3]["balanced_accuracy"].mean() < 0.6
            assert rows[rows.index < -0.3]["balanced_accuracy"].max() < 0.8
            assert (rows[rows.index > 0.3]["balanced_accuracy"] > 0.9).all()
            assert (rows[rows.index > 0.3]["auroc"] > 0.95).all()
        for region in ("R_MOp", "L_SSp-ul"):
            rows = self._rows(table, decoder, region, "all")
            assert rows["balanced_accuracy"].mean() < 0.6
            assert rows["auroc"].mean() < 0.65

    def test_response_label(self):
        table = pdc.rolling_table(_decoding_perievent(), label="response", min_trials=5, repeats=2, shuffles=0)
        assert table["label"].eq("response").all()
        assert table["group"].unique().tolist() == DECODING_GROUPS["response"]
        for group in ("all", "stim-go"):
            rows = self._rows(table, "lda", "L_SSp-ul", group)
            assert (rows[rows.index > 0.3]["balanced_accuracy"] > 0.9).all()
        # Every trial of the go group saw the go stimulus.
        assert self._rows(table, "lda", "L_MOp", "stim-go")["balanced_accuracy"].mean() < 0.6

    def test_matches_window_fits(self, decoding_rolling_default):
        table = decoding_rolling_default
        features = _rolling_features()
        y, _ = pdc.trial_labels(_gonogo_info(), "stim")
        splits = pdc.cross_validation_splits(y, 5, 2, 42)
        for w in (0, 20, len(ROLLING_STARTS) - 1):
            expected = pdc.score_summary(pdc.fit_scores(features[:, w, [0]], y, splits, "lda", 5))
            row = self._rows(table, "lda", "L_MOp", "all").iloc[w]
            for score in pdc.SCORES:
                assert row[score] == pytest.approx(expected[score])
            for decoder in pdc.DECODERS:
                expected = pdc.score_summary(pdc.fit_scores(features[:, w], y, splits, decoder, 5))
                row = self._rows(table, decoder, pdc.POPULATION, "all").iloc[w]
                for score in pdc.SCORES:
                    assert row[score] == pytest.approx(expected[score])

    def test_null(self):
        table = pdc.rolling_table(_decoding_perievent(), repeats=2, shuffles=20)
        assert list(table.columns) == list(pdc.ROLLING_COLUMNS)
        shared = [column for column in pdc.NULL_SCORES if column != "p_value"]
        for decoder in pdc.DECODERS:
            for region in ("L_MOp", pdc.POPULATION):
                rows = self._rows(table, decoder, region, "all")
                # One null per decoder, region and group, read against every window.
                assert (rows[shared].nunique() == 1).all()
                assert 0.35 < rows["shuffle_balanced_accuracy"].iloc[0] < 0.65
                assert rows["shuffle_balanced_accuracy_95"].iloc[0] >= rows["shuffle_balanced_accuracy"].iloc[0]
                assert np.allclose(rows[rows.index > 0.3]["p_value"], 1 / 21)
                assert rows[rows.index < -0.3]["p_value"].median() > 0.1
        fitted = table[table["group"] != "resp-nopush"]
        assert fitted["p_value"].between(1 / 21, 1).all()
        assert table[table["group"] == "resp-nopush"][list(pdc.NULL_SCORES)].isna().all().all()

    def test_baseline(self):
        perievent = _decoding_perievent()
        default = pdc.rolling_table(perievent, repeats=2, shuffles=10)
        explicit = pdc.rolling_table(perievent, baseline=(-1.0, 0.0), repeats=2, shuffles=10)
        pd.testing.assert_frame_equal(default, explicit)
        shorter = pdc.rolling_table(perievent, baseline=(-0.5, 0.0), repeats=2, shuffles=10)
        pd.testing.assert_frame_equal(default[list(pdc.SCORES)], shorter[list(pdc.SCORES)])
        assert not default["shuffle_balanced_accuracy"].equals(shorter["shuffle_balanced_accuracy"])
        with pytest.raises(ValueError, match=r"No sample within the baseline window \[5.0, 6.0\)"):
            pdc.rolling_table(perievent, baseline=(5.0, 6.0), repeats=2, shuffles=10)
        with pytest.raises(ValueError, match="No sample within the baseline window"):
            pdc.rolling_table(perievent[perievent["time"] >= 0], repeats=2, shuffles=10)

    def test_window_rounding(self, decoding_rolling_default):
        perievent = _decoding_perievent()
        rounded = pdc.rolling_table(perievent, window=0.49, step=0.09, repeats=2, shuffles=0)
        pd.testing.assert_frame_equal(decoding_rolling_default, rounded)
        wider = pdc.rolling_table(perievent, window=1.0, step=0.5, repeats=2, shuffles=0)
        starts = pc.window_starts(len(METRICS_GRID), 25, 12)
        assert wider["time"].nunique() == len(starts)
        assert wider["time"].iloc[0] == pytest.approx((METRICS_GRID[0] + METRICS_GRID[24]) / 2)

    def test_window_errors(self):
        with pytest.raises(ValueError, match="longer than the 101 samples"):
            pdc.rolling_table(_decoding_perievent(), window=10.0, repeats=2, shuffles=0)
        with pytest.raises(ValueError, match="at least one sample"):
            pdc.rolling_table(_decoding_perievent(), step=0.001, repeats=2, shuffles=0)

    def test_min_rt_drops_trials(self):
        table = pdc.rolling_table(_decoding_perievent(), min_rt=0.5, repeats=2, shuffles=0)
        assert table[table["group"] == "all"]["n_trials"].eq(34).all()
        assert table[table["group"] == "all"]["n_class_a"].eq(16).all()

    def test_joins_trials(self, decoding_rolling_default):
        info = _gonogo_info()
        columns = [column for column in info.columns if column not in {"trial_index", "event_time"}]
        joined = pdc.rolling_table(
            _decoding_perievent().drop(columns=columns), trials=info[columns], repeats=2, shuffles=0
        )
        pd.testing.assert_frame_equal(joined, decoding_rolling_default)

    def test_seed(self):
        first = pdc.rolling_table(_decoding_perievent(), repeats=2, shuffles=5)
        again = pdc.rolling_table(_decoding_perievent(), repeats=2, shuffles=5)
        pd.testing.assert_frame_equal(first, again)
        other = pdc.rolling_table(_decoding_perievent(), repeats=2, shuffles=5, seed=1)
        assert not first["accuracy_sd"].equals(other["accuracy_sd"])

    def test_nan_sample_kept(self, decoding_rolling_default):
        perievent = _decoding_perievent()
        first = (perievent["trial_index"] == 0) & (perievent["region"] == "R_MOp")
        perievent.loc[first & (perievent["time"] == METRICS_GRID[0]), "F"] = np.nan
        table = pdc.rolling_table(perievent, repeats=2, shuffles=0)
        assert table["n_trials"].tolist() == decoding_rolling_default["n_trials"].tolist()
        # The other regions are untouched.
        rows = self._rows(table, "lda", "L_MOp", "all")
        pd.testing.assert_frame_equal(rows, self._rows(decoding_rolling_default, "lda", "L_MOp", "all"))

    def test_nan_region_left_out(self):
        perievent = _decoding_perievent()
        perievent.loc[perievent["region"] == "R_MOp", "F"] = np.nan
        table = pdc.rolling_table(perievent, repeats=2, shuffles=0)
        assert table[table["region"] == "R_MOp"][list(pdc.SCORES)].isna().all().all()
        assert table[table["group"] == "all"]["n_trials"].eq(36).all()
        rows = self._rows(table, "lda", pdc.POPULATION, "all")
        assert (rows[rows.index > 0.3]["balanced_accuracy"] > 0.9).all()

    def test_missing_sdt_type_raises(self):
        with pytest.raises(ValueError, match="Cannot label the trials: no sdt_type column"):
            pdc.rolling_table(_decoding_perievent().drop(columns=["sdt_type"]), repeats=2, shuffles=0)

    @pytest.mark.parametrize(
        ("options", "match"),
        [
            ({"label": "reward"}, "Unknown label 'reward'"),
            ({"folds": 1}, "folds must be at least 2"),
            ({"shuffles": -1}, "shuffles must be at least 0"),
        ],
    )
    def test_bad_options_raise(self, options, match):
        with pytest.raises(ValueError, match=match):
            pdc.rolling_table(_decoding_perievent(), **options)


def _decode_paths(output_dir, label="stim"):
    stem = pathlib.Path(output_dir) / f"ses-01_regions_event-cueonset_label-{label}_decoding"
    return (
        stem.with_name(stem.name + ".csv"),
        stem.with_name(stem.name + "-weights.csv"),
        stem.with_name(stem.name + "-rolling.csv"),
    )


def test_decode_cmd(output_dir, tmp_path):
    path = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    _decoding_perievent().to_csv(path, index=False)
    result = CliRunner().invoke(mesoscopy.cli, args=f"process decode {path} -o {output_dir} --repeats 2 --shuffles 10")
    assert result.exit_code == 0, result.output
    table_path, weights_path, rolling_path = _decode_paths(output_dir)
    assert f"Saved decoding scores at {table_path}" in result.output
    assert f"Saved population weights at {weights_path}" in result.output
    assert "Warning" not in result.output
    expected = pdc.decoding_table(pd.read_csv(path), repeats=2, shuffles=10)
    pd.testing.assert_frame_equal(pd.read_csv(table_path), expected.table, check_dtype=False)
    pd.testing.assert_frame_equal(pd.read_csv(weights_path), expected.weights, check_dtype=False)
    assert list(pd.read_csv(table_path).columns) == list(pdc.TABLE_COLUMNS)
    assert not rolling_path.exists()


def test_decode_cmd_options(output_dir, tmp_path):
    path = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    _decoding_perievent().to_csv(path, index=False)
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=f"process decode {path} -o {output_dir} -l response --response 0 1 --min-trials 5 --no-mask-response"
        " --min-rt 0.5 --response-pad 0.2 --folds 4 --repeats 2 --shuffles 0 --seed 1",
    )
    assert result.exit_code == 0, result.output
    table_path, weights_path, _ = _decode_paths(output_dir, "response")
    expected = pdc.decoding_table(
        pd.read_csv(path),
        label="response",
        response=(0.0, 1.0),
        mask_response=False,
        min_rt=0.5,
        response_pad=0.2,
        min_trials=5,
        folds=4,
        repeats=2,
        shuffles=0,
        seed=1,
    )
    table = pd.read_csv(table_path)
    pd.testing.assert_frame_equal(table, expected.table, check_dtype=False)
    pd.testing.assert_frame_equal(pd.read_csv(weights_path), expected.weights, check_dtype=False)
    assert "p_value" not in table.columns
    assert table["label"].eq("response").all()
    assert table["group"].unique().tolist() == DECODING_GROUPS["response"]


def test_decode_cmd_per_trial_end(output_dir, tmp_path):
    path = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    _decoding_perievent().to_csv(path, index=False)
    result = CliRunner().invoke(
        mesoscopy.cli, args=f"process decode {path} -o {output_dir} --per-trial-end --repeats 2 --shuffles 0"
    )
    assert result.exit_code == 0, result.output
    table = pd.read_csv(_decode_paths(output_dir)[0])
    expected = pdc.decoding_table(pd.read_csv(path), per_trial_end=True, repeats=2, shuffles=0).table
    pd.testing.assert_frame_equal(table, expected, check_dtype=False)
    default = pdc.decoding_table(pd.read_csv(path), repeats=2, shuffles=0).table
    assert not table["accuracy"].equals(default["accuracy"])


def test_decode_cmd_rolling(output_dir, tmp_path):
    path = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    _decoding_perievent().to_csv(path, index=False)
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=f"process decode {path} -o {output_dir} --with-rolling --rolling-window 1 --rolling-step 0.5"
        " --baseline -0.5 0 --repeats 2 --shuffles 5",
    )
    assert result.exit_code == 0, result.output
    table_path, _, rolling_path = _decode_paths(output_dir)
    assert f"Saved rolling decoding scores at {rolling_path}" in result.output
    assert table_path.exists()
    expected = pdc.rolling_table(pd.read_csv(path), window=1.0, step=0.5, baseline=(-0.5, 0.0), repeats=2, shuffles=5)
    pd.testing.assert_frame_equal(pd.read_csv(rolling_path), expected, check_dtype=False)


def test_decode_cmd_rolling_window_too_long(output_dir, tmp_path):
    path = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    _decoding_perievent().to_csv(path, index=False)
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=f"process decode {path} -o {output_dir} --with-rolling --rolling-window 10 --repeats 2 --shuffles 0",
    )
    assert result.exit_code == 1
    assert "longer than the 101 samples" in result.output
    assert "Check --rolling-window, --rolling-step and --baseline." in result.output
    assert _decode_paths(output_dir)[0].exists()


def test_decode_cmd_joins_trials(output_dir, tmp_path):
    perievent = _decoding_perievent()
    columns = [column for column in perievent.columns if column not in pm.PERIEVENT_COLUMNS]
    path = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    perievent.drop(columns=columns).to_csv(path, index=False)
    trials_path = tmp_path / "ses-01_trials.csv"
    perievent.drop_duplicates("trial_index")[columns].to_csv(trials_path, index=False)
    result = CliRunner().invoke(
        mesoscopy.cli, args=f"process decode {path} -o {output_dir} -t {trials_path} --repeats 2 --shuffles 0"
    )
    assert result.exit_code == 0, result.output
    assert "Loading trials" in result.output
    table = pd.read_csv(_decode_paths(output_dir)[0])
    expected = pdc.decoding_table(perievent, repeats=2, shuffles=0).table
    pd.testing.assert_frame_equal(table, expected, check_dtype=False)


def test_decode_cmd_without_sdt_type(output_dir, tmp_path):
    path = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    _decoding_perievent().drop(columns="sdt_type").to_csv(path, index=False)
    result = CliRunner().invoke(mesoscopy.cli, args=f"process decode {path} -o {output_dir} --repeats 2 --shuffles 0")
    assert result.exit_code == 1
    assert "Cannot label the trials: no sdt_type column; pass the trials CSV." in result.output
    assert "with the trials columns" in result.output


def test_decode_cmd_warns_without_epoch_columns(output_dir, tmp_path):
    path = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    _decoding_perievent().drop(columns=["cue_onset", "response_time"]).to_csv(path, index=False)
    result = CliRunner().invoke(
        mesoscopy.cli, args=f"process decode {path} -o {output_dir} --with-rolling --repeats 2 --shuffles 0"
    )
    assert result.exit_code == 0, result.output
    assert result.output.count("Warning: Using the response window as the epoch") == 1


def test_decode_cmd_missing_columns(output_dir, tmp_path):
    path = tmp_path / "bad_perievent.csv"
    _decoding_perievent().drop(columns="F").to_csv(path, index=False)
    result = CliRunner().invoke(mesoscopy.cli, args=f"process decode {path} -o {output_dir}")
    assert result.exit_code == 1
    assert "lacks the F column(s)" in result.output


def test_decode_cmd_empty_input(output_dir, tmp_path):
    path = tmp_path / "empty_perievent.csv"
    pd.DataFrame(columns=list(pm.PERIEVENT_COLUMNS)).to_csv(path, index=False)
    result = CliRunner().invoke(mesoscopy.cli, args=f"process decode {path} -o {output_dir}")
    assert result.exit_code == 1
    assert "has no rows" in result.output


def test_decode_cmd_min_trials_below_folds(output_dir, tmp_path):
    path = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    _decoding_perievent().to_csv(path, index=False)
    result = CliRunner().invoke(mesoscopy.cli, args=f"process decode {path} -o {output_dir} --min-trials 3")
    assert result.exit_code == 1
    assert "min_trials must be at least folds (5), got 3." in result.output


def test_decode_cmd_creates_output_dir(tmp_path):
    path = tmp_path / "ses-01_regions_event-cueonset_perievent.csv"
    _decoding_perievent().to_csv(path, index=False)
    out_dir = tmp_path / "new" / "dir"
    result = CliRunner().invoke(mesoscopy.cli, args=f"process decode {path} -o {out_dir} --repeats 2 --shuffles 0")
    assert result.exit_code == 0, result.output
    assert (out_dir / "ses-01_regions_event-cueonset_label-stim_decoding.csv").exists()
