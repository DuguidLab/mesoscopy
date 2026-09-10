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


def test_zscore_cmd_h5(preproc_h5, output_dir):
    runner = CliRunner()
    result = runner.invoke(mesoscopy.cli, args=f"process zscore {preproc_h5} -o {output_dir}")
    assert result.exit_code == 0

    outpath = pathlib.Path(output_dir) / "preproc_zscored.h5"
    assert outpath.is_file()
    assert io.read_h5(str(outpath))["/F"].shape == (300, 40, 40)


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
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        result = extract_region_activity(region_deltaf_series, "REG1", "left")
    assert result.shape == (_N_FRAMES,)


def test_extract_region_activity_right_hemisphere_shape(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        result = extract_region_activity(region_deltaf_series, "REG1", "right")
    assert result.shape == (_N_FRAMES,)


def test_extract_region_activity_both_hemispheres_shape(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        result = extract_region_activity(region_deltaf_series, "REG1", "both")
    assert result.shape == (_N_FRAMES,)


def test_extract_region_activity_left_hemisphere_values(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    # Region 1 occupies rows 0-1, cols 0-2 in the left atlas.
    expected = region_deltaf_series[:, 0:2, 0:3].mean(axis=(1, 2))
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        result = extract_region_activity(region_deltaf_series, "REG1", "left")
    np.testing.assert_allclose(result, expected)


def test_extract_region_activity_right_hemisphere_values(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    # right_aba = flip(left_aba): region 1 lands in rows 0-1, cols 3-5.
    expected = region_deltaf_series[:, 0:2, 3:6].mean(axis=(1, 2))
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        result = extract_region_activity(region_deltaf_series, "REG1", "right")
    np.testing.assert_allclose(result, expected)


def test_extract_region_activity_both_hemispheres_values(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    # left + right: region 1 covers rows 0-1 across all 6 columns.
    expected = region_deltaf_series[:, 0:2, :].mean(axis=(1, 2))
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        result = extract_region_activity(region_deltaf_series, "REG1", "both")
    np.testing.assert_allclose(result, expected)


def test_extract_region_activity_different_regions_produce_different_results(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        reg1 = extract_region_activity(region_deltaf_series, "REG1", "left")
        reg2 = extract_region_activity(region_deltaf_series, "REG2", "left")
    assert not np.allclose(reg1, reg2), "Different regions should produce different activity signals"


def test_extract_region_activity_invalid_hemisphere_raises(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        with pytest.raises(ValueError, match="hemisphere"):
            extract_region_activity(region_deltaf_series, "REG1", "bilateral")


def test_extract_region_activity_unknown_acronym_raises(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    """An unrecognised acronym is a ValueError naming it, not an IndexError from the id lookup."""
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        with pytest.raises(ValueError, match="REG3 not recognised"):
            extract_region_activity(region_deltaf_series, "REG3", "left")


def test_extract_region_activity_unknown_acronym_suggests_close_matches(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        with pytest.raises(ValueError, match="Closest matches") as excinfo:
            extract_region_activity(region_deltaf_series, "REG3", "left")
    assert "REG1" in str(excinfo.value)


def test_extract_region_activity_close_match_suggestion_ignores_case(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    """Lookup stays case-sensitive, but the suggestion ignores case."""
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        with pytest.raises(ValueError, match="Closest matches") as excinfo:
            extract_region_activity(region_deltaf_series, "reg1", "left")
    assert "REG1" in str(excinfo.value)


def test_extract_region_activity_unknown_acronym_without_close_match(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    """Nothing similar in the atlas means no misleading suggestion."""
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        with pytest.raises(ValueError, match="not recognised") as excinfo:
            extract_region_activity(region_deltaf_series, "zzzz", "left")
    assert "Closest matches" not in str(excinfo.value)


def test_extract_region_activity_hemisphere_argument_case_insensitive(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        lower = extract_region_activity(region_deltaf_series, "REG1", "left")
        upper = extract_region_activity(region_deltaf_series, "REG1", "LEFT")
    np.testing.assert_allclose(lower, upper)


def test_extract_region_activity_returns_plain_ndarray(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        result = extract_region_activity(region_deltaf_series, "REG1", "left")
    assert not isinstance(result, np.ma.MaskedArray)


def test_extract_region_activity_ignores_nan_pixels(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    series = region_deltaf_series.copy()
    series[:, 0, 0] = np.nan
    expected = np.nanmean(series[:, 0:2, 0:3].reshape(_N_FRAMES, -1), axis=1)

    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
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

    with warnings.catch_warnings(), \
         patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        warnings.simplefilter("ignore", RuntimeWarning)
        result = extract_region_activity(series, "REG1", "left")

    assert np.isnan(result[0])
    assert not np.isnan(result[1:]).any()


def test_extract_region_activity_empty_region_is_nan(
    region_left_aba, region_right_aba, region_annotations_with_empty, region_deltaf_series
):
    """A region with no pixels in the atlas gives NaN rather than raising."""
    with warnings.catch_warnings(), \
         patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations_with_empty):
        warnings.simplefilter("ignore", RuntimeWarning)
        result = extract_region_activity(region_deltaf_series, "EMPTY", "left")

    assert result.shape == (_N_FRAMES,)
    assert np.isnan(result).all()


# ---------------------------------------------------------------------------
# region.extract_all_regions
# ---------------------------------------------------------------------------


def test_extract_all_regions_returns_dict(region_left_aba, region_right_aba, region_annotations, region_deltaf_series):
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        result = extract_all_regions(region_deltaf_series)
    assert isinstance(result, dict)


def test_extract_all_regions_keys_have_hemisphere_prefix(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        result = extract_all_regions(region_deltaf_series)
    for key in result:
        assert key.startswith("L_") or key.startswith("R_"), f"Unexpected key format: {key}"


def test_extract_all_regions_both_hemispheres_present_for_each_region(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        result = extract_all_regions(region_deltaf_series)
    for region in ("REG1", "REG2"):
        assert f"L_{region}" in result
        assert f"R_{region}" in result


def test_extract_all_regions_values_have_correct_shape(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        result = extract_all_regions(region_deltaf_series)
    for key, value in result.items():
        assert value.shape == (_N_FRAMES,), f"Wrong shape for {key}: {value.shape}"


def test_extract_all_regions_default_exclude_filters_regions(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    # FRP1 is in DEFAULT_EXCLUDE; it should be absent from the result.
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        result = extract_all_regions(region_deltaf_series)
    assert "L_FRP1" not in result
    assert "R_FRP1" not in result


def test_extract_all_regions_ignore_default_exclude_includes_all_regions(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        result = extract_all_regions(region_deltaf_series, ignore_default_exclude=True)
    assert "L_FRP1" in result
    assert "R_FRP1" in result


def test_extract_all_regions_custom_exclude_removes_region(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        result = extract_all_regions(region_deltaf_series, exclude=["REG1"])
    assert "L_REG1" not in result
    assert "R_REG1" not in result
    assert "L_REG2" in result
    assert "R_REG2" in result


def test_extract_all_regions_custom_exclude_combined_with_default(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        result = extract_all_regions(region_deltaf_series, exclude=["REG1"])
    assert "L_FRP1" not in result
    assert "L_REG1" not in result
    assert "L_REG2" in result


def test_extract_all_regions_does_not_mutate_default_exclude(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    original = list(DEFAULT_EXCLUDE)
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        extract_all_regions(region_deltaf_series, exclude=["REG1"])
    assert original == DEFAULT_EXCLUDE, "extract_all_regions must not mutate DEFAULT_EXCLUDE"


def test_extract_all_regions_values_match_extract_region_activity(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    """extract_all_regions results should match extract_region_activity for each region."""
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
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
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        result = extract_all_regions(region_deltaf_series, ignore_default_exclude=True)
    np.testing.assert_allclose(result["L_REG1"], expected)


def test_extract_all_regions_ignores_nan_pixels(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    series = region_deltaf_series.copy()
    series[:, 0, 0] = np.nan
    expected = np.nanmean(series[:, 0:2, 0:3].reshape(_N_FRAMES, -1), axis=1)

    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        result = extract_all_regions(series, ignore_default_exclude=True)

    assert not np.isnan(result["L_REG1"]).any()
    np.testing.assert_allclose(result["L_REG1"], expected)


def test_extract_all_regions_nan_does_not_leak_into_other_regions(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    """Pixel (0, 0) belongs to left REG1 only, so no other region's trace may change."""
    series = region_deltaf_series.copy()
    series[:, 0, 0] = np.nan

    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        clean = extract_all_regions(region_deltaf_series, ignore_default_exclude=True)
        result = extract_all_regions(series, ignore_default_exclude=True)

    for key in ("L_REG2", "L_FRP1", "R_REG1", "R_REG2", "R_FRP1"):
        np.testing.assert_allclose(result[key], clean[key], err_msg=f"{key} was affected by a NaN outside it")


def test_extract_all_regions_all_nan_region_frame_is_nan(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    series = region_deltaf_series.copy()
    series[0, 0:2, 0:3] = np.nan

    with warnings.catch_warnings(), \
         patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        warnings.simplefilter("ignore", RuntimeWarning)
        result = extract_all_regions(series, ignore_default_exclude=True)

    assert np.isnan(result["L_REG1"][0])
    assert not np.isnan(result["L_REG1"][1:]).any()
    assert not np.isnan(result["L_REG2"]).any()


def test_extract_all_regions_empty_region_is_nan(
    region_left_aba, region_right_aba, region_annotations_with_empty, region_deltaf_series
):
    """A region with no pixels in the atlas gives NaN rather than nulling its neighbours."""
    with warnings.catch_warnings(), \
         patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations_with_empty):
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

    with warnings.catch_warnings(), \
         patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
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

    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        result = extract_all_regions(series, ignore_default_exclude=True)

    np.testing.assert_allclose(result["L_REG1"], extract_mask_activity(series, region_left_aba == 1))


def test_extract_all_regions_handles_nans_across_frame_blocks(
    region_left_aba, region_right_aba, region_annotations
):
    """Frames are processed in blocks, so NaN counts must stay per-frame across a block boundary."""
    rng = np.random.default_rng(1)
    series = rng.random((_LONG_N_FRAMES, _ATLAS_H, _ATLAS_W))
    series[::3, 0, 0] = np.nan
    expected = np.nanmean(series[:, 0:2, 0:3].reshape(_LONG_N_FRAMES, -1), axis=1)

    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
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
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
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
