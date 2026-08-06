import pathlib
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
