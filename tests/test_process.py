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
from mesoscopy.process.region import DEFAULT_EXCLUDE
from mesoscopy.process.region import extract_all_regions
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


def test_extract_region_activity_hemisphere_argument_case_insensitive(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        lower = extract_region_activity(region_deltaf_series, "REG1", "left")
        upper = extract_region_activity(region_deltaf_series, "REG1", "LEFT")
    np.testing.assert_allclose(lower, upper)


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
