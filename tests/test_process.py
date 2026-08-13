import json
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
from mesoscopy.process import decoder as dec
from mesoscopy.process import regression as regr
from mesoscopy.process import smooth
from mesoscopy.process import zscore
from mesoscopy.process.region import DEFAULT_EXCLUDE
from mesoscopy.process.region import extract_all_regions
from mesoscopy.process.region import extract_region_activity

# Both decoders share an identical interface and result schema, so the behavioural tests below run
# against each of them.
DECODERS = [dec.logistic_decoder, dec.lda_decoder]
DECODER_IDS = ["logistic", "lda"]

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


# Decoder fixtures ----------------------------------------------------------
#
# `decodable_series` carries a label-driven signal in the central 3x3 pixel block only, at an
# amplitude low enough that neither decoder separates the test set perfectly — a degenerate
# confusion matrix would push d' to infinity and make the metric assertions vacuous.

_DECODER_FRAMES = 200
_DECODER_SHAPE = (6, 6)
_SIGNAL_BLOCK = (slice(1, 4), slice(1, 4))


@pytest.fixture
def decoder_labels():
    """Balanced binary labels for the decoder fixtures."""
    return np.tile([0, 1], _DECODER_FRAMES // 2)


@pytest.fixture
def decodable_series(decoder_labels):
    """A (200, 6, 6) series whose central pixel block encodes the label."""
    rng = np.random.default_rng(0)
    series = rng.normal(size=(_DECODER_FRAMES, *_DECODER_SHAPE))
    series[:, *_SIGNAL_BLOCK] += decoder_labels[:, None, None] * 0.6
    return series


@pytest.fixture
def signal_mask():
    """Boolean mask marking the pixels that carry the signal in `decodable_series`."""
    mask = np.zeros(_DECODER_SHAPE, dtype=bool)
    mask[*_SIGNAL_BLOCK] = True
    return mask


@pytest.fixture
def perfectly_decodable_series(decoder_labels):
    """A (200, 6, 6) series separable enough that both decoders classify the test split perfectly."""
    rng = np.random.default_rng(2)
    series = rng.normal(size=(_DECODER_FRAMES, *_DECODER_SHAPE))
    series[:, *_SIGNAL_BLOCK] += decoder_labels[:, None, None] * 5.0
    return series


@pytest.fixture
def undecodable_series():
    """A (200, 6, 6) series of pure noise, carrying no label information."""
    rng = np.random.default_rng(1)
    return rng.normal(size=(_DECODER_FRAMES, *_DECODER_SHAPE))


@pytest.fixture
def decoding_labels_npz(tmp_path_factory):
    """NPZ decoding labels matching the (300, 40, 40) preproc_h5 fixture."""
    tmpfile = tmp_path_factory.mktemp("data") / "labels.npz"
    np.savez(tmpfile, labels=np.tile([0, 1], 150))
    return str(tmpfile)


@pytest.fixture
def decoding_labels_h5(tmp_path_factory):
    """HDF5 decoding labels matching the (300, 40, 40) preproc_h5 fixture."""
    tmpfile = tmp_path_factory.mktemp("data") / "labels.h5"
    with h5.File(str(tmpfile), "w") as f:
        f.create_dataset("labels", data=np.tile([0, 1], 150))
    return str(tmpfile)


@pytest.fixture
def decoding_mask_npz(tmp_path_factory):
    """A 40x40 spatial mask matching the preproc_h5 fixture's spatial dimensions."""
    tmpfile = tmp_path_factory.mktemp("data") / "mask.npz"
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    np.savez(tmpfile, mask=mask)
    return str(tmpfile)


@pytest.fixture
def decoding_mask_wrong_shape_npz(tmp_path_factory):
    """A spatial mask whose shape does not match the preproc_h5 fixture."""
    tmpfile = tmp_path_factory.mktemp("data") / "mask_wrong.npz"
    np.savez(tmpfile, mask=np.ones((20, 20), dtype=bool))
    return str(tmpfile)


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
# decoder._d_prime
# ---------------------------------------------------------------------------


class TestDPrime:
    def test_matches_textbook_formula(self):
        """d' = z(hit) - z(false alarm), which for row-normalised rates is z(1-miss) - z(1-CR)."""
        conf_matrix = np.array([[45, 5], [10, 40]])
        expected = sst.norm.ppf(0.9) - sst.norm.ppf(0.2)
        assert dec._d_prime(conf_matrix) == pytest.approx(expected)

    def test_chance_performance_gives_zero(self):
        assert dec._d_prime(np.array([[25, 25], [25, 25]])) == pytest.approx(0.0)

    def test_perfect_classification_is_finite(self):
        """Without the 1/(2N) correction, rates of 0 and 1 would send d' to infinity."""
        result = dec._d_prime(np.array([[50, 0], [0, 50]]))
        assert np.isfinite(result)
        # Rates are clipped to 1 - 1/(2*50) and 1/(2*50).
        assert result == pytest.approx(sst.norm.ppf(0.99) - sst.norm.ppf(0.01))

    def test_perfectly_inverted_classification_is_finite_and_negative(self):
        result = dec._d_prime(np.array([[0, 50], [50, 0]]))
        assert np.isfinite(result)
        assert result < 0

    def test_correction_scales_with_test_set_size(self):
        """A larger test set supports a larger ceiling on d' for perfect classification."""
        small = dec._d_prime(np.array([[50, 0], [0, 50]]))
        large = dec._d_prime(np.array([[500, 0], [0, 500]]))
        assert large > small

    def test_correction_uses_per_class_counts(self):
        """Each rate is corrected using the count of its own true class, not the pooled total."""
        conf_matrix = np.array([[20, 0], [0, 100]])
        expected = sst.norm.ppf(1 - 1 / 40) - sst.norm.ppf(1 / 200)
        assert dec._d_prime(conf_matrix) == pytest.approx(expected)

    def test_leaves_non_extreme_rates_untouched(self):
        """Clipping must not perturb rates that are already within the corrected bounds."""
        conf_matrix = np.array([[49, 1], [1, 49]])
        expected = sst.norm.ppf(0.98) - sst.norm.ppf(0.02)
        assert dec._d_prime(conf_matrix) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# decoder.logistic_decoder / decoder.lda_decoder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decoder", DECODERS, ids=DECODER_IDS)
class TestDecoders:
    def test_returns_expected_keys(self, decoder, decodable_series, decoder_labels):
        result = decoder(decodable_series, decoder_labels)
        assert set(result) == {
            "accuracy",
            "balanced_accuracy",
            "d_prime",
            "confusion_matrix",
            "test_size",
            "f2_score",
            "coefficients",
        }

    def test_coefficients_have_spatial_shape(self, decoder, decodable_series, decoder_labels):
        result = decoder(decodable_series, decoder_labels)
        assert result["coefficients"].shape == _DECODER_SHAPE

    def test_coefficients_have_non_square_spatial_shape(self, decoder):
        """Coefficients are reshaped to (height, width), not to a square."""
        rng = np.random.default_rng(3)
        labels = np.tile([0, 1], 30)
        series = rng.normal(size=(60, 4, 6))
        series[:, 1, 2] += labels * 2.0

        result = decoder(series, labels)

        assert result["coefficients"].shape == (4, 6)

    def test_accepts_pre_flattened_series(self, decoder, decodable_series, decoder_labels, signal_mask):
        """A masked series arrives as (time, n_pixels); coefficients come back as one per pixel."""
        result = decoder(decodable_series[:, signal_mask], decoder_labels)
        assert result["coefficients"].shape == (signal_mask.sum(),)

    def test_pre_flattened_series_matches_masked_full_frame(
        self, decoder, decodable_series, decoder_labels, signal_mask
    ):
        """Decoding a masked series should equal decoding the full frame restricted to the mask."""
        flattened = decoder(decodable_series[:, signal_mask], decoder_labels)
        masked_full_frame = decoder(decodable_series * signal_mask, decoder_labels)
        assert flattened["accuracy"] == masked_full_frame["accuracy"]

    def test_coefficients_are_largest_over_informative_pixels(
        self, decoder, decodable_series, decoder_labels, signal_mask
    ):
        coefs = np.abs(decoder(decodable_series, decoder_labels)["coefficients"])
        assert coefs[signal_mask].mean() > coefs[~signal_mask].mean()

    def test_confusion_matrix_is_row_normalised(self, decoder, decodable_series, decoder_labels):
        conf_matrix = decoder(decodable_series, decoder_labels)["confusion_matrix"]
        assert conf_matrix.shape == (2, 2)
        np.testing.assert_allclose(conf_matrix.sum(axis=1), 1.0)

    def test_accuracies_are_high_for_decodable_data(self, decoder, decodable_series, decoder_labels):
        result = decoder(decodable_series, decoder_labels)
        assert result["accuracy"] > 0.8
        assert result["balanced_accuracy"] > 0.8

    def test_accuracies_are_near_chance_for_noise(self, decoder, undecodable_series, decoder_labels):
        result = decoder(undecodable_series, decoder_labels)
        assert result["accuracy"] < 0.7
        assert result["balanced_accuracy"] < 0.7

    def test_metrics_are_bounded(self, decoder, decodable_series, decoder_labels):
        result = decoder(decodable_series, decoder_labels)
        for metric in ("accuracy", "balanced_accuracy", "f2_score"):
            assert 0.0 <= result[metric] <= 1.0, f"{metric} out of range: {result[metric]}"

    def test_d_prime_is_higher_for_decodable_data(
        self, decoder, decodable_series, undecodable_series, decoder_labels
    ):
        decodable = decoder(decodable_series, decoder_labels)["d_prime"]
        undecodable = decoder(undecodable_series, decoder_labels)["d_prime"]
        assert np.isfinite(decodable)
        assert decodable > undecodable

    def test_d_prime_is_finite_for_a_perfect_split(self, decoder, perfectly_decodable_series, decoder_labels):
        """A perfectly separable test split must not produce an infinite d'."""
        result = decoder(perfectly_decodable_series, decoder_labels)

        np.testing.assert_allclose(result["confusion_matrix"], np.eye(2))
        assert np.isfinite(result["d_prime"])
        assert result["d_prime"] > 0

    def test_d_prime_is_a_python_float(self, decoder, decodable_series, decoder_labels):
        """Keeps the metric JSON-serialisable without relying on the encoder's scalar handling."""
        assert type(decoder(decodable_series, decoder_labels)["d_prime"]) is float

    def test_echoes_test_size(self, decoder, decodable_series, decoder_labels):
        result = decoder(decodable_series, decoder_labels, test_size=0.3)
        assert result["test_size"] == 0.3

    def test_test_size_is_passed_to_the_split(self, decoder, decodable_series, decoder_labels, mocker):
        spy = mocker.spy(dec, "train_test_split")

        decoder(decodable_series, decoder_labels, test_size=0.4)

        assert spy.call_args.kwargs["test_size"] == 0.4

    def test_default_test_size_is_a_fifth(self, decoder, decodable_series, decoder_labels, mocker):
        spy = mocker.spy(dec, "train_test_split")

        result = decoder(decodable_series, decoder_labels)

        assert spy.call_args.kwargs["test_size"] == 0.2
        assert result["test_size"] == 0.2

    def test_is_deterministic(self, decoder, decodable_series, decoder_labels):
        """The fixed random_state should make repeated runs identical."""
        first = decoder(decodable_series, decoder_labels)
        second = decoder(decodable_series, decoder_labels)

        assert first["accuracy"] == second["accuracy"]
        np.testing.assert_allclose(first["coefficients"], second["coefficients"])
        np.testing.assert_allclose(first["confusion_matrix"], second["confusion_matrix"])

    def test_does_not_mutate_input(self, decoder, decodable_series, decoder_labels):
        series_before = decodable_series.copy()
        labels_before = decoder_labels.copy()

        decoder(decodable_series, decoder_labels)

        np.testing.assert_array_equal(decodable_series, series_before)
        np.testing.assert_array_equal(decoder_labels, labels_before)


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


@pytest.mark.parametrize(
    ("command", "outfile"),
    [("smooth", "preproc_smoothed.h5"), ("zscore", "preproc_zscored.h5")],
)
def test_cmd_creates_missing_output_dir(preproc_h5, output_dir, command, outfile):
    out_dir = pathlib.Path(output_dir) / "nested" / command
    runner = CliRunner()
    result = runner.invoke(mesoscopy.cli, args=f"process {command} {preproc_h5} -o {out_dir}")
    assert result.exit_code == 0
    assert "Creating output directory" in result.output
    assert (out_dir / outfile).is_file()


def test_regions_cmd_creates_missing_output_dir(
    preproc_h5_bytes_timestamps, output_dir, mock_left_aba, mock_right_aba, mock_annotations
):
    out_dir = pathlib.Path(output_dir) / "nested" / "regions"
    runner = CliRunner()
    with (
        patch("mesoscopy.resources.get_atlas", return_value=(mock_left_aba, mock_right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=mock_annotations),
    ):
        result = runner.invoke(mesoscopy.cli, args=f"process regions {preproc_h5_bytes_timestamps} -o {out_dir}")
    assert result.exit_code == 0
    assert "Creating output directory" in result.output
    assert (out_dir / "preproc_bytes_regions.csv").is_file()


def test_regression_cmd_creates_missing_output_dir(preproc_h5, regressor_npz, output_dir):
    out_dir = pathlib.Path(output_dir) / "nested" / "regression"
    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli,
        args=f"process regression {preproc_h5} {regressor_npz} -o {out_dir} --fast",
    )
    assert result.exit_code == 0
    assert "Creating output directory" in result.output
    assert (out_dir / "preproc_regression.npz").is_file()


def test_decode_cmd_logistic(preproc_h5, decoding_labels_npz, output_dir):
    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli,
        args=f"process decode {preproc_h5} {decoding_labels_npz} -o {output_dir}",
    )
    assert result.exit_code == 0

    outpath = pathlib.Path(output_dir) / "preproc_decoding_logistic.json"
    assert outpath.is_file()

    with outpath.open() as f:
        results = json.load(f)
    assert np.array(results["coefficients"]).shape == (40, 40)
    assert results["test_size"] == 0.2


def test_decode_cmd_lda(preproc_h5, decoding_labels_h5, output_dir):
    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli,
        args=f"process decode {preproc_h5} {decoding_labels_h5} -o {output_dir} -f lda",
    )
    assert result.exit_code == 0
    assert (pathlib.Path(output_dir) / "preproc_decoding_lda.json").is_file()


def test_decode_cmd_with_mask(preproc_h5, decoding_labels_npz, decoding_mask_npz, output_dir):
    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli,
        args=f"process decode {preproc_h5} {decoding_labels_npz} -o {output_dir} -m {decoding_mask_npz}",
    )
    assert result.exit_code == 0
    assert "Loading spatial mask" in result.output

    outpath = pathlib.Path(output_dir) / "preproc_decoding_logistic.json"
    assert outpath.is_file()

    with outpath.open() as f:
        results = json.load(f)

    # Coefficients are scattered back onto the full frame, zero outside the mask.
    coefs = np.array(results["coefficients"])
    mask = np.load(decoding_mask_npz)["mask"]
    assert coefs.shape == (40, 40)
    assert np.all(coefs[~mask] == 0)
    assert np.any(coefs[mask] != 0)


def test_decode_cmd_mask_shape_mismatch_raises(
    preproc_h5, decoding_labels_npz, decoding_mask_wrong_shape_npz, output_dir
):
    out_dir = pathlib.Path(output_dir) / "nested" / "decode"
    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli,
        args=f"process decode {preproc_h5} {decoding_labels_npz} -o {out_dir} -m {decoding_mask_wrong_shape_npz}",
    )
    assert result.exit_code != 0
    assert "Creating output directory" in result.output
    assert isinstance(result.exception, ValueError)
    assert "does not match" in str(result.exception)


def test_decode_cmd_rejects_unknown_decoder(preproc_h5, decoding_labels_npz, output_dir):
    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli,
        args=f"process decode {preproc_h5} {decoding_labels_npz} -o {output_dir} -f svm",
    )
    assert result.exit_code != 0
    assert "svm" in result.output


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


# ---------------------------------------------------------------------------
# region.extract_all_regions (as_dataframe)
# ---------------------------------------------------------------------------


def test_extract_all_regions_as_dataframe_returns_dataframe(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        result = extract_all_regions(region_deltaf_series, as_dataframe=True)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["region", "time_idx", "F"]


def test_extract_all_regions_as_dataframe_is_long_format(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    """One row per (region, frame): REG1 and REG2 across both hemispheres, FRP1 excluded by default."""
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        result = extract_all_regions(region_deltaf_series, as_dataframe=True)

    assert set(result["region"]) == {"L_REG1", "R_REG1", "L_REG2", "R_REG2"}
    assert len(result) == 4 * _N_FRAMES
    assert sorted(result.loc[result["region"] == "L_REG1", "time_idx"]) == list(range(_N_FRAMES))


def test_extract_all_regions_as_dataframe_matches_dict_values(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        as_dict = extract_all_regions(region_deltaf_series)
        as_df = extract_all_regions(region_deltaf_series, as_dataframe=True)

    for region, activity in as_dict.items():
        rows = as_df[as_df["region"] == region].sort_values("time_idx")
        np.testing.assert_allclose(rows["F"].to_numpy(), activity)


def test_extract_all_regions_as_dataframe_respects_exclude(
    region_left_aba, region_right_aba, region_annotations, region_deltaf_series
):
    with patch("mesoscopy.resources.get_atlas", return_value=(region_left_aba, region_right_aba)), \
         patch("mesoscopy.resources.get_atlas_annotations", return_value=region_annotations):
        result = extract_all_regions(
            region_deltaf_series, exclude=["REG1"], ignore_default_exclude=True, as_dataframe=True
        )

    assert set(result["region"]) == {"L_REG2", "R_REG2", "L_FRP1", "R_FRP1"}
