from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mesoscopy.process.region import DEFAULT_EXCLUDE, extract_all_regions, extract_region_activity

# ---------------------------------------------------------------------------
# Fixtures
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
def mock_left_aba():
    aba = np.zeros((_ATLAS_H, _ATLAS_W), dtype=np.uint16)
    aba[0:2, 0:3] = 1
    aba[2:4, 0:3] = 2
    aba[4:6, 0:3] = 3
    return aba


@pytest.fixture(scope="module")
def mock_right_aba(mock_left_aba):
    return np.flip(mock_left_aba, axis=1).copy()


@pytest.fixture(scope="module")
def mock_annotations():
    return pd.DataFrame({"id": [1, 2, 3], "acronym": ["REG1", "REG2", "FRP1"]})


@pytest.fixture(scope="module")
def deltaf_series():
    rng = np.random.default_rng(0)
    return rng.random((_N_FRAMES, _ATLAS_H, _ATLAS_W))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_resources(left_aba, right_aba, annotations):
    """Return a context manager that patches both resource loaders."""
    return (
        patch("mesoscopy.resources.get_atlas", return_value=(left_aba, right_aba)),
        patch("mesoscopy.resources.get_atlas_annotations", return_value=annotations),
    )


# ---------------------------------------------------------------------------
# extract_region_activity
# ---------------------------------------------------------------------------


class TestExtractRegionActivity:
    def test_left_hemisphere_shape(self, mock_left_aba, mock_right_aba, mock_annotations, deltaf_series):
        with patch("mesoscopy.resources.get_atlas", return_value=(mock_left_aba, mock_right_aba)), \
             patch("mesoscopy.resources.get_atlas_annotations", return_value=mock_annotations):
            result = extract_region_activity(deltaf_series, "REG1", "left")
        assert result.shape == (_N_FRAMES,)

    def test_right_hemisphere_shape(self, mock_left_aba, mock_right_aba, mock_annotations, deltaf_series):
        with patch("mesoscopy.resources.get_atlas", return_value=(mock_left_aba, mock_right_aba)), \
             patch("mesoscopy.resources.get_atlas_annotations", return_value=mock_annotations):
            result = extract_region_activity(deltaf_series, "REG1", "right")
        assert result.shape == (_N_FRAMES,)

    def test_both_hemispheres_shape(self, mock_left_aba, mock_right_aba, mock_annotations, deltaf_series):
        with patch("mesoscopy.resources.get_atlas", return_value=(mock_left_aba, mock_right_aba)), \
             patch("mesoscopy.resources.get_atlas_annotations", return_value=mock_annotations):
            result = extract_region_activity(deltaf_series, "REG1", "both")
        assert result.shape == (_N_FRAMES,)

    def test_left_hemisphere_values(self, mock_left_aba, mock_right_aba, mock_annotations, deltaf_series):
        # Region 1 occupies rows 0-1, cols 0-2 in the left atlas.
        expected = deltaf_series[:, 0:2, 0:3].mean(axis=(1, 2))
        with patch("mesoscopy.resources.get_atlas", return_value=(mock_left_aba, mock_right_aba)), \
             patch("mesoscopy.resources.get_atlas_annotations", return_value=mock_annotations):
            result = extract_region_activity(deltaf_series, "REG1", "left")
        np.testing.assert_allclose(result, expected)

    def test_right_hemisphere_values(self, mock_left_aba, mock_right_aba, mock_annotations, deltaf_series):
        # right_aba = flip(left_aba): region 1 lands in rows 0-1, cols 3-5.
        expected = deltaf_series[:, 0:2, 3:6].mean(axis=(1, 2))
        with patch("mesoscopy.resources.get_atlas", return_value=(mock_left_aba, mock_right_aba)), \
             patch("mesoscopy.resources.get_atlas_annotations", return_value=mock_annotations):
            result = extract_region_activity(deltaf_series, "REG1", "right")
        np.testing.assert_allclose(result, expected)

    def test_both_hemispheres_values(self, mock_left_aba, mock_right_aba, mock_annotations, deltaf_series):
        # left + right: region 1 covers rows 0-1 across all 6 columns.
        expected = deltaf_series[:, 0:2, :].mean(axis=(1, 2))
        with patch("mesoscopy.resources.get_atlas", return_value=(mock_left_aba, mock_right_aba)), \
             patch("mesoscopy.resources.get_atlas_annotations", return_value=mock_annotations):
            result = extract_region_activity(deltaf_series, "REG1", "both")
        np.testing.assert_allclose(result, expected)

    def test_different_regions_produce_different_results(
        self, mock_left_aba, mock_right_aba, mock_annotations, deltaf_series
    ):
        with patch("mesoscopy.resources.get_atlas", return_value=(mock_left_aba, mock_right_aba)), \
             patch("mesoscopy.resources.get_atlas_annotations", return_value=mock_annotations):
            reg1 = extract_region_activity(deltaf_series, "REG1", "left")
            reg2 = extract_region_activity(deltaf_series, "REG2", "left")
        assert not np.allclose(reg1, reg2), "Different regions should produce different activity signals"

    def test_invalid_hemisphere_raises(self, mock_left_aba, mock_right_aba, mock_annotations, deltaf_series):
        with patch("mesoscopy.resources.get_atlas", return_value=(mock_left_aba, mock_right_aba)), \
             patch("mesoscopy.resources.get_atlas_annotations", return_value=mock_annotations):
            with pytest.raises(ValueError, match="hemisphere"):
                extract_region_activity(deltaf_series, "REG1", "bilateral")

    def test_hemisphere_argument_case_insensitive(
        self, mock_left_aba, mock_right_aba, mock_annotations, deltaf_series
    ):
        with patch("mesoscopy.resources.get_atlas", return_value=(mock_left_aba, mock_right_aba)), \
             patch("mesoscopy.resources.get_atlas_annotations", return_value=mock_annotations):
            lower = extract_region_activity(deltaf_series, "REG1", "left")
            upper = extract_region_activity(deltaf_series, "REG1", "LEFT")
        np.testing.assert_allclose(lower, upper)


# ---------------------------------------------------------------------------
# extract_all_regions
# ---------------------------------------------------------------------------


class TestExtractAllRegions:
    def test_returns_dict(self, mock_left_aba, mock_right_aba, mock_annotations, deltaf_series):
        with patch("mesoscopy.resources.get_atlas", return_value=(mock_left_aba, mock_right_aba)), \
             patch("mesoscopy.resources.get_atlas_annotations", return_value=mock_annotations):
            result = extract_all_regions(deltaf_series)
        assert isinstance(result, dict)

    def test_keys_have_hemisphere_prefix(self, mock_left_aba, mock_right_aba, mock_annotations, deltaf_series):
        with patch("mesoscopy.resources.get_atlas", return_value=(mock_left_aba, mock_right_aba)), \
             patch("mesoscopy.resources.get_atlas_annotations", return_value=mock_annotations):
            result = extract_all_regions(deltaf_series)
        for key in result:
            assert key.startswith("L_") or key.startswith("R_"), f"Unexpected key format: {key}"

    def test_both_hemispheres_present_for_each_region(
        self, mock_left_aba, mock_right_aba, mock_annotations, deltaf_series
    ):
        with patch("mesoscopy.resources.get_atlas", return_value=(mock_left_aba, mock_right_aba)), \
             patch("mesoscopy.resources.get_atlas_annotations", return_value=mock_annotations):
            result = extract_all_regions(deltaf_series)
        for region in ("REG1", "REG2"):
            assert f"L_{region}" in result
            assert f"R_{region}" in result

    def test_values_have_correct_shape(self, mock_left_aba, mock_right_aba, mock_annotations, deltaf_series):
        with patch("mesoscopy.resources.get_atlas", return_value=(mock_left_aba, mock_right_aba)), \
             patch("mesoscopy.resources.get_atlas_annotations", return_value=mock_annotations):
            result = extract_all_regions(deltaf_series)
        for key, value in result.items():
            assert value.shape == (_N_FRAMES,), f"Wrong shape for {key}: {value.shape}"

    def test_default_exclude_filters_regions(self, mock_left_aba, mock_right_aba, mock_annotations, deltaf_series):
        # FRP1 is in DEFAULT_EXCLUDE; it should be absent from the result.
        with patch("mesoscopy.resources.get_atlas", return_value=(mock_left_aba, mock_right_aba)), \
             patch("mesoscopy.resources.get_atlas_annotations", return_value=mock_annotations):
            result = extract_all_regions(deltaf_series)
        assert "L_FRP1" not in result
        assert "R_FRP1" not in result

    def test_ignore_default_exclude_includes_all_regions(
        self, mock_left_aba, mock_right_aba, mock_annotations, deltaf_series
    ):
        with patch("mesoscopy.resources.get_atlas", return_value=(mock_left_aba, mock_right_aba)), \
             patch("mesoscopy.resources.get_atlas_annotations", return_value=mock_annotations):
            result = extract_all_regions(deltaf_series, ignore_default_exclude=True)
        assert "L_FRP1" in result
        assert "R_FRP1" in result

    def test_custom_exclude_removes_region(self, mock_left_aba, mock_right_aba, mock_annotations, deltaf_series):
        with patch("mesoscopy.resources.get_atlas", return_value=(mock_left_aba, mock_right_aba)), \
             patch("mesoscopy.resources.get_atlas_annotations", return_value=mock_annotations):
            result = extract_all_regions(deltaf_series, exclude=["REG1"])
        assert "L_REG1" not in result
        assert "R_REG1" not in result
        assert "L_REG2" in result
        assert "R_REG2" in result

    def test_custom_exclude_combined_with_default(
        self, mock_left_aba, mock_right_aba, mock_annotations, deltaf_series
    ):
        with patch("mesoscopy.resources.get_atlas", return_value=(mock_left_aba, mock_right_aba)), \
             patch("mesoscopy.resources.get_atlas_annotations", return_value=mock_annotations):
            result = extract_all_regions(deltaf_series, exclude=["REG1"])
        assert "L_FRP1" not in result
        assert "L_REG1" not in result
        assert "L_REG2" in result

    def test_does_not_mutate_default_exclude(self, mock_left_aba, mock_right_aba, mock_annotations, deltaf_series):
        original = list(DEFAULT_EXCLUDE)
        with patch("mesoscopy.resources.get_atlas", return_value=(mock_left_aba, mock_right_aba)), \
             patch("mesoscopy.resources.get_atlas_annotations", return_value=mock_annotations):
            extract_all_regions(deltaf_series, exclude=["REG1"])
        assert DEFAULT_EXCLUDE == original, "extract_all_regions must not mutate DEFAULT_EXCLUDE"

    def test_values_match_extract_region_activity(
        self, mock_left_aba, mock_right_aba, mock_annotations, deltaf_series
    ):
        """extract_all_regions results should match extract_region_activity for each region."""
        with patch("mesoscopy.resources.get_atlas", return_value=(mock_left_aba, mock_right_aba)), \
             patch("mesoscopy.resources.get_atlas_annotations", return_value=mock_annotations):
            all_regions = extract_all_regions(deltaf_series, ignore_default_exclude=True)
            single_left = extract_region_activity(deltaf_series, "REG2", "left")
            single_right = extract_region_activity(deltaf_series, "REG2", "right")
        np.testing.assert_allclose(all_regions["L_REG2"], single_left)
        np.testing.assert_allclose(all_regions["R_REG2"], single_right)
