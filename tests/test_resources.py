import numpy as np
import pytest

import mesoscopy.resources as res

# ---------------------------------------------------------------------------
# resize_labels
# ---------------------------------------------------------------------------


def test_resize_labels_integer_upscale_repeats_pixels():
    labels = np.arange(6, dtype=np.uint16).reshape(2, 3)
    resized = res.resize_labels(labels, (4, 6))
    np.testing.assert_array_equal(resized, np.repeat(np.repeat(labels, 2, axis=0), 2, axis=1))
    assert resized.dtype == labels.dtype


def test_resize_labels_introduces_no_new_labels():
    left, _ = res.get_atlas()
    for shape in ((280, 284), (210, 213), (100, 101), (140, 142)):
        resized = res.resize_labels(left, shape)
        assert resized.shape == shape
        assert set(np.unique(resized)) <= set(np.unique(left))


def test_resize_labels_rejects_bad_shapes():
    labels = np.zeros((4, 4), dtype=np.uint16)
    with pytest.raises(ValueError, match="positive integers"):
        res.resize_labels(labels, (0, 4))
    with pytest.raises(ValueError, match="positive integers"):
        res.resize_labels(labels, (4, 4, 4))


# ---------------------------------------------------------------------------
# Atlas and landmarks at scale
# ---------------------------------------------------------------------------


def test_get_atlas_native_shape_is_unchanged():
    left, right = res.get_atlas()
    left_explicit, right_explicit = res.get_atlas(res.atlas_shape())
    np.testing.assert_array_equal(left, left_explicit)
    np.testing.assert_array_equal(right, right_explicit)
    np.testing.assert_array_equal(right, np.flip(left, axis=1))


def test_get_atlas_scaled_right_hemisphere_mirrors_left():
    left, right = res.get_atlas((280, 284))
    assert left.shape == right.shape == (280, 284)
    np.testing.assert_array_equal(right, np.flip(left, axis=1))
    # Doubling the atlas duplicates every pixel, so subsampling recovers it exactly.
    np.testing.assert_array_equal(left[::2, ::2], res.get_atlas()[0])


def test_template_shape_and_scale_round_trip():
    height, width = res.atlas_shape()
    assert res.template_shape(1.0) == (height, width)
    assert res.template_shape(2.0) == (2 * height, 2 * width)
    assert res.template_scale(res.template_shape(2.0)) == (2.0, 2.0)
    # Non-integer scales round to whole pixels, and the effective scale follows the rounded shape.
    shape = res.template_shape(1.5)
    assert shape == (round(1.5 * height), round(1.5 * width))
    scale_x, scale_y = res.template_scale(shape)
    assert scale_x == shape[1] / width
    assert scale_y == shape[0] / height


def test_template_shape_rejects_non_positive_scale():
    with pytest.raises(ValueError, match="positive"):
        res.template_shape(0)


def test_scale_landmarks_keeps_the_midline_in_place():
    """The midline of a 142-wide image is at x=70.5, and must map to the midline at every scale."""
    _, width = res.atlas_shape()
    for scale in (2.0, 1.5, 0.5):
        scaled_width = round(width * scale)
        scaled = res.scale_landmarks({"mid": ((width - 1) / 2, 0.0)}, (scaled_width / width, 1.0))
        assert scaled["mid"][0] == pytest.approx((scaled_width - 1) / 2)


@pytest.mark.parametrize("shape", [(280, 284), (210, 213), (350, 355)])
def test_default_landmarks_hit_the_same_region_at_scale(shape):
    """Scaled landmarks and the resampled atlas agree on which region each landmark sits in."""
    left, _ = res.get_atlas()
    left_scaled, _ = res.get_atlas(shape)
    native = res.get_default_landmarks()
    scaled = res.get_default_landmarks(shape)

    assert list(scaled) == list(native)
    for name, (x, y) in native.items():
        scaled_x, scaled_y = scaled[name]
        assert left_scaled[round(scaled_y), round(scaled_x)] == left[round(y), round(x)], name


def test_default_landmarks_paired_across_hemispheres_at_scale():
    landmarks = res.get_default_landmarks((280, 284))
    left, right = res.get_atlas((280, 284))
    for left_name, right_name in (("lFP", "rFP"), ("lPB", "rPB"), ("lpRSP", "rpRSP")):
        left_x, left_y = landmarks[left_name]
        right_x, right_y = landmarks[right_name]
        assert left[round(left_y), round(left_x)] == right[round(right_y), round(right_x)] != 0
