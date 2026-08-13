from importlib import resources

import numpy as np
import pytest
from click.testing import CliRunner
from skimage import transform as trf

import mesoscopy
import mesoscopy.register as reg
import mesoscopy.register.landmarks_gui as reg_gui
import mesoscopy.register.qa as reg_qa
import mesoscopy.resources as res
from mesoscopy import io
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

def test_landmarks_affine_defaults_to_atlas_shape(small_series, landmark_pair):
    """Registered frames are in template space, so they default to the atlas shape.

    This is the shape mesoscopy.process.region requires; the shape of the recording is irrelevant.
    """
    recording_lm, template_lm = landmark_pair
    warped, _ = landmarks_affine(small_series, recording_lm, template_lm)

    atlas_shape = res.get_atlas()[0].shape
    assert atlas_shape != small_series.shape[1:]  # guard: the fixture must not already be atlas-shaped
    assert warped.shape == (small_series.shape[0], *atlas_shape)


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
    reference = np.array(
        [trf.warp(small_series[i], ref_tform, order=3, output_shape=(40, 40)) for i in range(small_series.shape[0])]
    )

    warped, _ = landmarks_affine(small_series, recording_lm, template_lm, output_shape=(40, 40))

    np.testing.assert_array_equal(warped, reference)


def test_landmarks_affine_explicit_output_shape(landmark_pair):
    """An explicit output_shape overrides the atlas default."""
    rng = np.random.default_rng(1)
    series = rng.random((5, 40, 40), dtype=np.float32)
    recording_lm, template_lm = landmark_pair

    warped, _ = landmarks_affine(series, recording_lm, template_lm, output_shape=(25, 30))

    assert warped.shape == (5, 25, 30)


def test_landmarks_affine_output_shape_does_not_crop_the_source(landmark_pair):
    """Sizing the output must not discard source pixels.

    The previous implementation cropped the *input* to (crop_y, crop_x), which threw away exactly
    the source pixels the warp needs whenever the brain was not in the top-left corner.
    """
    series = np.zeros((1, 200, 200), dtype=np.float32)
    series[0, 120:160, 130:170] = 1.0  # signal well outside the top-left 40x40 corner
    recording_lm, template_lm = landmark_pair

    # Landmarks describing a pure translation that brings the blob into a 60x60 output window.
    recording = {"a": (130.0, 120.0), "b": (170.0, 120.0), "c": (130.0, 160.0)}
    template = {"a": (10.0, 10.0), "b": (50.0, 10.0), "c": (10.0, 50.0)}

    warped, _ = landmarks_affine(series, recording, template, output_shape=(60, 60))

    assert warped.shape == (1, 60, 60)
    assert warped[0].sum() > 0.9 * series[0].sum()


def test_landmarks_affine_identity_landmarks(small_series):
    """When recording and template landmarks are identical the warp is a near-identity."""
    landmarks = {"A": (5.0, 5.0), "B": (35.0, 5.0), "C": (5.0, 35.0)}
    warped, _ = landmarks_affine(small_series, landmarks, landmarks, output_shape=(40, 40))

    # Interior pixels should be reproduced almost exactly (boundary pixels may differ
    # due to the constant-zero padding convention of skimage warp).
    interior = small_series[:, 5:-5, 5:-5]
    warped_interior = warped[:, 5:-5, 5:-5]
    np.testing.assert_allclose(warped_interior, interior, atol=1e-5)


def test_landmarks_affine_registers_a_known_transform_to_the_atlas():
    """Ground truth: warp the atlas by a known affine, then register it back.

    Unlike the reference tests above, the expected result here is not derived from the
    implementation - the recording is synthesised from the atlas itself.
    """
    atlas = res.get_atlas()[0].astype(np.float32)
    template_lm = res.get_default_landmarks()

    # Synthesise a "recording": the atlas seen through a known rotation, scale and shift.
    true_tform = trf.AffineTransform(scale=(1.8, 1.7), rotation=np.deg2rad(8), translation=(40, 25))
    recording = trf.warp(atlas, true_tform.inverse, output_shape=(300, 320), order=1)
    recording_lm = {
        name: tuple(true_tform(np.array([point]))[0]) for name, point in template_lm.items()
    }

    warped, tform = landmarks_affine(recording[None], recording_lm, template_lm)

    # Output is in atlas space...
    assert warped.shape == (1, *atlas.shape)

    # ...the recovered transform is the one we applied...
    np.testing.assert_allclose(tform.params, true_tform.params, atol=1e-3)

    # ...and every landmark lands within a pixel of its template position.
    landed = tform.inverse(np.array(list(recording_lm.values())))
    expected = np.array(list(template_lm.values()))
    assert np.linalg.norm(landed - expected, axis=1).max() < 1.0

    # The registered image reproduces the atlas it was synthesised from.
    labelled = atlas > 0
    assert np.abs(warped[0][labelled] - atlas[labelled]).mean() < 0.02 * atlas.max()


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


# ---------------------------------------------------------------------------
# Landmark coordinate convention
#
# Landmarks are stored as (x, y) == (column, row) everywhere on disk, matching
# skimage.transform. Napari is the only (row, column) surface, and mark_landmarks
# is responsible for transposing on the way in and out.
# ---------------------------------------------------------------------------


def test_template_landmarks_are_xy_not_rowcol():
    """Paired left/right template landmarks must hit the same atlas region in each hemisphere.

    This only holds when the shipped landmark file is read as (x, y); reading it as
    (row, column) puts every pair in the wrong place.
    """
    landmarks = res.get_default_landmarks()
    left_aba, right_aba = res.get_atlas()
    midline = left_aba.shape[1] / 2

    for left_name, right_name in (("lFP", "rFP"), ("lPB", "rPB"), ("lpRSP", "rpRSP")):
        left_x, left_y = landmarks[left_name]
        right_x, right_y = landmarks[right_name]

        # Lateral landmarks sit on their own side of the midline...
        assert left_x < midline
        assert right_x > midline

        # ...and pick out the same labelled region in the matching hemisphere map.
        left_region = left_aba[int(left_y), int(left_x)]
        right_region = right_aba[int(right_y), int(right_x)]
        assert left_region != 0
        assert left_region == right_region

    # Midline landmarks sit on the midline.
    for name in ("bregma", "cFP", "aIPB"):
        assert landmarks[name][0] == pytest.approx(midline, abs=1.0)


def test_points_roundtrip_preserves_xy(tmp_path):
    """write_points -> read_points must not transpose the coordinates."""
    points = {"bregma": (71.0, 60.0), "lFP": (51.0, 19.0)}
    path = str(tmp_path / "points.csv")

    io.write_points(path, points)

    assert io.read_points(path) == points


def test_mark_landmarks_transposes_between_napari_and_storage(monkeypatch):
    """Seed points enter napari as (row, col); marked points come back out as (x, y)."""
    captured = {}

    class FakePointsLayer:
        def __init__(self, data):
            self.data = data
            self.mode = None
            self.face_color_mode = None

    class FakeWindow:
        def add_dock_widget(self, widget):
            pass

    class FakeViewer:
        def __init__(self):
            self.window = FakeWindow()

        def add_image(self, *args, **kwargs):
            pass

        def add_points(self, data, **kwargs):
            captured["seed"] = np.asarray(data, dtype=float)
            # Simulate the user dragging the first point by +3 rows / +5 columns.
            marked = np.asarray(data, dtype=float).copy()
            marked[0] += (3.0, 5.0)
            return FakePointsLayer(marked)

    monkeypatch.setattr(reg_gui.napari, "view_image", lambda *args, **kwargs: FakeViewer())
    monkeypatch.setattr(reg_gui.napari, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(reg_gui, "_create_label_menu", lambda points_layer, labels: None)

    template = {"a": (10.0, 20.0), "b": (30.0, 40.0)}
    marked = reg_gui.mark_landmarks(np.zeros((100, 120)), None, template)

    # (x, y) seeds are handed to napari as (row, col).
    np.testing.assert_array_equal(captured["seed"], [[20.0, 10.0], [40.0, 30.0]])

    # Marked points come back as (x, y), so the +3 row / +5 col drag reads as +5 x / +3 y.
    assert marked["a"] == (15.0, 23.0)
    assert marked["b"] == (30.0, 40.0)


def test_qa_plot_landmarks_uses_xy():
    """The QA scatter must plot x on the x-axis, not the row index."""
    source = np.array([[10.0, 20.0], [30.0, 40.0]])
    target = np.array([[11.0, 21.0], [31.0, 41.0]])

    fig = reg_qa.plot_landmarks(source, target)

    np.testing.assert_array_equal(fig.data[0].x, source[:, 0])
    np.testing.assert_array_equal(fig.data[0].y, source[:, 1])
    np.testing.assert_array_equal(fig.data[1].x, target[:, 0])
    np.testing.assert_array_equal(fig.data[1].y, target[:, 1])


def test_qa_plot_frame_without_landmarks():
    """plot_frame must be callable without landmarks."""
    fig = reg_qa.plot_frame(np.random.default_rng(0).random((10, 12)))
    assert len(fig.data) == 1


def test_qa_plot_frame_landmarks_use_xy():
    landmarks = np.array([[10.0, 20.0], [30.0, 40.0]])
    fig = reg_qa.plot_frame(np.random.default_rng(0).random((50, 50)), landmarks=landmarks)

    np.testing.assert_array_equal(fig.data[1].x, landmarks[:, 0])
    np.testing.assert_array_equal(fig.data[1].y, landmarks[:, 1])
