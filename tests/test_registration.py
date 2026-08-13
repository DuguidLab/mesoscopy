import pathlib
import shutil
from importlib import resources
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from click.testing import CliRunner
from skimage import transform as trf

import mesoscopy
import mesoscopy.preprocess as preproc
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
    recording_lm = {name: tuple(true_tform(np.array([point]))[0]) for name, point in template_lm.items()}

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
    tform_params = trf.AffineTransform(scale=(1.2, 1.1), rotation=0.05, translation=(3, -4)).params

    nwb = reg.update_nwb(nwbfile, preproc_h5, tform_params)

    assert nwb.processing["ophys"]["CCFRegisteredSeries"].original.data
    assert nwb.processing["ophys"]["CCFRegisteredSeries"].xy_translation.data.any()


def test_update_nwb_stores_one_transform_matrix_per_timestamp(nwbfile, preproc_h5):
    """xy_translation must be (n_timestamps, 3, 3), not the flattened (3 * n_timestamps, 3).

    pynwb only warns when data and timestamps disagree in length, so a mis-shaped array is written
    without complaint.
    """
    tform_params = trf.AffineTransform(scale=(1.2, 1.1), rotation=0.05, translation=(3, -4)).params

    reg.update_nwb(nwbfile, preproc_h5, tform_params)

    # Read the file back rather than trusting the in-memory object.
    with h5py.File(preproc_h5, "r") as f:
        n_timestamps = len(f["/timestamps"])

    written = io.read_nwb(nwbfile, mode="r")
    stack = written.processing["ophys"]["CCFRegisteredSeries"]
    xy_translation = np.asarray(stack.xy_translation.data)

    assert xy_translation.shape == (n_timestamps, 3, 3)
    assert len(xy_translation) == len(stack.xy_translation.timestamps)
    # Every frame carries the same global transform.
    for frame_params in xy_translation:
        np.testing.assert_allclose(frame_params, tform_params)


def test_register_landmarks_cli_nwb_stores_the_transform_matrix(preproc_nwb, output_dir):
    """End-to-end NWB path: the stored transform must be the matrix, correctly shaped per frame."""
    points = str(resources.files(res).joinpath("ccf_template_landmarks_140x142.csv"))
    runner = CliRunner()

    result = runner.invoke(
        mesoscopy.cli,
        args=f"register landmarks {preproc_nwb} -r {points} -o {output_dir}",
    )

    assert result.exit_code == 0, result.output

    written = io.read_nwb(preproc_nwb, mode="r")
    stack = written.processing["ophys"]["CCFRegisteredSeries"]
    xy_translation = np.asarray(stack.xy_translation.data)

    assert xy_translation.shape == (len(stack.xy_translation.timestamps), 3, 3)
    # Recording and template landmarks are the same file here, so the transform is the identity.
    np.testing.assert_allclose(xy_translation[0], np.eye(3), atol=1e-6)


# ---------------------------------------------------------------------------
# Landmark file discovery and maxip sourcing
# ---------------------------------------------------------------------------


def test_session_id_drops_extension_and_preprocessed_suffix():
    assert reg.session_id_from_path("/data/sub-01_preprocessed.h5") == "sub-01"
    assert reg.session_id_from_path("/data/sub-01.nwb") == "sub-01"
    # A directory containing ".h5" must not be mangled - the old str.replace did exactly that.
    assert reg.session_id_from_path("/data.h5archive/sub-01_preprocessed.h5") == "sub-01"


def test_landmarks_inference_finds_the_file_label_actually_writes(preproc_h5, output_dir, tmp_path):
    """`label` names its output after the session ID, so it drops the "_preprocessed" suffix.

    The previous inference looked for "<recording>_landmarks.csv", which for a preprocessed HDF5
    could never match what `label` had written.
    """
    recording = tmp_path / "sub-01_preprocessed.h5"
    shutil.copy(preproc_h5, recording)
    # Exactly what `register label` would have written for this recording.
    io.write_points(str(tmp_path / "sub-01_landmarks.csv"), res.get_default_landmarks())

    runner = CliRunner()
    result = runner.invoke(mesoscopy.cli, args=f"register landmarks {recording} -o {output_dir}")

    assert result.exit_code == 0, result.output
    assert "sub-01_landmarks.csv" in result.output


def test_landmarks_inference_error_lists_where_it_looked(preproc_h5, output_dir):
    runner = CliRunner()
    result = runner.invoke(mesoscopy.cli, args=f"register landmarks {preproc_h5} -o {output_dir}")

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "Searched" in str(result.exception)
    assert "_landmarks.csv" in str(result.exception)


def test_load_maxips_returns_none_when_absent(preproc_h5):
    """preproc_h5 holds only /F and /timestamps, so there is nothing to load."""
    assert reg.load_maxips(preproc_h5) == (None, None)


def test_linked_preprocessed_path_resolves_the_external_link(nwbfile, preproc_h5):
    """Preprocessing links the NWB dF/F series to the HDF5 file holding the QA projections."""
    preproc.update_nwb(nwbfile, preproc_h5)

    resolved = reg.linked_preprocessed_path(nwbfile)

    assert resolved is not None
    assert pathlib.Path(resolved).resolve() == pathlib.Path(preproc_h5).resolve()


def test_linked_preprocessed_path_is_none_without_an_external_link(preproc_nwb):
    """A dF/F series stored inline rather than linked resolves to nothing."""
    assert reg.linked_preprocessed_path(preproc_nwb) is None


def test_label_falls_back_to_a_deltaf_projection(monkeypatch, tmp_path):
    """Without QA projections the maxip must come from the dF/F series, not the raw frames.

    The raw frames are neither cropped nor binned, so projecting them yields landmarks in a
    different pixel space to the data being registered.
    """
    recording = tmp_path / "sub-01_preprocessed.h5"
    rng = np.random.default_rng(0)
    deltaf = rng.random((10, 40, 40), dtype=np.float32)
    with h5py.File(recording, "w") as f:
        f.create_dataset("/F", data=deltaf)
        f.create_dataset("/timestamps", data=np.arange(10.0))

    captured = {}
    _patch_napari(monkeypatch, captured=captured)

    runner = CliRunner()
    result = runner.invoke(mesoscopy.cli, args=f"register label {recording} -o {tmp_path}")

    assert result.exit_code == 0, result.output
    assert "∆F/F" in result.output
    # The projection is of the dF/F series, in the dF/F pixel space.
    np.testing.assert_allclose(captured["image"], deltaf.max(axis=0))


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


class _FakePointsLayer:
    """Stands in for a napari Points layer: (row, col) data plus an aligned 'label' property."""

    def __init__(self, data, labels):
        self.data = np.asarray(data, dtype=float)
        self.properties = {"label": np.asarray(labels, dtype=str)}
        self.mode = None
        self.face_color_mode = None


def _patch_napari(monkeypatch, final_state=None, captured=None):
    """Replace the napari viewer with a fake that records seeds and returns a scripted end state.

    `final_state` is the (data, labels) pair the points layer holds when the user closes the viewer,
    i.e. the result of whatever marking, dragging and deleting they did. Defaults to the seed points
    untouched.
    """

    class FakeViewer:
        def __init__(self):
            self.window = SimpleNamespace(add_dock_widget=lambda widget: None)

        def add_image(self, *args, **kwargs):
            pass

        def add_points(self, data, **kwargs):
            if captured is not None:
                captured["seed"] = np.asarray(data, dtype=float)
                captured["labels"] = list(kwargs["properties"]["label"])
            if final_state is None:
                return _FakePointsLayer(data, kwargs["properties"]["label"])
            return _FakePointsLayer(*final_state)

    def view_image(image, *args, **kwargs):
        if captured is not None:
            captured["image"] = np.asarray(image)
        return FakeViewer()

    monkeypatch.setattr(reg_gui.napari, "view_image", view_image)
    monkeypatch.setattr(reg_gui.napari, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(reg_gui, "_create_label_menu", lambda points_layer, labels: None)


def test_mark_landmarks_transposes_between_napari_and_storage(monkeypatch):
    """Seed points enter napari as (row, col); marked points come back out as (x, y)."""
    captured = {}
    # The user drags the first point by +3 rows / +5 columns.
    _patch_napari(monkeypatch, final_state=([[23.0, 15.0], [40.0, 30.0]], ["a", "b"]), captured=captured)

    template = {"a": (10.0, 20.0), "b": (30.0, 40.0)}
    marked = reg_gui.mark_landmarks(np.zeros((100, 120)), None, template)

    # (x, y) seeds are handed to napari as (row, col).
    np.testing.assert_array_equal(captured["seed"], [[20.0, 10.0], [40.0, 30.0]])

    # Marked points come back as (x, y), so the +3 row / +5 col drag reads as +5 x / +3 y.
    assert marked["a"] == (15.0, 23.0)
    assert marked["b"] == (30.0, 40.0)


def test_mark_landmarks_identifies_points_by_label_not_position(monkeypatch):
    """Deleting and re-marking a point appends it to the layer; labels must still hold.

    Zipping names against point order - as the previous implementation did - silently attaches every
    name to the wrong coordinates as soon as the user re-marks anything.
    """
    template = {"bregma": (71.0, 60.0), "cFP": (71.0, 19.0), "lPB": (30.0, 35.0)}
    # bregma was deleted and re-marked, so it sits last in the layer and the others shifted up.
    _patch_napari(
        monkeypatch,
        final_state=([[19.0, 71.0], [35.0, 30.0], [63.0, 71.0]], ["cFP", "lPB", "bregma"]),
    )

    marked = reg_gui.mark_landmarks(np.zeros((140, 142)), None, template)

    assert marked == {"bregma": (71.0, 63.0), "cFP": (71.0, 19.0), "lPB": (30.0, 35.0)}
    # Returned in template order, which is the order the transform pairs the point sets in.
    assert list(marked) == list(template)


def test_mark_landmarks_reports_missing_duplicate_and_unknown_points(monkeypatch, capsys):
    """Edited point sets are reported rather than silently mismapped."""
    template = {"bregma": (71.0, 60.0), "cFP": (71.0, 19.0), "lPB": (30.0, 35.0)}
    _patch_napari(
        monkeypatch,
        final_state=(
            [[19.0, 71.0], [10.0, 10.0], [40.0, 40.0], [35.0, 30.0]],
            ["cFP", "bregma", "bregma", "notALandmark"],
        ),
    )

    marked = reg_gui.mark_landmarks(np.zeros((140, 142)), None, template)

    assert marked["cFP"] == (71.0, 19.0)
    assert marked["bregma"] == (40.0, 40.0)  # the most recently marked of the two
    assert "lPB" not in marked  # never marked

    output = capsys.readouterr().out
    assert "lPB" in output
    assert "bregma" in output
    assert "notALandmark" in output


def test_mark_landmarks_scales_seed_points_to_the_recording(monkeypatch):
    """Template seeds are scaled to the recording so they don't bunch up in a corner."""
    captured = {}
    _patch_napari(monkeypatch, captured=captured)

    # Recording is twice the size of the template in both axes.
    reg_gui.mark_landmarks(np.zeros((280, 284)), None, {"bregma": (71.0, 60.0)}, template_shape=(140, 142))

    np.testing.assert_allclose(captured["seed"], [[120.0, 142.0]])  # (row, col) of (x=142, y=120)


def test_mark_landmarks_does_not_scale_seeds_without_a_template_shape(monkeypatch):
    """Without a known template shape the seeds are used as-is."""
    captured = {}
    _patch_napari(monkeypatch, captured=captured)

    reg_gui.mark_landmarks(np.zeros((280, 284)), None, {"bregma": (71.0, 60.0)})

    np.testing.assert_allclose(captured["seed"], [[60.0, 71.0]])


def test_register_label_cli_writes_landmarks(monkeypatch, tmp_path):
    """The label command round-trips GUI points to a landmarks CSV in the (x, y) convention."""
    preproc = tmp_path / "sub-test_preprocessed.h5"
    with h5py.File(preproc, "w") as f:
        f.create_dataset("/qa/gcamp_maxip_projection", data=np.random.default_rng(0).random((140, 142)))
        f.create_dataset("/qa/isosb_maxip_projection", data=np.random.default_rng(1).random((140, 142)))

    _patch_napari(monkeypatch)  # the user closes the viewer without moving anything

    runner = CliRunner()
    result = runner.invoke(mesoscopy.cli, args=f"register label {preproc} -o {tmp_path}")

    assert result.exit_code == 0

    # Recording is atlas-sized, so unmoved seeds must round-trip to the template landmarks exactly.
    written = io.read_points(str(tmp_path / "sub-test_landmarks.csv"))
    assert written == res.get_default_landmarks()


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
