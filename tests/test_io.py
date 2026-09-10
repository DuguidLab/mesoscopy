from datetime import datetime
from importlib import resources
from uuid import uuid4

import dask.array as da
import imageio.v2 as iio
import numpy as np
import pytest
from dateutil.tz import tzlocal
from pynwb import NWBHDF5IO
from pynwb import NWBFile

import mesoscopy.resources
from mesoscopy import io


def test_read_h5(raw_h5):
    _ = io.read_h5(raw_h5)


def test_h5_write(tmp_path):
    path = tmp_path / "test.h5"
    data = {"dataset1": np.array([1, 2, 3]), "dataset2": np.array([[1, 2], [3, 4]])}
    io.write_h5(path, data)
    assert io.read_h5(path)["dataset1"][0] == 1
    assert io.read_h5(path)["dataset2"][0][0] == 1


def test_read_nwb(nwbfile):
    _ = io.read_nwb(nwbfile)


def test_read_nwb_return_io(nwbfile):
    _, _ = io.read_nwb(nwbfile, return_io=True)


def test_write_nwb(tmp_path):
    path = tmp_path / "test.nwb"
    session_start_time = datetime(2024, 1, 1, 14, 0, 0, tzinfo=tzlocal())

    nwbfile = NWBFile(
        session_description="Test file, not real data",
        identifier=str(uuid4()),
        session_start_time=session_start_time,
        session_id="session_1234",
    )
    io.write_nwb(path, nwbfile, mode="w")
    _ = io.read_nwb(path)


def test_write_nwb_io(tmp_path):
    path = tmp_path / "test.nwb"
    session_start_time = datetime(2024, 1, 1, 14, 0, 0, tzinfo=tzlocal())

    nwbfile = NWBFile(
        session_description="Test file, not real data",
        identifier=str(uuid4()),
        session_start_time=session_start_time,
        session_id="session_1234",
    )
    io.write_nwb(path, nwbfile, mode="w", io=NWBHDF5IO(path, mode="w"))
    _ = io.read_nwb(path)


def test_store_interim(tmp_path):
    path = tmp_path / "test.zarr"
    data = np.random.rand(300, 142, 142)
    io.store_interim(data, str(path))


def test_store_interim_dask(tmp_path):
    path = tmp_path / "test.zarr"
    dask_data = da.from_array(np.random.rand(300, 142, 142), chunks=(100, 142, 142))
    io.store_interim(dask_data, str(path))


def test_load_interim(tmp_path):
    path = tmp_path / "test.h5"
    data = np.random.rand(300, 142, 142)
    assert io.store_interim(data, str(path)).any()


def test_read_points_fiji():
    assert io.read_points(str(resources.files(mesoscopy.resources).joinpath("ccf_template_top_140x142.points")))


def test_read_points_csv():
    assert io.read_points(str(resources.files(mesoscopy.resources).joinpath("ccf_template_landmarks_140x142.csv")))


def test_read_points_unsupported():
    with pytest.raises(ValueError):
        io.read_points("unsupported.file")


def test_load_preprocessed_h5(preproc_h5):
    session_id, deltaf, ts = io.load_deltaf(preproc_h5)
    assert session_id == "preproc"
    assert deltaf.shape == (300, 40, 40)
    assert ts.shape == (300,)


def test_load_preprocessed_nwb(preproc_nwb):
    session_id, deltaf, ts = io.load_deltaf(preproc_nwb, nwb=True)
    assert session_id == "session_1234"
    assert deltaf.shape == (300, 40, 40)
    assert ts.shape == (300,)


def test_write_points_csv(tmp_path):
    path = tmp_path / "test_points"
    data = {"testArea": [0, 200], "anotherTestArea": [250, 20]}
    io.write_points(str(path), data)


# ---------------------------------------------------------------------------
# io.read_mask
# ---------------------------------------------------------------------------


@pytest.fixture
def labelled_mask():
    """A 6x6 label image: rows 0-1 are label 1, rows 2-3 are label 2, rows 4-5 are 0."""
    mask = np.zeros((6, 6), dtype=np.uint16)
    mask[0:2] = 1
    mask[2:4] = 2
    return mask


def test_read_mask_npy_boolean(tmp_path):
    path = tmp_path / "barrel_cortex.npy"
    mask = np.zeros((6, 6), dtype=bool)
    mask[0:2, 0:3] = True
    np.save(path, mask)

    masks = io.read_mask(str(path))

    assert list(masks) == ["barrel_cortex"]
    assert masks["barrel_cortex"].dtype == bool
    np.testing.assert_array_equal(masks["barrel_cortex"], mask)


def test_read_mask_npy_single_label_uses_file_stem(tmp_path):
    path = tmp_path / "roi.npy"
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[0:2, 0:3] = 1
    np.save(path, mask)

    masks = io.read_mask(str(path))

    assert list(masks) == ["roi"]
    np.testing.assert_array_equal(masks["roi"], mask.astype(bool))


def test_read_mask_npy_labelled_splits_into_one_mask_per_label(tmp_path, labelled_mask):
    path = tmp_path / "roi.npy"
    np.save(path, labelled_mask)

    masks = io.read_mask(str(path))

    assert list(masks) == ["roi_1", "roi_2"]
    np.testing.assert_array_equal(masks["roi_1"], labelled_mask == 1)
    np.testing.assert_array_equal(masks["roi_2"], labelled_mask == 2)
    assert all(mask.dtype == bool for mask in masks.values())


def test_read_mask_npy_labels_need_not_be_sequential(tmp_path):
    path = tmp_path / "roi.npy"
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[0:2] = 2
    mask[2:4] = 7
    np.save(path, mask)

    masks = io.read_mask(str(path))

    assert list(masks) == ["roi_2", "roi_7"]


def test_read_mask_npz_uses_keys_as_names(tmp_path):
    path = tmp_path / "rois.npz"
    s1 = np.zeros((6, 6), dtype=bool)
    s1[0:2] = True
    m2 = np.zeros((6, 6), dtype=bool)
    m2[4:6] = True
    np.savez(path, S1=s1, M2=m2)

    masks = io.read_mask(str(path))

    assert set(masks) == {"S1", "M2"}
    np.testing.assert_array_equal(masks["S1"], s1)
    np.testing.assert_array_equal(masks["M2"], m2)


def test_read_mask_npz_labelled_entry_is_split(tmp_path, labelled_mask):
    path = tmp_path / "rois.npz"
    np.savez(path, cortex=labelled_mask)

    masks = io.read_mask(str(path))

    assert set(masks) == {"cortex_1", "cortex_2"}


def test_read_mask_tiff(tmp_path):
    path = tmp_path / "roi.tiff"
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[0:2, 0:3] = 1
    iio.imwrite(path, mask)

    masks = io.read_mask(str(path))

    assert list(masks) == ["roi"]
    np.testing.assert_array_equal(masks["roi"], mask.astype(bool))


def test_read_mask_tif_suffix(tmp_path, labelled_mask):
    path = tmp_path / "roi.tif"
    iio.imwrite(path, labelled_mask)

    masks = io.read_mask(str(path))

    assert list(masks) == ["roi_1", "roi_2"]


def test_read_mask_binary_float_mask_is_accepted(tmp_path):
    path = tmp_path / "roi.npy"
    mask = np.zeros((6, 6), dtype=np.float64)
    mask[0:2, 0:3] = 1.0
    np.save(path, mask)

    masks = io.read_mask(str(path))

    np.testing.assert_array_equal(masks["roi"], mask.astype(bool))


def test_read_mask_non_binary_float_mask_raises(tmp_path):
    path = tmp_path / "roi.npy"
    mask = np.zeros((6, 6), dtype=np.float64)
    mask[0:2, 0:3] = 0.5
    np.save(path, mask)

    with pytest.raises(ValueError, match="values other than 0 and 1"):
        io.read_mask(str(path))


def test_read_mask_non_2d_raises(tmp_path):
    path = tmp_path / "roi.npy"
    np.save(path, np.ones((3, 6, 6), dtype=bool))

    with pytest.raises(ValueError, match="must be a 2D array"):
        io.read_mask(str(path))


def test_read_mask_unsupported_format_raises(tmp_path):
    path = tmp_path / "roi.csv"
    path.write_text("not,a,mask\n")

    with pytest.raises(ValueError, match="Unsupported file format"):
        io.read_mask(str(path))


def test_read_mask_duplicate_names_raise(tmp_path, labelled_mask):
    # "roi" splits into roi_1/roi_2, which collides with the explicit "roi_1" key.
    path = tmp_path / "rois.npz"
    np.savez(path, roi=labelled_mask, roi_1=np.ones((6, 6), dtype=bool))

    with pytest.raises(ValueError, match="occurs more than once"):
        io.read_mask(str(path))


def test_read_mask_empty_mask_is_loaded(tmp_path):
    # An all-zero mask loads cleanly; it is the extraction step that rejects it.
    path = tmp_path / "roi.npy"
    np.save(path, np.zeros((6, 6), dtype=np.uint8))

    masks = io.read_mask(str(path))

    assert list(masks) == ["roi"]
    assert not masks["roi"].any()
