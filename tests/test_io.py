import json
from datetime import datetime
from importlib import resources
from uuid import uuid4

import dask.array as da
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


def test_write_json(tmp_path):
    path = tmp_path / "test.json"
    outpath = io.write_json(str(path), {"a": 1, "b": "two", "c": [3.0, 4.0]})

    assert outpath == str(path)
    with path.open() as f:
        assert json.load(f) == {"a": 1, "b": "two", "c": [3.0, 4.0]}


def test_write_json_serialises_numpy_arrays(tmp_path):
    path = tmp_path / "test.json"
    io.write_json(str(path), {"matrix": np.arange(6).reshape(2, 3)})

    with path.open() as f:
        assert json.load(f)["matrix"] == [[0, 1, 2], [3, 4, 5]]


def test_write_json_serialises_numpy_scalars(tmp_path):
    path = tmp_path / "test.json"
    io.write_json(str(path), {"score": np.float64(0.5), "count": np.int64(3), "flag": np.bool_(True)})

    with path.open() as f:
        results = json.load(f)
    assert results == {"score": 0.5, "count": 3, "flag": True}


def test_write_json_round_trips_an_array(tmp_path):
    path = tmp_path / "test.json"
    array = np.random.default_rng(0).random((4, 5))
    io.write_json(str(path), {"array": array})

    with path.open() as f:
        np.testing.assert_allclose(np.array(json.load(f)["array"]), array)


def test_write_json_rejects_unsupported_types(tmp_path):
    """Non-NumPy objects should still raise rather than be silently coerced."""
    path = tmp_path / "test.json"
    with pytest.raises(TypeError):
        io.write_json(str(path), {"when": datetime(2024, 1, 1, tzinfo=tzlocal())})
