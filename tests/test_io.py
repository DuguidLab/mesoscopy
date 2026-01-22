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


def test_write_points_csv(tmp_path):
    path = tmp_path / "test_points"
    data = {"testArea": [0, 200], "anotherTestArea": [250, 20]}
    io.write_points(str(path), data)
