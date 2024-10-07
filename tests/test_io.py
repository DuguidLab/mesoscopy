import pytest

import numpy as np

from datetime import datetime, timedelta
from dateutil.tz import tzlocal
from pynwb import NWBFile, NWBHDF5IO
from uuid import uuid4

import mesoscopy.io as io


def test_read_h5(raw_h5):
    _ = io.read_h5(raw_h5)


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
    path = tmp_path / "test.h5"
    data = np.random.rand(300, 142, 142)
    io.store_interim(data, str(path))
