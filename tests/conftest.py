import pytest

import numpy as np
import h5py as h5

from datetime import datetime
from dateutil.tz import tzlocal
from uuid import uuid4
from pynwb import NWBFile, NWBHDF5IO


@pytest.fixture(scope="session")
def nwbfile(tmp_path_factory):
    """Create an NWBFile object for testing."""
    # Create a temporary file
    tmpfile = tmp_path_factory.mktemp("data") / "test.nwb"
    # Create an NWBFile object
    session_start_time = datetime(2024, 1, 1, 14, 0, 0, tzinfo=tzlocal())

    nwbfile = NWBFile(
        session_description="Test file, not real data",
        identifier=str(uuid4()),
        session_start_time=session_start_time,
        session_id="session_1234",
    )
    # Write the NWBFile object to file
    with NWBHDF5IO(str(tmpfile), "w") as io:
        io.write(nwbfile)
    # Return the path to the temporary file
    return str(tmpfile)


@pytest.fixture(scope="session")
def preproc_h5(tmp_path_factory):
    """Create an HDF5 file with preprocessed data for testing."""
    # Create a temporary file
    tmpfile = tmp_path_factory.mktemp("data") / "preproc.h5"
    # Create an HDF5 file
    with h5.File(str(tmpfile), "w") as f:
        f.create_dataset("data", data=np.random.rand(300, 142, 142))
        f.create_dataset("timestamps", data=np.arange(300))
    # Return the path to the temporary file
    return str(tmpfile)
