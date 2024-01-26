import pytest

import numpy as np
import h5py as h5
from pynwb import NWBFile, NWBHDF5IO


@pytest.fixture(scope="session")
def nwbfile(tmpdir_factory):
    """Create an NWBFile object for testing."""
    # Create a temporary file
    tmpfile = tmpdir_factory.mktemp("data").join("test.nwb")
    # Create an NWBFile object
    nwbfile = NWBFile("session_description", "identifier", "session_start_time")
    # Add some data
    nwbfile.add_acquisition(np.arange(100.0), name="acquisition")
    # Write the NWBFile object to file
    with NWBHDF5IO(str(tmpfile), "w") as io:
        io.write(nwbfile)
    # Return the path to the temporary file
    return str(tmpfile)


@pytest.fixture(scope="session")
def preproc_h5(tmpdir_factory):
    """Create an HDF5 file with preprocessed data for testing."""
    # Create a temporary file
    tmpfile = tmpdir_factory.mktemp("data").join("preproc.h5")
    # Create an HDF5 file
    with h5.File(str(tmpfile), "w") as f:
        # Create a group
        grp = f.create_group("preproc")
        # Add some data
        grp.create_dataset("data", data=np.arange(100.0))
    # Return the path to the temporary file
    return str(tmpfile)
