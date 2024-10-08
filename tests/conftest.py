import pytest

import numpy as np
import h5py as h5

from datetime import datetime, timedelta
from dateutil.tz import tzlocal
from uuid import uuid4
from pynwb import NWBFile, NWBHDF5IO
from pynwb.ophys import OpticalChannel, OnePhotonSeries


@pytest.fixture
def nwbfile(tmp_path_factory):
    """Create an NWBFile object for testing."""
    # Create a temporary file
    tmpfile = tmp_path_factory.mktemp("data") / "test.nwb"
    # Create an NWBFile object
    session_start_time = datetime(2024, 1, 1, 14, 0, 0, tzinfo=tzlocal())

    nwbfile = NWBFile(
        session_description="Test file, not real data",
        identifier="session_1234",
        session_start_time=session_start_time,
        # session_id="session_1234",
    )

    # Add fake raw imaging data
    device = nwbfile.create_device(
        name="Mesoscope",
        description="Single-photon widefield imaging scope.",
        manufacturer="INSS UK",
    )

    optical_channel = OpticalChannel(
        name="DualAcquisitionChannel",
        description="Acquisition channel for GCaMP6s excited at 470 and 405 nm.",
        emission_lambda=500.0,
    )

    imaging_plane = nwbfile.create_imaging_plane(
        name="DualChannelImagingPlane",
        optical_channel=optical_channel,
        # imaging_rate=50.,
        description="Random data for testing.",
        device=device,
        excitation_lambda=470.0,
        indicator="GCaMP6s",
        location="dorsal cortex",
        grid_spacing=[20.0, 20.0],
        grid_spacing_unit="micrometers",
    )

    frames_num = 300
    timestamps = [
        (timedelta(milliseconds=i)).total_seconds() for i in range(frames_num)
    ]

    imaging_series = OnePhotonSeries(
        name="DualChannelImagingSeries",
        imaging_plane=imaging_plane,
        data=np.random.rand(frames_num, 142, 142),
        timestamps=timestamps,
        unit="pixel_intensity",
        binning=2,
        power=10.0,
        exposure_time=0.01,
        pmt_gain=10.0,
        description="Random data for testing.",
        comments="Completely random, no std or mean differentiation between test channels.",
    )

    nwbfile.add_acquisition(imaging_series)

    # Write the NWBFile object to file
    with NWBHDF5IO(str(tmpfile), "w") as io:
        io.write(nwbfile)
    # Return the path to the temporary file
    return str(tmpfile)


@pytest.fixture(scope="session")
def raw_h5(tmp_path_factory):
    """Create an HDF5 file with fake raw data for testing."""
    # Create a temporary file
    tmpfile = tmp_path_factory.mktemp("data") / "preproc_test.h5"
    # Create timestamps
    timestamps = [
        (datetime.now() + timedelta(milliseconds=i))
        .isoformat()
        .replace("-", "")
        .replace(":", "")
        .replace(".", "")
        for i in range(300)
    ]
    print(timestamps)
    # Create an HDF5 file
    with h5.File(str(tmpfile), "w") as f:
        f.create_dataset("frames", data=np.random.rand(300, 142, 142))
        f.create_dataset("timestamps", data=timestamps)
    # Return the path to the temporary file
    return str(tmpfile)


@pytest.fixture(scope="session")
def preproc_h5(tmp_path_factory):
    """Create an HDF5 file with fake preprocessed data for testing."""
    # Create a temporary file
    tmpfile = tmp_path_factory.mktemp("data") / "preproc.h5"
    # Create an HDF5 file
    with h5.File(str(tmpfile), "w") as f:
        f.create_dataset("data", data=np.random.rand(300, 142, 142))
        f.create_dataset("timestamps", data=np.arange(300))
    # Return the path to the temporary file
    return str(tmpfile)
