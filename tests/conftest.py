import importlib
import importlib.resources
import json
from datetime import datetime
from datetime import timedelta

import h5py as h5
import numpy as np
import pytest
import yaml
from dateutil.tz import tzlocal
from pynwb import NWBHDF5IO
from pynwb import NWBFile
from pynwb.ophys import ImageSeries
from pynwb.ophys import OnePhotonSeries
from pynwb.ophys import OpticalChannel

import tests.resources


@pytest.fixture(scope="session")
def random_idx():
    """Return a random index for testing."""
    return np.sort(np.random.choice(600, 300, replace=False))


@pytest.fixture
def nwbfile(tmp_path_factory, random_idx):
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

    # Generate mock dual-channel imaging data
    mock_gcamp = np.random.normal(70, 200, size=(300, 40, 40))
    mock_isosb = np.random.normal(65, 50, size=(300, 40, 40))

    # Merge the two channels in random order
    mock_dual_channel = np.insert(mock_gcamp, random_idx - np.arange(len(random_idx)), mock_isosb, axis=0)

    frames_num = 600
    timestamps = [(timedelta(milliseconds=i)).total_seconds() for i in range(frames_num)]

    imaging_series = OnePhotonSeries(
        name="DualChannelImagingSeries",
        imaging_plane=imaging_plane,
        data=mock_dual_channel,
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


@pytest.fixture
def nwbfile_noacquisition(tmp_path_factory):
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

    # Write the NWBFile object to file
    with NWBHDF5IO(str(tmpfile), "w") as io:
        io.write(nwbfile)
    # Return the path to the temporary file
    return str(tmpfile)


@pytest.fixture(scope="session")
def raw_h5(tmp_path_factory, random_idx):
    """Create an HDF5 file with fake raw data for testing."""
    # Create a temporary file
    tmpfile = tmp_path_factory.mktemp("data") / "preproc_test.h5"
    # Create timestamps
    timestamps = [
        (datetime.now() + timedelta(milliseconds=i)).isoformat().replace("-", "").replace(":", "").replace(".", "")
        for i in range(600)
    ]
    # Generate mock dual-channel imaging data
    mock_gcamp = np.random.normal(70, 200, size=(300, 40, 40))
    mock_isosb = np.random.normal(65, 50, size=(300, 40, 40))

    # Merge the two channels in random order
    mock_dual_channel = np.insert(mock_gcamp, random_idx - np.arange(len(random_idx)), mock_isosb, axis=0)

    # Create an HDF5 file
    with h5.File(str(tmpfile), "w") as f:
        f.create_dataset("frames", data=mock_dual_channel)
        f.create_dataset("timestamps", data=timestamps, dtype="S26")
        f.create_dataset("test_isosb_idx", data=random_idx)

    # Return the path to the temporary file
    return str(tmpfile)


@pytest.fixture
def raw_avi():
    with importlib.resources.path(tests.resources, "example_recording.avi") as path:
        return str(path)


@pytest.fixture
def raw_timestamps():
    with importlib.resources.path(tests.resources, "example_recording_ts.csv") as path:
        return str(path)


@pytest.fixture
def preproc_h5(tmp_path_factory):
    """Create an HDF5 file with fake preprocessed data for testing."""
    # Create a temporary file
    tmpfile = tmp_path_factory.mktemp("data") / "preproc.h5"
    frames_num = 300
    timestamps = [(timedelta(milliseconds=i)).total_seconds() for i in range(frames_num)]
    # Create an HDF5 file
    with h5.File(str(tmpfile), "w") as f:
        f.create_dataset("/F", data=np.random.rand(frames_num, 40, 40))
        f.create_dataset("/timestamps", data=timestamps)
    # Return the path to the temporary file
    return str(tmpfile)


@pytest.fixture
def preproc_nwb(nwbfile, preproc_h5):
    f = h5.File(preproc_h5, "r")
    io = NWBHDF5IO(nwbfile, "a")
    nwb = io.read()
    deltaF_series = ImageSeries(
        name="DeltaFSeries",
        data=f["/F"][:],
        timestamps=f["/timestamps"][:],
        unit="df/f",
        description="dF/F widefield cortical imaging series.",
        comments="This imaging series is corrected for the haemodynamic response.",
    )

    ophys_module = nwb.create_processing_module(name="ophys", description="optical physiology processed data")

    ophys_module.add(deltaF_series)

    io.write(nwb)

    return nwbfile


@pytest.fixture
def output_dir(tmp_path_factory):
    """Create a temporary directory for output."""
    return str(tmp_path_factory.mktemp("output"))


@pytest.fixture
def meta_yaml(tmp_path_factory, partial=False):
    tmpfile = tmp_path_factory.mktemp("data") / "test_meta.yml"

    metadata = {
        "subject_id": "testyaml",
        "sex": "Male",
        "genotype": "Wt",
        "species": "Mus musculus",
        "strain": "C57/B6",
        "dob": "2025-01-11",
        "session_description": "This is but a test.",
        "experimenter": "John Doe",
        "lab": "Doe lab",
        "institution": "University of Someplace with lots of funding",
    }

    if partial:
        metadata.pop("lab")
        metadata.pop("genotype")

    with open(tmpfile, "w") as fp:
        yaml.safe_dump(metadata, fp)

    return str(tmpfile)


@pytest.fixture
def meta_json(tmp_path_factory, partial=False):
    tmpfile = tmp_path_factory.mktemp("data") / "test_meta.json"

    metadata = {
        "subject_id": "testjson",
        "sex": "Male",
        "genotype": "Wt",
        "species": "Mus musculus",
        "strain": "C57/B6",
        "dob": "2025-01-11",
        "session_description": "This is but a test.",
        "experimenter": "John Doe",
        "lab": "Doe lab",
        "institution": "University of Someplace with lots of funding",
    }

    if partial:
        metadata.pop("lab")
        metadata.pop("genotype")

    with open(tmpfile, "w") as fp:
        json.dump(metadata, fp)

    return str(tmpfile)
