import pytest
from click.testing import CliRunner

import numpy as np
from dask import array as da

import mesoscopy.preprocess as preprocess
import mesoscopy.preprocess.calculations as calculations


def test_load_raw_h5(raw_h5):
    session_id, imaging_data, timestamps = preprocess.load_raw(raw_h5, nwb=False)
    assert session_id == "preproc_test"
    assert imaging_data.shape == (600, 142, 142)
    assert len(timestamps) == 600


def test_load_raw_nwb(nwbfile):
    session_id, imaging_data, timestamps = preprocess.load_raw(nwbfile, nwb=True)
    assert session_id == "session_1234"
    assert imaging_data.shape == (600, 142, 142)
    assert len(timestamps) == 600


def test_channel_qa():
    pass


def test_nwb_link(nwbfile, preproc_h5):
    nwb = preprocess.update_nwb(nwbfile, preproc_h5)
    assert nwb.processing["ophys"]["DeltaFSeries"].data
    assert nwb.processing["ophys"]["DeltaFSeries"].data.shape == (300, 142, 142)
    assert nwb.processing["ophys"]["DeltaFSeries"].timestamps


def test_binning(output_dir):
    array = calculations.bin_array(
        array=da.random.random((300, 142, 142)),
        bins=2,
        interim_dir=output_dir,
        session_id="null",
    )
    assert array.shape == (300, 71, 71)


def test_separate_channels(output_dir, random_idx):
    # Generate mock dual-channel imaging data
    mock_gcamp = np.random.normal(70, 200, size=(300, 142, 142))
    mock_isosb = np.random.normal(65, 50, size=(300, 142, 142))

    # Merge the two channels in random order
    mock_dual_channel = np.insert(
        mock_gcamp, random_idx - np.arange(len(random_idx)), mock_isosb, axis=0
    )

    gcamp_filter, isosb_filter = calculations.separate_channels(
        array=da.from_array(mock_dual_channel, chunks=(100, 142, 142)),
        qa_dir=output_dir,
        session_id="null",
    )

    assert mock_dual_channel[gcamp_filter].shape[0] == 300
    assert mock_dual_channel[isosb_filter].shape[0] == 300
    assert (np.where(isosb_filter) == random_idx).all()


def test_separate_channels_use_means(output_dir, random_idx):
    # Generate mock dual-channel imaging data
    mock_gcamp = np.random.normal(200, 50, size=(300, 142, 142))
    mock_isosb = np.random.normal(65, 50, size=(300, 142, 142))

    # Merge the two channels in random order
    mock_dual_channel = np.insert(
        mock_gcamp, random_idx - np.arange(len(random_idx)), mock_isosb, axis=0
    )

    gcamp_filter, isosb_filter = calculations.separate_channels(
        array=da.from_array(mock_dual_channel, chunks=(100, 142, 142)),
        qa_dir=output_dir,
        session_id="null",
        use_means=True,
    )

    assert mock_dual_channel[gcamp_filter].shape[0] == 300
    assert mock_dual_channel[isosb_filter].shape[0] == 300
    assert (np.where(isosb_filter) == random_idx).all()


def test_separate_channels_flip_channels(output_dir, random_idx):
    # Generate mock dual-channel imaging data
    mock_gcamp = np.random.normal(70, 200, size=(300, 142, 142))
    mock_isosb = np.random.normal(65, 50, size=(300, 142, 142))

    # Merge the two channels in random order
    mock_dual_channel = np.insert(
        mock_gcamp, random_idx - np.arange(len(random_idx)), mock_isosb, axis=0
    )

    gcamp_filter, isosb_filter = calculations.separate_channels(
        array=da.from_array(mock_dual_channel, chunks=(100, 142, 142)),
        qa_dir=output_dir,
        session_id="null",
        flip_channels=True,
    )

    assert mock_dual_channel[gcamp_filter].shape[0] == 300
    assert mock_dual_channel[isosb_filter].shape[0] == 300
    assert (np.where(gcamp_filter) == random_idx).all()


def test_preprocess_h5(raw_h5, output_dir):
    # preprocess.run_preprocessing(
    #     path=raw_h5, out_dir=output_dir, interim_dir=output_dir
    # )
    ...


def test_preprocess_h5_crop(raw_h5): ...


def test_preprocess_h5_frameskip(raw_h5): ...


def test_preprocess_h5_bin(raw_h5): ...


def test_preprocess_nwb(nwbfile): ...


def test_preprocess_nwb_crop(nwbfile): ...


def test_preprocess_nwb_frameskip(nwbfile): ...


def test_preprocess_nwb_bin(nwbfile): ...


def test_preprocess_nwb_channel_means(nwbfile): ...


def test_preprocess_nwb_use_means(nwbfile): ...


def test_preprocess_nwb_flip_channels(nwbfile): ...


def test_preprocess_nwb_use_means_flip_channels(nwbfile): ...


def test_preprocess_nwb_interim_dir(nwbfile): ...


def test_preprocess_nwb_chunksize(nwbfile): ...
