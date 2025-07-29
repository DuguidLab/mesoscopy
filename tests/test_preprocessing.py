import pathlib

import numpy as np
import pytest
import zarr
from click.testing import CliRunner
from dask import array as da

import mesoscopy
from mesoscopy import preprocess
from mesoscopy.preprocess import compute


def test_load_raw_h5(raw_h5):
    session_id, imaging_data, timestamps = preprocess.load_raw(raw_h5, nwb=False)
    assert session_id == "preproc_test"
    assert imaging_data.shape == (600, 40, 40)
    assert len(timestamps) == 600


def test_load_raw_nwb(nwbfile):
    session_id, imaging_data, timestamps = preprocess.load_raw(nwbfile, nwb=True)
    assert session_id == "session_1234"
    assert imaging_data.shape == (600, 40, 40)
    assert len(timestamps) == 600


def test_channel_qa():
    pass


def test_nwb_link(nwbfile, preproc_h5):
    nwb = preprocess.update_nwb(nwbfile, preproc_h5)
    assert nwb.processing["ophys"]["DeltaFSeries"].data
    assert nwb.processing["ophys"]["DeltaFSeries"].data.shape == (300, 40, 40)
    assert nwb.processing["ophys"]["DeltaFSeries"].timestamps


def test_binning(output_dir):
    array = compute.bin_array(
        array=da.random.random((300, 40, 40)),
        bins=2,
        interim_dir=output_dir,
        session_id="null",
    )
    assert array.shape == (300, 20, 20)


def test_separate_channels(output_dir, random_idx):
    # Generate mock dual-channel imaging data
    mock_gcamp = np.random.normal(70, 200, size=(300, 40, 40))
    mock_isosb = np.random.normal(65, 50, size=(300, 40, 40))

    # Merge the two channels in random order
    mock_dual_channel = np.insert(mock_gcamp, random_idx - np.arange(len(random_idx)), mock_isosb, axis=0)

    gcamp_filter, isosb_filter = compute.channel_separation_filters(
        frame_means=mock_dual_channel.mean(axis=(1, 2)),
        frame_stds=mock_dual_channel.std(axis=(1, 2)),
        use_means=False,
        flip_channels=False,
    )

    assert mock_dual_channel[gcamp_filter].shape[0] == 300
    assert mock_dual_channel[isosb_filter].shape[0] == 300
    assert (isosb_filter == random_idx).all()


def test_separate_channels_use_means(output_dir, random_idx):
    # Generate mock dual-channel imaging data
    mock_gcamp = np.random.normal(200, 50, size=(300, 40, 40))
    mock_isosb = np.random.normal(65, 50, size=(300, 40, 40))

    # Merge the two channels in random order
    mock_dual_channel = np.insert(mock_gcamp, random_idx - np.arange(len(random_idx)), mock_isosb, axis=0)

    gcamp_filter, isosb_filter = compute.channel_separation_filters(
        frame_means=mock_dual_channel.mean(axis=(1, 2)),
        frame_stds=mock_dual_channel.std(axis=(1, 2)),
        use_means=True,
        flip_channels=False,
    )

    assert mock_dual_channel[gcamp_filter].shape[0] == 300
    assert mock_dual_channel[isosb_filter].shape[0] == 300
    assert (isosb_filter == random_idx).all()


def test_separate_channels_flip_channels(output_dir, random_idx):
    # Generate mock dual-channel imaging data
    mock_gcamp = np.random.normal(70, 200, size=(300, 40, 40))
    mock_isosb = np.random.normal(65, 50, size=(300, 40, 40))

    # Merge the two channels in random order
    mock_dual_channel = np.insert(mock_gcamp, random_idx - np.arange(len(random_idx)), mock_isosb, axis=0)

    gcamp_filter, isosb_filter = compute.channel_separation_filters(
        frame_means=mock_dual_channel.mean(axis=(1, 2)),
        frame_stds=mock_dual_channel.std(axis=(1, 2)),
        use_means=False,
        flip_channels=True,
    )

    assert mock_dual_channel[gcamp_filter].shape[0] == 300
    assert mock_dual_channel[isosb_filter].shape[0] == 300
    assert (gcamp_filter == random_idx).all()


def test_channel_dff(output_dir, random_idx):
    # Generate mock dual-channel imaging data
    mock_gcamp = np.random.normal(70, 200, size=(300, 20, 20))
    mock_isosb = np.random.normal(65, 50, size=(300, 20, 20))

    # Merge the two channels in random order
    mock_dual_channel = np.insert(mock_gcamp, random_idx - np.arange(len(random_idx)), mock_isosb, axis=0)

    gcamp_filter, _ = compute.channel_separation_filters(
        frame_means=mock_dual_channel.mean(axis=(1, 2)),
        frame_stds=mock_dual_channel.std(axis=(1, 2)),
        use_means=False,
        flip_channels=False,
    )

    dff = compute.rolling_dff(
        array=mock_dual_channel[gcamp_filter],
        interim_dir=output_dir,
        session_id="null",
        window_width=10,
    )
    assert dff.shape == (300, 20, 20)


def test_channel_dff_invalid_window(output_dir, random_idx):
    # Generate mock dual-channel imaging data
    mock_gcamp = np.random.normal(70, 200, size=(300, 20, 20))
    mock_isosb = np.random.normal(65, 50, size=(300, 20, 20))

    # Merge the two channels in random order
    mock_dual_channel = np.insert(mock_gcamp, random_idx - np.arange(len(random_idx)), mock_isosb, axis=0)

    gcamp_filter, isosb_filter = compute.channel_separation_filters(
        frame_means=mock_dual_channel.mean(axis=(1, 2)),
        frame_stds=mock_dual_channel.std(axis=(1, 2)),
        use_means=False,
        flip_channels=False,
    )

    with pytest.raises(ValueError):
        compute.rolling_dff(
            array=mock_dual_channel[gcamp_filter],
            interim_dir=output_dir,
            session_id="null",
            window_width=700,
        )


def test_channel_dff_insert(output_dir, random_idx):
    # Generate mock dual-channel imaging data
    mock_gcamp = np.random.normal(70, 200, size=(300, 20, 20))
    mock_isosb = np.random.normal(65, 50, size=(300, 20, 20))

    # Merge the two channels in random order
    mock_dual_channel = np.insert(mock_gcamp, random_idx - np.arange(len(random_idx)), mock_isosb, axis=0)

    gcamp_filter, _ = compute.channel_separation_filters(
        frame_means=mock_dual_channel.mean(axis=(1, 2)),
        frame_stds=mock_dual_channel.std(axis=(1, 2)),
        use_means=False,
        flip_channels=False,
    )

    _ = compute.rolling_dff(
        array=mock_dual_channel[gcamp_filter],
        interim_dir=output_dir,
        session_id="null",
        window_width=10,
    )

    # Assert insertion happens in a block
    f0 = zarr.load(pathlib.Path(output_dir) / "null_null_f0_appended.zarr")
    assert da.all(da.equal(f0[0], f0[1]))
    assert da.all(da.equal(f0[-2], f0[-1]))


def test_preprocess_h5(raw_h5, output_dir):
    preprocess.run_preprocessing(path=raw_h5, out_dir=output_dir, interim_dir=output_dir)


def test_preprocess_h5_crop(raw_h5, output_dir):
    preprocess.run_preprocessing(path=raw_h5, out_dir=output_dir, interim_dir=output_dir, crop=5)


def test_preprocess_h5_frameskip(raw_h5, output_dir):
    # Skip odd frames
    preprocess.run_preprocessing(
        path=raw_h5,
        out_dir=output_dir,
        interim_dir=output_dir,
        skip_start=5,
        skip_end=5,
    )

    # Skip even frames
    preprocess.run_preprocessing(
        path=raw_h5,
        out_dir=output_dir,
        interim_dir=output_dir,
        skip_start=50,
        skip_end=50,
    )


def test_preprocess_h5_bin(raw_h5, output_dir):
    preprocess.run_preprocessing(path=raw_h5, out_dir=output_dir, interim_dir=output_dir, bins=4)


def test_preprocess_h5_channelmeansonly(raw_h5, output_dir):
    preprocess.run_preprocessing(path=raw_h5, out_dir=output_dir, interim_dir=output_dir, channel_means_only=True)


def test_preprocess_nwb(nwbfile, output_dir):
    preprocess.run_preprocessing(path=nwbfile, out_dir=output_dir, interim_dir=output_dir)


def test_preprocess_nwb_crop(nwbfile, output_dir):
    preprocess.run_preprocessing(path=nwbfile, out_dir=output_dir, interim_dir=output_dir, crop=5)


def test_preprocess_nwb_frameskip_odd(nwbfile, output_dir):
    # Skip odd frames
    preprocess.run_preprocessing(
        path=nwbfile,
        out_dir=output_dir,
        interim_dir=output_dir,
        skip_start=5,
        skip_end=5,
    )


def test_preprocess_nwb_frameskip_even(nwbfile, output_dir):
    # Skip even frames
    preprocess.run_preprocessing(
        path=nwbfile,
        out_dir=output_dir,
        interim_dir=output_dir,
        skip_start=50,
        skip_end=50,
    )


def test_preprocess_nwb_bin(nwbfile, output_dir):
    preprocess.run_preprocessing(path=nwbfile, out_dir=output_dir, interim_dir=output_dir, bins=4)


def test_preprocess_nwb_channelmeansonly(nwbfile, output_dir):
    preprocess.run_preprocessing(
        path=nwbfile,
        out_dir=output_dir,
        interim_dir=output_dir,
        channel_means_only=True,
    )


def test_cmd_parity(raw_h5, output_dir):
    """Check if click preprocess cmd caller works"""
    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli,
        args=f"preprocess --out_dir {output_dir} --interim_dir {output_dir} {raw_h5}",
    )
    assert result.exit_code == 0
