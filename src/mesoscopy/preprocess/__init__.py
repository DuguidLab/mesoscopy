#  Copyright (c) 2022 Constantinos Eleftheriou <Constantinos.Eleftheriou@ed.ac.uk>.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this
#   software and associated documentation files (the "Software"), to deal in the
#   Software without restriction, including without limitation the rights to use, copy,
#   modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#   and to permit persons to whom the Software is furnished to do so, subject to the
#  following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies
#  or substantial portions of the Software
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
#  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
#  IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
#  IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
import os
import typing
import shutil
import click
import dask

import time

import mesoscopy.io as io
import mesoscopy.plots as plots
import mesoscopy.preprocess.calculations as calc

import numpy as np
from dask import array as da

from pynwb.image import ImageSeries


@click.command()
@click.argument(
    "path",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--out_dir",
    type=click.Path(dir_okay=True),
    default="./",
    help="Output directory for preprocessed recording.",
)
@click.option("--chunks", default=100, help="Number of chunks to load in memory.")
@click.option(
    "--crop",
    default=0,
    help="Number of pixels to crop from the edges of the recording.",
)
@click.option("--bins", default=2, help="Recording pixel binning factor.")
@click.option(
    "--channel-means-only",
    is_flag=True,
    show_default=True,
    default=False,
    help="Extract the channel means and exit without extracting a delta F series.",
)
@click.option(
    "--use-means",
    is_flag=True,
    show_default=True,
    default=False,
    help="Separate channels using means histogram instead of standard deviation.",
)
@click.option(
    "--flip-channels",
    is_flag=True,
    show_default=True,
    default=False,
    help="Flip channel order.",
)
@click.option("--interim_dir", type=click.Path(dir_okay=True), default="interim/")
@click.option(
    "--skip-start",
    default=None,
    type=int,
    help="Number of frames to skip at the start of the recording.",
)
@click.option(
    "--skip-end",
    default=None,
    type=int,
    help="Number of frames to skip at the end of the recording.",
)
def preprocess(
    path: str,
    out_dir: str,
    chunks: int = 100,
    crop: int = 0,
    bins: int = 2,
    channel_means_only: bool = False,
    use_means: bool = False,
    flip_channels: bool = False,
    interim_dir: str = "interim/",
    skip_start: typing.Optional[int] = None,
    skip_end: typing.Optional[int] = None,
) -> None:
    """Preprocessing to extract deltaF from a single session dual-channel mixed recording.

    Preprocessing separates the two channels, applies the haemodynamic correction,
    and extracts the delta F signal.

    Args:
        raw_path (str): Path to the raw recording HDF5 or NWB file.
        out_dir (str): Path to the output directory.
        chunks (int, optional): Number of chunks to load in memory. Defaults to 100.
        crop (int, optional): Number of pixels to crop from the edges of the recording. Defaults to 0.
        bins (int, optional): Recording pixel binning factor. Defaults to 2.
        channel_means_only (bool, optional): Extract the channel means and exit without extracting a delta F series. Defaults to False.
        use_means (bool, optional): Use means histogram instead of standard deviation to separate channels. Defaults to False.
        flip_channels (bool, optional): Flip extracted channel order. Defaults to False.
        interim_dir (str, optional): Path to the interim directory. Defaults to "interim/".
        skip_start (int, optional): Number of frames to skip at the start of the recording. Defaults to None.
        skip_end (int, optional): Number of frames to skip at the end of the recording. Defaults to None.
    """
    click.echo("Preprocessing file {}.".format(path))

    preprocessing_start = time.time()

    os.makedirs(out_dir, exist_ok=True)

    qa_dir = out_dir + os.sep + "qa"
    os.makedirs(qa_dir, exist_ok=True)

    os.makedirs(interim_dir, exist_ok=True)

    click.echo("Loading data...")

    # Determine whether we're working with an NWB file
    nwb = True if path.endswith(".nwb") else False

    session_id, d, ts = _load_raw(path, nwb=nwb)

    if skip_end:
        skip_end = -skip_end

    if skip_start or skip_end:
        d = d[skip_start:skip_end]
        ts = ts[skip_start:skip_end]

    raw_frames = da.from_array(d, chunks="auto")
    if chunks > 0:
        raw_frames = raw_frames.rechunk(chunks=(chunks, d.shape[1], d.shape[2]))

    if crop > 0:
        raw_frames = raw_frames[:, crop:-crop, crop:-crop]
        click.echo("Cropping to shape {}".format(raw_frames.shape))

    # Binning
    click.echo(
        "{}x{} binning to shape {} by {}".format(
            bins, bins, raw_frames.shape[1] // bins, raw_frames.shape[2] // bins
        )
    )
    start = time.time()
    binned_frames = calc.bin_array(
        raw_frames, bins=bins, interim_dir=interim_dir, session_id=session_id
    )
    end = time.time()
    click.echo("Binned frames saved in {} s".format(end - start))

    del raw_frames

    # Channel separation
    # Get the global mean and std values for each frame
    click.echo("Calculating frame means & standard deviations...")
    start = time.time()
    gcamp_filter, isosb_filter = calc.separate_channels(
        binned_frames,
        session_id=session_id,
        use_means=use_means,
        flip_channels=flip_channels,
        qa_dir=qa_dir,
    )
    end = time.time()
    click.echo(
        "Frame means & standard deviations calculated in {} s".format(end - start)
    )

    # Check that the separation works
    click.echo("Separating channels...")
    start = time.time()
    gcamp_mean, isosb_mean = dask.compute(
        binned_frames[gcamp_filter].mean(axis=(1, 2), dtype=np.float32),
        binned_frames[isosb_filter].mean(axis=(1, 2), dtype=np.float32),
    )
    end = time.time()
    click.echo("Channel means calculated in {} s".format(end - start))

    outpath = qa_dir + os.sep + session_id + "_qa_channel_means.png"
    plots.plot_lines(
        [gcamp_mean, isosb_mean],
        outpath,
        message="Saved channel means at {}".format(outpath),
    )

    if channel_means_only:
        click.echo("Channel means saved as txt files. Exiting.")
        return

    # Generate the mean gcamp frame and its std
    click.echo("Generating mean gcamp frame and its maximum intensity projection...")
    start = time.time()
    _channel_qa(
        binned_frames,
        gcamp_filter,
        qa_dir=qa_dir,
        session_id=session_id,
        channel="gcamp",
    )
    end = time.time()
    click.echo(
        "GCaMP average frame, std and maximum intensity projection calculated in {} s".format(
            end - start
        )
    )

    # Generate the mean isosbestic frame and its std
    click.echo(
        "Generating mean isosbestic frame and its maximum intensity projection..."
    )
    start = time.time()
    _channel_qa(
        binned_frames,
        isosb_filter,
        qa_dir=qa_dir,
        session_id=session_id,
        channel="isosb",
    )
    end = time.time()
    click.echo(
        "Isosbestic average frame, std and maximum intensity projection calculated in {} s".format(
            end - start
        )
    )

    # Calculate the dff per channel using a rolling baseline (mean in a 30s window)

    window_width = 30 * 25

    click.echo("Calculating ∂F for the gcamp channel...")
    start = time.time()
    gcamp_dff = calc.channel_dff(
        binned_frames,
        gcamp_filter,
        window_width,
        channel_name="gcamp",
        interim_dir=interim_dir,
        session_id=session_id,
    )
    end = time.time()
    click.echo("gcamp ∂F calculated in {} s".format(end - start))

    click.echo("Calculating ∂F for the isosb channel...")
    start = time.time()
    isosb_dff = calc.channel_dff(
        binned_frames,
        isosb_filter,
        window_width,
        channel_name="isosb",
        interim_dir=interim_dir,
        session_id=session_id,
    )
    end = time.time()
    click.echo("isosb ∂F calculated in {} s".format(end - start))

    click.echo("Calculating mean ∂F per frame for gcamp and isosb channels...")
    start = time.time()
    gcamp_signal_mean, isosb_signal_mean = da.compute(
        gcamp_dff.mean(axis=(1, 2)), isosb_dff.mean(axis=(1, 2))
    )
    end = time.time()
    click.echo("Channel signal means calculated in {} s".format(end - start))

    outpath = qa_dir + os.sep + session_id + "_qa_channel_signal_mean.png"
    plots.plot_lines(
        [gcamp_signal_mean, isosb_signal_mean],
        outpath,
        message="Saved lineplot for channel signal {}".format(outpath),
    )

    # Max common index (to avoid array overflow)
    if len(gcamp_mean) != len(isosb_mean):
        click.echo("WARNING: GCaMP & Isosb channels have mismatching indexes")
    max_idx = min(len(gcamp_mean), len(isosb_mean))

    click.echo("Extracting corrected F signal (gcamp - isosb)...")
    f_signal = da.subtract(
        gcamp_dff[:max_idx],
        isosb_dff[:max_idx],
    )

    # f_signal.visualize(
    #     filename=qa_dir + os.sep + session_id + "_calc_f_signal_graph.png"
    # )

    outpath = out_dir + os.sep + session_id + "_preprocessed.h5"
    start = time.time()
    da.to_hdf5(outpath, "/data", f_signal, compression="lzf")
    end = time.time()
    click.echo("F signal calculated in {} s".format(end - start))
    click.echo("Saved F signal at {}".format(outpath))

    outpath = qa_dir + os.sep + session_id + "_qa_f_example.png"
    plots.plot_frame(
        f_signal[200],
        outpath,
        message="Saved F example at {}".format(outpath),
    )

    click.echo("Calculating mean F per frame...")
    f_signal_mean = f_signal.mean(axis=(1, 2)).compute()

    outpath = qa_dir + os.sep + session_id + "_qa_f_signal_mean.png"
    plots.plot_line(
        f_signal_mean,
        outpath,
        message="Saved lineplot for F signal {}".format(outpath),
    )

    # Save timestamps
    outpath = out_dir + os.sep + session_id + "_preprocessed.h5"
    click.echo("Appending timestamps to {}".format(outpath))
    timestamps = da.from_array(ts[gcamp_filter], chunks="auto")
    da.to_hdf5(outpath, "/timestamps", timestamps)

    preprocessing_end = time.time()
    click.echo(
        "Preprocessing took a total of {} mins.".format(
            (preprocessing_end - preprocessing_start) / 60
        )
    )

    if nwb:
        click.echo("Updating NWB file...")
        _update_nwb(path, outpath)
        click.echo("Updated NWB file at {}".format(path))

    click.echo("Cleaning up...")
    shutil.rmtree(interim_dir)


def _load_raw(
    raw_path: str, nwb: bool = False
) -> tuple[str, da.Array | np.ndarray, da.Array | np.ndarray]:
    """Load raw imaging data from an HDF5 or NWB file.

    Args:
        raw_path (str): Path to the raw recording HDF5 or NWB file.
        nwb (bool, optional): Whether the file is an NWB file. Defaults to False.

    Returns:
        tuple[str, da.Array | np.ndarray, da.Array | np.ndarray]: Session ID, imaging data, and timestamps.
    """
    if nwb:
        nwbfile = io.read_nwb(raw_path)

        session_id = nwbfile.identifier

        imaging_data = nwbfile.acquisition["DualChannelImagingSeries"].data
        timestamps = nwbfile.acquisition["DualChannelImagingSeries"].timestamps
    else:
        session_id = raw_path.split("/")[-1].replace(".h5", "")

        # Lazy-load the data into a dask array
        f_raw = io.read_h5(raw_path)
        imaging_data = f_raw["/frames"]
        timestamps = f_raw["/timestamps"]

    return session_id, imaging_data, timestamps


def _update_nwb(nwb_path: str, h5_path: str) -> None:
    """Update an NWB file with a delta F imaging series stored in an HDF5 file.

    Creates a link between the NWB file and the HDF5 file. See https://pynwb.readthedocs.io/en/stable/tutorials/advanced_io/linking_data.html.

    Args:
        nwb_path (str): Path to NWB file.
        h5_path (str): Path to HDF5 file containing the delta F imaging series.
    """
    f = io.read_h5(h5_path)
    nwbfile, nwbio = io.read_nwb(nwb_path, return_io=True)
    deltaF_series = ImageSeries(
        name="DeltaFSeries",
        data=f["/data"],
        timestamps=f["/timestamps"],
        unit="df/f",
        description="dF/F widefield cortical imaging series.",
        comments="This imaging series is corrected for the haemodynamic response.",
    )

    ophys_module = nwbfile.create_processing_module(
        name="ophys", description="optical physiology processed data"
    )

    ophys_module.add(deltaF_series)

    io.write_nwb(nwb_path, nwbfile, io=nwbio)


def _channel_qa(
    array: da.Array | np.ndarray,
    channel_filter: list | da.Array | np.ndarray,
    qa_dir: str = ".",
    session_id: str = "null",
    channel: str = "null",
) -> None:
    """Generate QA plots for a channel.

    Args:
        array (da.Array | np.ndarray): Imaging data array.
        channel_filter (list | da.Array | np.ndarray): Calculated channel filter.
        qa_dir (str, optional): Directory to save QA plots. Defaults to ".".
        session_id (str, optional): Session identifier. Defaults to "null".
        channel (str, optional): Channel name. Defaults to "null".
    """
    mean_frame, std_frame, maxip = dask.compute(
        array[channel_filter].mean(axis=0),
        array[channel_filter].std(axis=0),
        array[channel_filter].max(axis=0),
    )

    outpath = qa_dir + os.sep + session_id + "_qa_{}_mean.png".format(channel)
    plots.plot_frame(
        mean_frame, outpath, message="Saved mean frame at {}".format(outpath)
    )

    outpath = qa_dir + os.sep + session_id + "_qa_{}_std.png".format(channel)
    plots.plot_frame(
        std_frame, outpath, message="Saved std frame at {}".format(outpath)
    )

    outpath = qa_dir + os.sep + session_id + "_qa_{}_maxip.png".format(channel)
    plots.plot_frame(maxip, outpath, message="Saved maxip frame at {}".format(outpath))
