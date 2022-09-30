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

"""Preprocessing submodule."""
import os
import shutil
import click
import h5py
import dask
import zarr

import time

import numpy as np
from dask import array as da
from matplotlib import pyplot as plt


@click.command()
@click.argument("raw_path", type=click.Path(exists=True))
@click.argument("out_dir", type=click.Path(dir_okay=True))
@click.option("--chunks", default=100, help="Number of chunks to load in memory.")
@click.option("--crop", default=0)
@click.option("--bins", default=2, help="Binning.")
@click.option("--channel-means-only", is_flag=True, show_default=True, default=False)
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
    raw_path,
    out_dir,
    chunks=100,
    crop=0,
    bins=2,
    channel_means_only=False,
    use_means=False,
    flip_channels=False,
    interim_dir="interim/",
    skip_start=None,
    skip_end=None,
):
    """Preprocessing to extract deltaF from a single session.

    Preprocessing separates the two channels, applies the haemodynamic correction,
    and extracts the delta F signal.

    Args:
        raw_path: Path to raw HDF5 file
        out_dir: Path to output directory for preprocessed data. This directory doesn't have to exist.

    """
    click.echo("Preprocessing file {}.".format(raw_path))

    preprocessing_start = time.time()

    session_id = raw_path.split("/")[-1].replace(".h5", "")
    os.makedirs(out_dir, exist_ok=True)

    qa_dir = out_dir + os.sep + "qa"
    os.makedirs(qa_dir, exist_ok=True)

    os.makedirs(interim_dir, exist_ok=True)

    click.echo("Loading data...")

    # Lazy-load the data into a dask array
    f = h5py.File(raw_path)
    d = f["/frames"]
    ts = f["/timestamps"]

    if skip_end:
        skip_end = -skip_end

    if skip_start or skip_end:
        d = f["/frames"][skip_start:skip_end]
        ts = f["/timestamps"][skip_start:skip_end]

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
    binned_frames = bin(
        raw_frames, bins=bins, interim_dir=interim_dir, session_id=session_id
    )
    end = time.time()
    click.echo("Binned frames saved in {} s".format(end - start))

    del raw_frames

    # Channel separation
    # Get the global mean and std values for each frame
    click.echo("Calculating frame means & standard deviations...")
    start = time.time()
    gcamp_filter, isosb_filter = calc_channel_filters(
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

    plt.clf()
    plt.plot(gcamp_mean)
    plt.plot(isosb_mean)
    outpath = qa_dir + os.sep + session_id + "_qa_channel_means.png"
    plt.savefig(outpath)
    click.echo("Saved channel means at {}".format(outpath))

    if channel_means_only:
        click.echo("Channel means saved as txt files. Exiting.")
        return

    # Generate the mean gcamp frame and its std
    click.echo("Generating mean gcamp frame and its maximum intensity projection...")
    start = time.time()
    channel_qa(
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
    channel_qa(
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
    gcamp_dff = channel_dff(
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
    isosb_dff = channel_dff(
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

    plt.clf()
    plt.plot(gcamp_signal_mean)
    plt.plot(isosb_signal_mean)
    outpath = qa_dir + os.sep + session_id + "_qa_channel_signal_mean.png"
    plt.savefig(outpath)
    click.echo("Saved lineplot for channel signal {}".format(outpath))

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
    da.to_hdf5(outpath, "/F", f_signal, compression="lzf")
    end = time.time()
    click.echo("F signal calculated in {} s".format(end - start))
    click.echo("Saved F signal at {}".format(outpath))

    plt.clf()
    outpath = qa_dir + os.sep + session_id + "_qa_f_example.png"
    plt.imsave(outpath, f_signal[200])
    click.echo("Saved F example at {}".format(outpath))

    click.echo("Calculating mean F per frame...")
    f_signal_mean = f_signal.mean(axis=(1, 2)).compute()

    plt.clf()
    plt.plot(f_signal_mean)
    outpath = qa_dir + os.sep + session_id + "_qa_f_signal_mean.png"
    plt.savefig(outpath)
    click.echo("Saved lineplot for F signal {}".format(outpath))

    # Save timestamps
    outpath = out_dir + os.sep + session_id + "_preprocessed.h5"
    click.echo("Appending timestamps to {}".format(outpath))
    timestamps = da.from_array(ts[gcamp_filter], chunks="auto")
    da.to_hdf5(outpath, "/ts", timestamps)

    preprocessing_end = time.time()
    click.echo(
        "Preprocessing took a total of {} mins.".format(
            (preprocessing_end - preprocessing_start) / 60
        )
    )

    shutil.rmtree(interim_dir)


def bin(array, bins, interim_dir=".", session_id="null"):
    binned_array = array.reshape(
        array.shape[0],
        1,
        array.shape[1] / bins,
        array.shape[1] // (array.shape[1] / bins),
        array.shape[2] / bins,
        array.shape[2] // (array.shape[2] / bins),
    ).mean(axis=(-1, 1, 3), dtype=np.float32)
    interim_path = interim_dir + os.sep + session_id + "_binned.zarr"
    return store_interim(binned_array, interim_path)


def store_interim(array, interim_path, compute=True, chunks=500):
    z_interim = zarr.open_array(
        interim_path,
        shape=array.shape,
        dtype=array.dtype,
        chunks=(chunks, array.shape[1], array.shape[2]),
    )
    return array.store(z_interim, return_stored=True, compute=compute)


def calc_channel_filters(
    array,
    qa_dir=".",
    session_id="null",
    use_means=False,
    flip_channels=False,
    interim_dir=None,
):
    frame_means, frame_stds = dask.compute(
        array.mean(axis=(1, 2), dtype=np.float32),
        array.std(axis=(1, 2), dtype=np.float32),
    )

    outpath = qa_dir + os.sep + session_id + "_qa_frame_means_histogram.png"
    msg = "Saved histogram for frame means at {}".format(outpath)
    plot_hist(frame_means, outpath, message=msg)

    outpath = qa_dir + os.sep + session_id + "_qa_frame_means_line.png"
    msg = "Saved lineplot for frame means at {}".format(outpath)
    plot_line(frame_means, outpath, message=msg)

    outpath = qa_dir + os.sep + session_id + "_qa_frame_std_histogram.png"
    msg = "Saved histogram for frame means at {}".format(outpath)
    plot_hist(frame_stds, outpath, message=msg)

    outpath = qa_dir + os.sep + session_id + "_qa_frame_std_line.png"
    msg = "Saved lineplot for frame means at {}".format(outpath)
    plot_line(frame_stds, outpath, message=msg)

    threshold = frame_stds.mean()
    gcamp_filter = frame_stds > threshold
    isosb_filter = frame_stds < threshold

    if use_means:
        threshold = frame_means.mean()
        gcamp_filter = frame_means > threshold
        isosb_filter = frame_means < threshold

    if flip_channels:
        return isosb_filter, gcamp_filter

    return gcamp_filter, isosb_filter


def channel_qa(array, channel_filter, qa_dir=".", session_id="null", channel="null"):
    mean_frame, std_frame, maxip = dask.compute(
        array[channel_filter].mean(axis=0),
        array[channel_filter].std(axis=0),
        array[channel_filter].max(axis=0),
    )

    plt.clf()
    outpath = qa_dir + os.sep + session_id + "_qa_{}_mean.png".format(channel)
    plt.imsave(outpath, mean_frame)

    plt.clf()
    outpath = qa_dir + os.sep + session_id + "_qa_{}_std.png".format(channel)
    plt.imsave(outpath, std_frame)

    plt.clf()
    outpath = qa_dir + os.sep + session_id + "_qa_{}_maxip.png".format(channel)
    plt.imsave(outpath, maxip)


def channel_dff(
    array,
    channel_filter,
    window_width,
    channel_name="null",
    interim_dir=".",
    session_id="null",
):
    cumsum_vec = da.cumsum(
        da.insert(array[channel_filter], 0, 0, axis=0), dtype=np.uint32, axis=0
    )

    interim_path = (
        interim_dir + os.sep + session_id + "_" + channel_name + "_cumsum.zarr"
    )
    cumsum_vec = store_interim(cumsum_vec, interim_path)

    f0 = da.true_divide(
        (cumsum_vec[window_width:] - cumsum_vec[:-window_width]),
        window_width,
        dtype=np.float32,
    )

    interim_path = interim_dir + os.sep + session_id + "_" + channel_name + "_f0.zarr"
    f0 = store_interim(f0, interim_path)

    f0_start = da.mean(f0[: int(window_width / 2)]).compute()
    f0_end = da.mean(f0[-int(window_width / 2) :]).compute()

    f0 = da.insert(
        f0,
        da.arange(0, int(window_width / 2) - 1),
        f0_start,
        axis=0,
    )

    f0 = da.insert(
        f0,
        da.arange(f0.shape[0] - int(window_width / 2), f0.shape[0]),
        f0_end,
        axis=0,
    )

    interim_path = (
        interim_dir + os.sep + session_id + "_" + channel_name + "_f0_appended.zarr"
    )
    f0 = store_interim(f0, interim_path)

    dff = da.true_divide(da.subtract(array[channel_filter], f0), f0, dtype=np.float32)

    interim_path = interim_dir + os.sep + session_id + "_" + channel_name + "_dff.zarr"

    return store_interim(dff, interim_path)


def plot_line(array, outpath, message=None):
    plt.clf()
    plt.plot(array)
    plt.savefig(outpath)
    if message:
        click.echo(message)


def plot_hist(array, outpath, message=None):
    plt.clf()
    plt.hist(array)
    plt.savefig(outpath)
    if message:
        click.echo(message)
    pass
