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
import click
import h5py
import dask

import time

import numpy as np
import dask_image.ndfilters as difilters

from dask import array as da
from matplotlib import pyplot as plt


@click.command()
@click.argument("raw_path", type=click.Path(exists=True))
@click.argument("out_dir", type=click.Path(dir_okay=True))
@click.option("--chunks", default=100, help="Number of chunks to load in memory.")
@click.option("--channel-means-only", is_flag=True, show_default=True, default=False)
def preprocess(raw_path, out_dir, chunks=100, channel_means_only=False):
    """Preprocessing to extract deltaF from a single session.

    Preprocessing separates the two channels, applies the haemodynamic correction,
    and extracts the delta F signal.

    Args:
        raw: Path to raw HDF5 file
        out_dir: Path to output directory for preprocessed data. This directory doesn't have to exist.

    """
    click.echo("Preprocessing file {}.".format(raw_path))

    preprocessing_start = time.time()

    session_id = raw_path.split("/")[-1].replace(".h5", "")
    os.makedirs(out_dir, exist_ok=True)

    qa_dir = out_dir + os.sep + "qa"
    os.makedirs(qa_dir, exist_ok=True)

    click.echo("Loading data...")

    # Lazy-load the data into a dask array
    f = h5py.File(raw_path)
    d = f["/frames"]

    raw_frames = da.from_array(d, chunks=(chunks, d.shape[1], d.shape[2]))

    # 2x2 binning
    raw_frames = raw_frames.reshape(
        d.shape[0],
        1,
        d.shape[1] / 2,
        d.shape[1] // (d.shape[1] / 2),
        d.shape[2] / 2,
        d.shape[2] // (d.shape[2] / 2),
    ).mean(axis=(-1, 1, 3))
    click.echo("2x2 binning to shape {}".format(raw_frames.shape))

    # Channel separation
    # Get the global mean and std values for each frame
    click.echo("Calculating frame means & standard deviations...")
    start = time.time()
    frame_means, frame_stds = dask.compute(
        raw_frames.mean(axis=(1, 2)), raw_frames.std(axis=(1, 2))
    )
    end = time.time()
    click.echo(
        "Frame means & standard deviations calculated in {} s".format(end - start)
    )

    plt.clf()
    plt.hist(frame_means)
    outpath = qa_dir + os.sep + session_id + "_qa_frame_means_histogram.png"
    plt.savefig(outpath)
    click.echo("Saved histogram for frame means at {}".format(outpath))

    plt.clf()
    plt.plot(frame_means)
    outpath = qa_dir + os.sep + session_id + "_qa_frame_means_line.png"
    plt.savefig(outpath)
    click.echo("Saved lineplot for frame means at {}".format(outpath))

    plt.clf()
    plt.hist(frame_stds)
    outpath = qa_dir + os.sep + session_id + "_qa_frame_std_histogram.png"
    plt.savefig(outpath)
    click.echo("Saved histogram for frame std at {}".format(outpath))

    plt.clf()
    plt.plot(frame_stds)
    outpath = qa_dir + os.sep + session_id + "_qa_frame_std_line.png"
    plt.savefig(outpath)
    click.echo("Saved lineplot for frame std at {}".format(outpath))

    # Threshold based on standard deviation - 470nm frames have a higher std than 405nm ones
    std_threshold = frame_stds.mean()
    click.echo("Standard deviation threshold is {}".format(std_threshold))

    gcamp_filter = frame_stds > std_threshold
    isosb_filter = frame_stds < std_threshold

    # Check that the separation works
    click.echo("Separating channels...")
    start = time.time()
    gcamp_mean, isosb_mean = dask.compute(
        raw_frames[gcamp_filter].mean(axis=(1, 2)),
        raw_frames[isosb_filter].mean(axis=(1, 2)),
    )
    end = time.time()
    click.echo("Channel means calculated in {} s".format(end - start))

    plt.clf()
    plt.plot(gcamp_mean)
    plt.plot(isosb_mean)
    outpath = qa_dir + os.sep + session_id + "_qa_channel_means.png"
    plt.savefig(outpath)
    click.echo("Saved channel means at {}".format(outpath))

    gcamp_mean.tofile(
        qa_dir + os.sep + session_id + "_qa_gcamp_channel_means.txt", sep=","
    )
    isosb_mean.tofile(
        qa_dir + os.sep + session_id + "_qa_isosb_channel_means.txt", sep=","
    )

    if channel_means_only:
        click.echo("Channel means saved as txt files. Exiting.")
        return

    # Generate the mean gcamp frame and its std
    click.echo("Generating mean gcamp frame and its maximum intensity projection...")
    start = time.time()
    gcamp_mean_frame, gcamp_std_frame, gcamp_maxip = dask.compute(
        raw_frames[gcamp_filter].mean(axis=0),
        raw_frames[gcamp_filter].std(axis=0),
        raw_frames[gcamp_filter].max(axis=0),
    )
    end = time.time()
    click.echo(
        "GCaMP average frame, std and maximum intensity projection calculated in {} s".format(
            end - start
        )
    )

    plt.clf()
    outpath = qa_dir + os.sep + session_id + "_qa_gcamp_mean.png"
    plt.imsave(outpath, gcamp_mean_frame)
    click.echo("Saved gcamp average frame at {}".format(outpath))

    plt.clf()
    outpath = qa_dir + os.sep + session_id + "_qa_gcamp_std.png"
    plt.imsave(outpath, gcamp_std_frame)
    click.echo("Saved gcamp standard deviation frame at {}".format(outpath))

    plt.clf()
    outpath = qa_dir + os.sep + session_id + "_qa_gcamp_maxip.png"
    plt.imsave(outpath, gcamp_maxip)
    click.echo("Saved gcamp maximum intensity projection at {}".format(outpath))

    # Generate the mean isosbestic frame and its std
    click.echo(
        "Generating mean isosbestic frame and its maximum intensity projection..."
    )
    start = time.time()
    isosb_mean_frame, isosb_std_frame, isosb_maxip = dask.compute(
        raw_frames[isosb_filter].mean(axis=0),
        raw_frames[isosb_filter].std(axis=0),
        raw_frames[isosb_filter].max(axis=0),
    )
    end = time.time()
    click.echo(
        "Isosbestic average frame, std and maximum intensity projection calculated in {} s".format(
            end - start
        )
    )

    plt.clf()
    outpath = qa_dir + os.sep + session_id + "_qa_isosb_mean.png"
    plt.imsave(outpath, isosb_mean_frame)
    click.echo("Saved isosbestic average frame at {}".format(outpath))

    plt.clf()
    outpath = qa_dir + os.sep + session_id + "_qa_isosb_std.png"
    plt.imsave(outpath, isosb_std_frame)
    click.echo("Saved isosbestic standard deviation frame at {}".format(outpath))

    plt.clf()
    outpath = qa_dir + os.sep + session_id + "_qa_isosb_maxip.png"
    plt.imsave(outpath, isosb_maxip)
    click.echo("Saved isosbestic maximum intensity projection at {}".format(outpath))

    # Calculate the mean gcamp:isosb ratio frame
    start = time.time()
    gcamp_isosb_mean_ratio = gcamp_mean_frame / isosb_mean_frame
    end = time.time()
    click.echo("Mean GCaMP:Isosb ratio calculated in {} s".format(end - start))

    plt.clf()
    outpath = qa_dir + os.sep + session_id + "_qa_gcamp_isosb_mean_ratio.png"
    plt.imsave(outpath, gcamp_isosb_mean_ratio)
    click.echo("Saved mean gcamp:isosb ratio frame at {}".format(outpath))

    # Max common index (to avoid array overflow)
    if len(gcamp_mean) != len(isosb_mean):
        click.echo("WARNING: GCaMP & Isosb channels have mismatching indexes")
    max_idx = min(len(gcamp_mean), len(isosb_mean))

    click.echo("Extracting F signal per pixel and smoothing with a 3x3 Gaussian...")
    f_signal = da.true_divide(
        raw_frames[gcamp_filter][:max_idx],
        raw_frames[isosb_filter][:max_idx],
    )

    f_signal = f_signal - gcamp_isosb_mean_ratio

    f_signal = da.true_divide(f_signal, gcamp_isosb_mean_ratio, dtype=np.float32)

    f_signal = difilters.gaussian_filter(f_signal, sigma=[0, 3, 3])

    f_signal.visualize(
        filename=qa_dir + os.sep + session_id + "_calc_f_signal_graph.png"
    )

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

    preprocessing_end = time.time()
    click.echo(
        "Preprocessing took a total of {} mins.".format(
            (preprocessing_end - preprocessing_start) / 60
        )
    )


def rebin(arr, new_shape):
    shape = (
        new_shape[0],
        arr.shape[0] // new_shape[0],
        new_shape[1],
        arr.shape[1] // new_shape[1],
    )
    return arr.reshape(shape).mean(-1).mean(1)
