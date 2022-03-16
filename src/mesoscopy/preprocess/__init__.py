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
import numba

import time

from dask import array as da
from matplotlib import pyplot as plt


@click.command()
@click.argument("raw_path", type=click.Path(exists=True))
@click.argument("out_dir", type=click.Path(dir_okay=True))
def preprocess(raw_path, out_dir):
    """Preprocessing to extract deltaF from a single session.

    Preprocessing separates the two channels, applies the haemodynamic correction,
    realigns the images to a common coordinate space (ABA) and extracts the delta F signal.

    Args:
        raw: Path to raw HDF5 file
        out_dir: Path to output directory for preprocessed data. This directory doesn't have to exist.

    """
    click.echo("Preprocessing file {}.".format(raw_path))

    session_id = raw_path.split("/")[-1].replace(".h5", "")
    os.makedirs(out_dir, exist_ok=True)

    qa_dir = out_dir + os.sep + "qa"
    os.makedirs(qa_dir, exist_ok=True)

    # Lazy-load the data into a dask array
    f = h5py.File(raw_path)
    d = f["/frames"]

    raw_frames = da.from_array(d, chunks="auto")

    # Channel separation
    # Get the global mean and std values for each frame
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

    # Check that the separation works
    start = time.time()
    gcamp_mean, isosb_mean = dask.compute(
        raw_frames[frame_stds > std_threshold].mean(axis=(1, 2)),
        raw_frames[frame_stds < std_threshold].mean(axis=(1, 2)),
    )
    end = time.time()
    click.echo("Channel means calculated in {} s".format(end - start))

    plt.clf()
    plt.plot(gcamp_mean)
    plt.plot(isosb_mean)
    outpath = qa_dir + os.sep + session_id + "_qa_channel_means.png"
    plt.savefig(outpath)
    click.echo("Saved channel means at {}".format(outpath))

    # Generate the mean isosbestic frame
    start = time.time()
    isosb_mean_frame = raw_frames[frame_stds < std_threshold].mean(axis=0).compute()
    end = time.time()
    click.echo("Isosbestic average frame calculated in {} s".format(end - start))

    plt.clf()
    outpath = qa_dir + os.sep + session_id + "_qa_isosb_mean.png"
    plt.imsave(outpath, isosb_mean_frame)
    click.echo("Saved isosbestic average frame at {}".format(outpath))


def separate_channels():
    pass


def haemodynamic_correction():
    pass


def register_to_frame():
    pass


def realign():
    pass


def extract_signal():
    pass
