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
import imageio

import time

import numpy as np
import pandas as pd


@click.command()
@click.argument("recording_path", type=click.Path(exists=True))
@click.option("-o", "--out_dir", type=click.Path(dir_okay=True), default="./")
@click.option("-a", "--atlas", type=click.Path(dir_okay=False))
@click.option("-n", "--annotations", type=click.Path(dir_okay=False))
def aba(recording_path, atlas, annotations, out_dir):
    """Extract area responses based on the Allen Brain Atlas"""
    click.echo("Processing file {}.".format(recording_path))

    processing_start = time.time()

    session_id = (
        recording_path.split("/")[-1]
        .replace(".h5", "")
        .replace("preprocessed-registered", "processed")
    )
    os.makedirs(out_dir, exist_ok=True)

    qa_dir = out_dir + os.sep + "qa"
    os.makedirs(qa_dir, exist_ok=True)

    click.echo("Loading recording file...")
    # Lazy-load the data into a dask array
    f = h5py.File(recording_path)
    d = f["/F"]
    ts = f["/ts"]

    click.echo("Loading ABA mask...")
    annotations = pd.read_csv(annotations, delimiter=", ", engine="python")

    aba_exclude = [
        "FRP1",
        "VISpl1",
        "VISpor1",
        "VISli1",
        "TEa1",
        "AUDd1",
        "AUDp1",
        "AUDpo1",
        "AUDv1",
        "ORBm1",
    ]

    annotations = annotations[~annotations.acronym.isin(aba_exclude)]

    l_aba = imageio.imread(atlas)
    r_aba = np.flip(l_aba, axis=1)

    total_frames = d.shape[0]
    activity = []
    with click.progressbar(
        range(total_frames), label="Calculating mean deltaF per area..."
    ) as frame_ids:
        for idx in frame_ids:
            for _, area in annotations.iterrows():
                l_mask = np.ma.masked_array(d[idx], np.not_equal(l_aba, area.id))
                r_mask = np.ma.masked_array(d[idx], np.not_equal(r_aba, area.id))
                activity.append(
                    {
                        "frame": idx,
                        "area": "L_" + area.acronym,
                        "mean": np.ma.mean(l_mask),
                        "std": np.ma.std(l_mask),
                        "timestamp": ts[idx],
                    }
                )
                activity.append(
                    {
                        "frame": idx,
                        "area": "R_" + area.acronym,
                        "mean": np.ma.mean(r_mask),
                        "std": np.ma.std(r_mask),
                        "timestamp": ts[idx],
                    }
                )

    outpath = out_dir + os.sep + session_id + "processed.csv"
    print("Saving to {}".format(outpath))
    df = pd.DataFrame(activity)
    df.to_csv(outpath)
