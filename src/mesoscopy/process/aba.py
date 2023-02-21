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
import itertools
import click
import h5py
import imageio

import time

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.pyplot as plt


@click.group()
def aba():
    pass


@aba.command()
@click.argument("recording_path", type=click.Path(exists=True))
@click.option("-o", "--out_dir", type=click.Path(dir_okay=True), default="./")
@click.option("-a", "--atlas", type=click.Path(dir_okay=False))
@click.option("-n", "--annotations", type=click.Path(dir_okay=False))
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
def activity(
    recording_path, atlas, annotations, out_dir, skip_start=None, skip_end=None
):
    """Extract area responses based on the Allen Brain Atlas"""
    click.echo("Processing file {}.".format(recording_path))

    processing_start = time.time()

    session_id = (
        recording_path.split("/")[-1]
        .replace(".h5", "")
        .replace("_preprocessed-registered", "")
    )
    os.makedirs(out_dir, exist_ok=True)

    qa_dir = out_dir + os.sep + "qa"
    os.makedirs(qa_dir, exist_ok=True)

    click.echo("Loading recording file...")
    f = h5py.File(recording_path)

    d = f["/F"]
    ts = f["/ts"]

    if skip_end:
        skip_end = -skip_end

    if skip_start or skip_end:
        d = f["/F"][skip_start:skip_end]
        ts = f["/ts"][skip_start:skip_end]

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
        annotations.iterrows(),
        label="Calculating mean deltaF per area...",
        length=annotations.shape[0],
    ) as areas:
        for _, area in areas:
            for idx in range(total_frames):

                l_mask = np.ma.masked_array(d[idx], np.not_equal(l_aba, area.id))
                r_mask = np.ma.masked_array(d[idx], np.not_equal(r_aba, area.id))
                activity.append(
                    {
                        "frame": idx,
                        "area": "L_" + area.acronym,
                        "F": np.ma.mean(l_mask),
                        "std": np.ma.std(l_mask),
                        "timestamp": ts[idx],
                    }
                )
                activity.append(
                    {
                        "frame": idx,
                        "area": "R_" + area.acronym,
                        "F": np.ma.mean(r_mask),
                        "std": np.ma.std(r_mask),
                        "timestamp": ts[idx],
                    }
                )

    outpath = out_dir + os.sep + session_id + "_area-activity.csv"
    print("Saving to {}".format(outpath))
    df = pd.DataFrame(activity)
    df["z_score"] = df.groupby("area")["F"].apply(stats.zscore)
    df.to_csv(outpath)

    processing_end = time.time()
    click.echo(
        "Processing took a total of {} mins.".format(
            (processing_end - processing_start) / 60
        )
    )


@aba.command()
@click.argument("activity_path", type=click.Path(exists=True))
@click.option("-o", "--out_dir", type=click.Path(dir_okay=True), default="./")
@click.option("-b", "--behaviour_path", type=click.Path(exists=True))
@click.option("--stimulus_interval", type=float, default=4.0)
def sync_behaviour(activity_path, behaviour_path, out_dir, stimulus_interval=4):
    click.echo("Labelling epochs for {} from {}.".format(activity_path, behaviour_path))
    processing_start = time.time()

    session_id = activity_path.split("/")[-1].replace("_area-activity.csv", "")
    os.makedirs(out_dir, exist_ok=True)

    behaviour_df = pd.read_csv(behaviour_path, parse_dates=["timestamp"])
    behaviour_df.loc[
        behaviour_df.response_time < 0, "response_time"
    ] = stimulus_interval
    behaviour_df["timestamp_end"] = (
        behaviour_df.timestamp
        + pd.to_timedelta(behaviour_df.iti, unit="s")
        + pd.to_timedelta(behaviour_df["response_time"], unit="s")
    )

    img_df = pd.read_csv(activity_path)
    img_df["timestamp"] = pd.to_datetime(img_df.timestamp.str.strip("b'"))

    session_df = behaviour_df[
        (behaviour_df.session_date == str(img_df.timestamp[0].date()))
    ]

    trials = []

    for idx, row in session_df.reset_index().iterrows():

        if idx < 5:
            continue

        trial_start = row.timestamp
        trial_end = row.timestamp_end

        trial = img_df[
            (img_df.timestamp > trial_start)
            & (img_df.timestamp < trial_end + pd.Timedelta(seconds=1))
        ].drop("Unnamed: 0", axis=1)

        if trial.empty:
            continue

        trial_type = None
        if (row.outcome == "correct") & (row.response_time > 0):
            trial_type = "hit"
        elif (row.outcome == "correct") & (row.response_time < 0):
            trial_type = "correct_rejection"
        elif (row.outcome == "incorrect") & (row.response_time > 0):
            trial_type = "false_alarm"
        elif row.outcome == "precued":
            trial_type = "precued"
        elif row.outcome == "no_response":
            trial_type = "miss"

        trial["epoch"] = ""

        trial.loc[
            trial.timestamp < (trial_start + pd.Timedelta(seconds=row.iti)), "epoch"
        ] = "iti"
        trial.loc[
            trial.timestamp >= (trial_start + pd.Timedelta(seconds=row.iti)), "epoch"
        ] = "cue_response"
        trial.loc[trial.timestamp >= trial_end, "epoch"] = (
            "reward" if row.outcome == "correct" else "post_trial"
        )

        if trial.loc[trial.epoch == "cue_response"].empty:
            continue

        cue_onset = trial.loc[trial.epoch == "cue_response"].iloc[0].timestamp
        trial["cue_offset"] = (
            (trial.timestamp - cue_onset)
            .dt.total_seconds()
            .apply(lambda x: 0.040 * round(float(x) / 0.040))
        )
        trial["idx"] = idx
        trial["trial_type"] = trial_type
        trial["correction"] = row.correction

        trial["warped_offset"] = pd.Series(
            np.interp(
                trial.loc[trial.epoch == "cue_response"].cue_offset,
                (0, trial.loc[trial.epoch == "cue_response"].cue_offset.max()),
                (0, +1),
            ),
            index=trial[trial.epoch == "cue_response"].index,
        ).apply(lambda x: 0.040 * round(float(x) / 0.040))
        trial.loc[trial.epoch == "iti", "warped_offset"] = pd.Series(
            np.interp(
                trial.loc[trial.epoch == "iti"].cue_offset,
                (
                    trial.loc[trial.epoch == "iti"].cue_offset.min(),
                    0,
                ),
                (-5, 0),
            ),
            index=trial[trial.epoch == "iti"].index,
        ).apply(lambda x: 0.040 * round(float(x) / 0.040))
        trial.loc[trial.epoch == "reward", "warped_offset"] = pd.Series(
            np.interp(
                trial.loc[trial.epoch == "reward"].cue_offset,
                (
                    trial.loc[trial.epoch == "reward"].cue_offset.min(),
                    trial.loc[trial.epoch == "reward"].cue_offset.max(),
                ),
                (1, 1.5),
            ),
            index=trial[trial.epoch == "reward"].index,
        ).apply(lambda x: 0.040 * round(float(x) / 0.040))

        trials.append(trial)

    df = pd.concat(trials)

    outpath = out_dir + os.sep + session_id + "_area-activity-epochs.csv"
    print("Saving to {}".format(outpath))
    df.to_csv(outpath)

    processing_end = time.time()
    click.echo(
        "Processing took a total of {} mins.".format(
            (processing_end - processing_start) / 60
        )
    )


@aba.command()
@click.argument("activity_path", type=click.Path(exists=True))
@click.option("-o", "--out_dir", type=click.Path(dir_okay=True), default="./")
@click.option("-a", "--annotations", type=click.Path(dir_okay=False))
@click.option("--epoch", type=str, default=None)
def connectivity(activity_path, annotations, out_dir, epoch=None):
    click.echo("Generating connectivity for {}...".format(activity_path))
    processing_start = time.time()

    df = pd.read_csv(activity_path)

    if epoch:
        df = df.loc[df.epoch == epoch]

    session_id = (
        activity_path.split("/")[-1]
        .replace("_area-activity.csv", "")
        .replace("_area-activity-epochs.csv", "")
    )
    os.makedirs(out_dir, exist_ok=True)

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

    l_annotations = ["L_" + acr for acr in annotations.acronym.tolist()]
    r_annotations = ["R_" + acr for acr in annotations.acronym.tolist()]

    areas_personsr = []
    with click.progressbar(
        itertools.combinations(l_annotations + r_annotations, 2),
        label="Calculating activity correlations for ABA pairs...",
        length=sum(
            [1 for _ in itertools.combinations(l_annotations + r_annotations, 2)]
        ),
    ) as pairs:
        for pair in pairs:
            if pair[0] == pair[1]:
                continue
            stim = df[df.area == pair[0]]["F"].to_numpy()
            resp = df[df.area == pair[1]]["F"].to_numpy()
            corr = stats.pearsonr(stim, resp)
            areas_personsr.append(
                {
                    "stim": pair[0],
                    "resp": pair[1],
                    "r": corr[0],
                    "p": corr[1],
                }
            )
            areas_personsr.append(
                {
                    "stim": pair[1],
                    "resp": pair[0],
                    "r": corr[0],
                    "p": corr[1],
                }
            )

    outpath = out_dir + os.sep + session_id + "_area-connectivity.csv"
    print("Saving to {}".format(outpath))
    df_pearsons = pd.DataFrame(areas_personsr)
    df_pearsons.to_csv(outpath)

    processing_end = time.time()
    click.echo(
        "Processing took a total of {} mins.".format(
            (processing_end - processing_start) / 60
        )
    )


@aba.command()
@click.argument("activity_path", type=click.Path(exists=True))
@click.option("-o", "--out_dir", type=click.Path(dir_okay=True), default="./")
@click.option("--epoch", type=str, default=None)
@click.option("--key", type=str, default=None, help="Activity column")
def cross_correlations(activity_path, out_dir, key="F", epoch=None):
    click.echo("Generating cross correlation lags for {}...".format(activity_path))
    processing_start = time.time()

    df = pd.read_csv(activity_path)

    if epoch:
        df = df.loc[df.epoch == epoch]

    session_id = (
        activity_path.split("/")[-1]
        .replace("_area-activity.csv", "")
        .replace("_area-activity-epochs.csv", "")
    )
    os.makedirs(out_dir, exist_ok=True)

    annotations = df.area.unique()

    areas_lags = []
    with click.progressbar(
        itertools.combinations(annotations, 2),
        label="Calculating activity cross-correlation lags for ABA pairs...",
        length=sum([1 for _ in itertools.combinations(annotations, 2)]),
    ) as pairs:
        for pair in pairs:
            if pair[0] == pair[1]:
                continue
            stim = df[df.area == pair[0]][key].to_numpy()
            resp = df[df.area == pair[1]][key].to_numpy()
            xcorr = signal.correlate(stim, resp, mode="full")
            lags = signal.correlation_lags(stim.size, resp.size, mode="full")
            lag = lags[np.argmax(xcorr)]
            areas_lags.append(
                {
                    "stim": pair[0],
                    "resp": pair[1],
                    "lag": lag,
                }
            )
            areas_lags.append(
                {
                    "stim": pair[1],
                    "resp": pair[0],
                    "lag": lag,
                }
            )

    outpath = out_dir + os.sep + session_id + "_area-xcorr-lags.csv"
    print("Saving to {}".format(outpath))
    df_pearsons = pd.DataFrame(areas_lags)
    df_pearsons.to_csv(outpath)

    processing_end = time.time()
    click.echo(
        "Processing took a total of {} mins.".format(
            (processing_end - processing_start) / 60
        )
    )


@aba.command()
@click.argument("epochs_path", type=click.Path(exists=True))
@click.option("-o", "--out_dir", type=click.Path(dir_okay=True), default="./")
@click.option("--trial", type=str)
@click.option(
    "--with-profile",
    is_flag=True,
    show_default=True,
    default=False,
    help="Export response profile.",
)
def average_trial(epochs_path, out_dir, trial, with_profile=False):
    click.echo(
        "Generating average warped timeseries for trial type '{}'...".format(trial)
    )
    processing_start = time.time()

    session_id = epochs_path.split("/")[-1].replace("_area-activity-epochs.csv", "")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(epochs_path)

    df_warped = (
        df[df.trial_type == trial]
        .groupby(["area", "warped_offset"])
        .mean()
        .reset_index()
        .drop(["correction", "idx", "Unnamed: 0", "frame"], axis=1)
    )

    df_realtime = (
        df[df.trial_type == trial]
        .groupby(["area", "cue_offset"])
        .mean()
        .reset_index()
        .drop(["correction", "idx", "Unnamed: 0", "frame"], axis=1)
    )

    outpath = (
        out_dir
        + os.sep
        + session_id
        + "_area-activity-epochs-avg_trial-{}.csv".format(trial)
    )
    print("Saving to {}".format(outpath))
    df_realtime.to_csv(outpath)

    outpath = (
        out_dir
        + os.sep
        + session_id
        + "_area-activity-epochs-avg_trial-{}-warped.csv".format(trial)
    )
    print("Saving to {}".format(outpath))
    df_warped.to_csv(outpath)

    if with_profile:
        print("Generating response profile...")
        areas_of_interest = [
            "R_MOp1",
            "R_MOs",
            "R_VISp",
            "R_VISa1",
            "R_VISal1",
            "L_MOp1",
            "L_MOs",
            "L_VISp",
            "L_VISa1",
            "L_VISal1",
        ]
        # t2h, peak, spread
        df = df[(df.trial_type == trial) & (df.area.isin(areas_of_interest))]

        trials = []
        for area in areas_of_interest:
            for idx in sorted(df.idx.unique()):
                trial_response = df[(df.idx == idx) & (df.area == area)]

                iti = trial_response[
                    (trial_response.cue_offset > -2) & (trial_response.cue_offset <= 0)
                ]
                response = trial_response[
                    (trial_response.cue_offset > 0) & (trial_response.cue_offset <= 2)
                ]

                iti_offset = iti.z_score.mean()

                avg_response = response.z_score.mean()
                peak = response.z_score.max()
                half_peak = peak / 2

                hp_timing = (
                    response[response.z_score >= half_peak]
                    .cue_offset.sort_values()
                    .values
                )
                t2h = hp_timing[0] if hp_timing.size > 0 else np.nan

                trials.append(
                    {
                        "trial_idx": idx,
                        "area": area,
                        "t2h": t2h,
                        "peak": peak - iti_offset,
                        "half_peak": half_peak - iti_offset,
                        "avg_response": avg_response - iti_offset,
                    }
                )

        df_prof = pd.DataFrame(trials)

        outpath = (
            out_dir
            + os.sep
            + session_id
            + "_area-activity-epochs-{}-profile.csv".format(trial)
        )
        print("Saving to {}".format(outpath))
        df_prof.to_csv(outpath)

    processing_end = time.time()
    click.echo(
        "Processing took a total of {} mins.".format(
            (processing_end - processing_start) / 60
        )
    )


@aba.command()
@click.argument("recording_path", type=click.Path(exists=True))
@click.option("-o", "--out_dir", type=click.Path(dir_okay=True), default="./")
@click.option("-e", "--epochs-path", type=click.Path(dir_okay=False))
@click.option("--trial-type", type=str, default="hit")
@click.option("--vmin", type=float, default=0)
@click.option("--vmax", type=float, default=0.10)
def trial_average_map(recording_path, out_dir, epochs_path, trial_type, vmin, vmax):
    click.echo("Processing file {}.".format(recording_path))

    processing_start = time.time()

    session_id = (
        recording_path.split("/")[-1]
        .replace(".h5", "")
        .replace("_preprocessed-registered", "")
    )
    os.makedirs(out_dir, exist_ok=True)

    click.echo("Loading recording file...")
    f = h5py.File(recording_path)
    d = f["/F"]

    df = pd.read_csv(epochs_path)

    with click.progressbar(
        sorted(df[df.trial_type == trial_type].cue_offset.unique())
    ) as offsets:
        for idx, offset in enumerate(offsets):
            frame_ids = (
                df[
                    (df.cue_offset == offset)
                    & (df.trial_type == trial_type)
                    & (df.frame > 1500)
                    & (df.frame < df.frame.max() - 1500)
                ]
                .frame.unique()
                .tolist()
            )
            frame = d[frame_ids].mean(axis=0)

            out_path = (
                out_dir
                + os.sep
                + "{}_frame{:04d}_offset_{:.2f}.png".format(session_id, idx, offset)
            )
            plt.imsave(out_path, frame, cmap="jet", vmin=vmin, vmax=vmax)

    processing_end = time.time()
    click.echo(
        "Processing took a total of {} mins.".format(
            (processing_end - processing_start) / 60
        )
    )
