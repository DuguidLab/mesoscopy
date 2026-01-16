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
import shutil
import time
import typing

import click
import dask
import numpy as np
from dask import array as da
from pynwb.image import ImageSeries

import mesoscopy.preprocess.compute as calc
import mesoscopy.preprocess.qa as qa
from mesoscopy import io
from mesoscopy import timer


@click.command(name="preprocess")
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
@click.option(
    "--no-qa",
    is_flag=True,
    show_default=True,
    default=False,
    help="Skip automatic quality control checks.",
)
def preprocess_cmd(
    **kwargs: typing.Any,
) -> None:
    """This command will preprocess a single session dual-channel mixed recording to extract a deltaF signal corrected for the haemodynamic response."""
    run_preprocessing(**kwargs)


def run_preprocessing(
    path: str,
    out_dir: str,
    chunks: int = 100,
    crop: int = 0,
    bins: int = 2,
    channel_means_only: bool = False,
    use_means: bool = False,
    flip_channels: bool = False,
    interim_dir: str = "interim/",
    skip_start: int | None = None,
    skip_end: int | None = None,
    no_qa: bool = False,
) -> None:
    """Preprocessing to extract deltaF from a single session dual-channel mixed recording.

    Preprocessing separates the two channels, applies the haemodynamic correction,
    and extracts the delta F signal.

    Args:
        path (str): Path to the raw recording HDF5 or NWB file.
        out_dir (str): Path to the output directory.
        chunks (int, optional): Number of chunks to load in memory. Defaults to 100.
        crop (int, optional): Number of pixels to crop from the edges of the recording. Defaults to 0.
        bins (int, optional): Recording pixel binning factor. Defaults to 2.
        channel_means_only (bool, optional): Extract the channel means and exit without extracting a delta F series.
        Defaults to False.
        use_means (bool, optional): Use means histogram instead of standard deviation to separate channels.
        Defaults to False.
        flip_channels (bool, optional): Flip extracted channel order. Defaults to False.
        interim_dir (str, optional): Path to the interim directory. Defaults to "interim/".
        skip_start (int, optional): Number of frames to skip at the start of the recording. Defaults to None.
        skip_end (int, optional): Number of frames to skip at the end of the recording. Defaults to None.
        no_qa (bool, optional): Skip automatic quality control checks. Defaults to False.
    """
    click.echo(f"Preprocessing file {path}.")

    preprocessing_start = time.time()

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(interim_dir, exist_ok=True)

    click.echo("Loading data...")

    # Determine whether we're working with an NWB file
    nwb = bool(path.endswith(".nwb"))

    session_id, d, ts = load_raw(path, nwb=nwb)

    if skip_start and skip_start != 0:
        skip_start -= 1

    # If odd or even skips, add one to make them even to avoid strangeness
    if skip_start and skip_start % 2 != 0:
        skip_start += 1

    if skip_end and skip_end % 2 != 0:
        skip_end += 1

    if skip_end:
        skip_end = -abs(skip_end)

    if skip_start or skip_end:
        d = d[skip_start:skip_end]
        ts = ts[skip_start:skip_end]

    raw_frames = da.from_array(d, chunks="auto")
    if chunks > 0:
        raw_frames = raw_frames.rechunk(chunks=(chunks, d.shape[1], d.shape[2]))

    if crop > 0:
        raw_frames = raw_frames[:, crop:-crop, crop:-crop]
        click.echo(f"Cropping to shape {raw_frames.shape}")

    with timer.Timer(
        f"Binning frames to shape {raw_frames.shape[1] // bins} by {raw_frames.shape[2] // bins} ({bins}x{bins})"
    ):
        binned_frames = calc.bin_array(
            raw_frames,
            bins=bins,
            interim_dir=interim_dir,
            session_id=session_id,
        )

    del raw_frames

    # Channel separation
    # Get the global mean and std values for each frame
    with timer.Timer("Calculating frame means & standard deviations"):
        frame_means, frame_stds = calc.frame_statistics(binned_frames)

    # Generate the channel separation filters
    with timer.Timer("Generating channel separation filters"):
        gcamp_filter, isosb_filter = calc.channel_separation_filters(
            frame_means,
            frame_stds,
            use_means=use_means,
            flip_channels=flip_channels,
        )

    with timer.Timer("Calculating channel mean timeseries"):
        gcamp_mean, isosb_mean = dask.compute(
            binned_frames[gcamp_filter].mean(axis=(1, 2), dtype=np.float32),
            binned_frames[isosb_filter].mean(axis=(1, 2), dtype=np.float32),
        )

    if channel_means_only:
        np.savetxt(
            os.path.join(out_dir, f"{session_id}_gcamp_mean.txt"),
            gcamp_mean,
            fmt="%.4f",
        )
        np.savetxt(
            os.path.join(out_dir, f"{session_id}_isosb_mean.txt"),
            isosb_mean,
            fmt="%.4f",
        )
        click.echo("Channel means saved as txt files. Exiting.")
        return

    gcamp_channel = binned_frames[gcamp_filter]
    isosb_channel = binned_frames[isosb_filter]

    with timer.Timer("Generating projection images per channel"):
        gcamp_projections = calc.projections(gcamp_channel)
        isosb_projections = calc.projections(isosb_channel)

    # Calculate the dff per channel using a rolling baseline (mean in a 30s window)
    window_width = 30 * 25

    if window_width > binned_frames.shape[0]:
        click.echo(
            f"WARNING: Default window width is larger than the number of frames. Setting window width to {
                binned_frames.shape[0] // 4
            }."
        )
        window_width = binned_frames.shape[0] // 4

    with timer.Timer("Calculating dF/F per channel"):
        gcamp_dff = calc.rolling_dff(
            gcamp_channel,
            window_width=window_width,
            channel_name="gcamp",
            interim_dir=interim_dir,
            session_id=session_id,
        )
        isosb_dff = calc.rolling_dff(
            isosb_channel,
            window_width=window_width,
            channel_name="isosb",
            interim_dir=interim_dir,
            session_id=session_id,
        )

    with timer.Timer("Calculating mean ∂F per frame per channel"):
        gcamp_signal_mean, isosb_signal_mean = da.compute(gcamp_dff.mean(axis=(1, 2)), isosb_dff.mean(axis=(1, 2)))

    # Max common index (to avoid array overflow)
    if len(gcamp_mean) != len(isosb_mean):
        click.echo("WARNING: GCaMP & Isosb channels have mismatching indexes")
    max_idx = min(len(gcamp_mean), len(isosb_mean))

    click.echo("Extracting corrected F signal (gcamp - isosb)...")

    with timer.Timer("Calculating F signal"):
        f_signal = da.subtract(
            gcamp_dff[:max_idx],
            isosb_dff[:max_idx],
        )

    with timer.Timer("Calculating mean F timeseries"):
        f_mean_timeseries = f_signal.mean(axis=(1, 2))

    with timer.Timer("Generating maximum intensity projection for F signal"):
        f_maxip = calc.projections(f_signal).get("maxip")

    timestamps = da.from_array(np.array(ts[gcamp_filter[:max_idx]], dtype="S25"), chunks="auto")

    # Quality control checks
    qa_histogram_separation = None
    qa_timestamp_consistency = None
    qa_timestamp_jump = None
    qa_check_noise, qa_noise_levels = None, None
    qa_check_snr, qa_snr = None, None
    qa_check_bleaching, qa_bleaching = None, None
    if not no_qa:
        click.echo("Running QA checks...")

        qa_histogram_separation = qa.check_histogram_separation(np.array(frame_stds if not use_means else frame_means))
        if qa_histogram_separation:
            click.echo("✅ Histogram separation check passed.")
        else:
            click.echo("❌ ERROR - histogram separation failed, channels could not be fully separated!")

        qa_timestamp_consistency = qa.check_timestamp_consistency(timestamps)
        if qa_timestamp_consistency:
            click.echo("✅ Timestamp consistency check passed, low interval standard deviation.")
        else:
            click.echo("❌ ERROR - timestamp consistency check failed, timestamp drift detected!!")

        qa_timestamp_jump = qa.check_timestamp_jumps(timestamps)
        if qa_timestamp_jump:
            click.echo("✅ Timestamp jump check passed, no timestamp interval above two standard deviations.")
        else:
            click.echo("❌ ERROR - timestamp jump check failed, intervals above two standard deviations detected!")

        qa_check_noise, qa_noise_levels = qa.check_noise(f_signal, return_noise=True)
        if qa_check_noise:
            click.echo(f"✅ Noise level check passed, median recording shot noise {np.median(qa_noise_levels):.2f} %.")
        else:
            click.echo(
                "❌ ERROR - noise level check failed, recording has high noise "
                f"(median {np.median(qa_noise_levels):.2f} %)!"
            )

        qa_check_snr, qa_snr = qa.check_snr(f_signal, return_snr=True)
        if qa_check_snr:
            click.echo(f"✅ SNR check passed (recording SNR is {qa_snr:.2f}).")
        else:
            click.echo(f"❌ ERROR - SNR check failed (recording SNR is {qa_snr:.2f})!")

        qa_check_bleaching, qa_bleaching = qa.check_bleaching(f_mean_timeseries, return_slope=True)
        if qa_check_bleaching:
            click.echo("✅ Photobleaching check passed.")
        else:
            click.echo(f"❌ ERROR - photobleaching check failed (bleaching factor is {qa_bleaching})")

    outpath = out_dir + os.sep + session_id + "_preprocessed.h5"
    with timer.Timer(f"Writing data to {outpath}"):
        # da.to_hdf5(
        io.write_h5(
            path=outpath,
            data={
                "/F": f_signal,
                "/timestamps": timestamps,
                "/qa/frame_means_timeseries": frame_means,
                "/qa/frame_stds_timeseries": frame_stds,
                "/qa/gcamp_mean_timeseries": gcamp_mean[:max_idx],
                "/qa/isosb_mean_timeseries": isosb_mean[:max_idx],
                "/qa/gcamp_dff_timeseries": gcamp_signal_mean[:max_idx],
                "/qa/isosb_dff_timeseries": isosb_signal_mean[:max_idx],
                "/qa/gcamp_filter": gcamp_filter[:max_idx],
                "/qa/isosb_filter": isosb_filter[:max_idx],
                "/qa/gcamp_mean_projection": gcamp_projections.get("mean"),
                "/qa/gcamp_std_projection": gcamp_projections.get("std"),
                "/qa/gcamp_maxip_projection": gcamp_projections.get("maxip"),
                "/qa/isosb_mean_projection": isosb_projections.get("mean"),
                "/qa/isosb_std_projection": isosb_projections.get("std"),
                "/qa/isosb_maxip_projection": isosb_projections.get("maxip"),
                "/qa/f_mean_timeseries": f_mean_timeseries,
                "/qa/f_maxip": f_maxip,
                "/qa/noise_levels": qa_noise_levels,
            },
            attributes={
                "/qa/checks/histogram_separation": qa_histogram_separation,
                "/qa/checks/timestamp_consistency": qa_timestamp_consistency,
                "/qa/checks/timestamp_jump": qa_timestamp_jump,
                "/qa/checks/noise_check": qa_check_noise,
                "/qa/checks/snr_check": qa_check_snr,
                "/qa/checks/snr": qa_snr,
                "/qa/checks/bleaching_check": qa_check_bleaching,
                "/qa/checks/bleaching_factor": qa_bleaching,
            },
            compression="lzf",
        )

    preprocessing_end = time.time()
    click.echo(f"Preprocessing took a total of {(preprocessing_end - preprocessing_start) / 60} mins.")

    if nwb:
        click.echo("Updating NWB file...")
        update_nwb(path, outpath)
        click.echo(f"Updated NWB file at {path}")

    click.echo("Cleaning up...")
    shutil.rmtree(interim_dir)


def load_raw(raw_path: str, nwb: bool = False) -> tuple[str, da.Array | np.ndarray, da.Array | np.ndarray]:
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

        imaging_data = nwbfile.acquisition["DualChannelImagingSeries"].data[:]
        timestamps = nwbfile.acquisition["DualChannelImagingSeries"].timestamps[:]
    else:
        session_id = raw_path.split("/")[-1].replace(".h5", "")

        # Lazy-load the data into a dask array
        f_raw = io.read_h5(raw_path)
        imaging_data = f_raw["/frames"]
        timestamps = f_raw["/timestamps"]

    return session_id, imaging_data, timestamps


def update_nwb(nwb_path: str, h5_path: str) -> None:
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
        data=f["/F"],
        timestamps=f["/timestamps"],
        unit="df/f",
        description="dF/F widefield cortical imaging series.",
        comments="This imaging series is corrected for the haemodynamic response.",
    )

    ophys_module = nwbfile.create_processing_module(name="ophys", description="optical physiology processed data")

    ophys_module.add(deltaF_series)

    io.write_nwb(nwb_path, nwbfile, io=nwbio)

    return nwbfile
