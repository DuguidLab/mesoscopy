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

"""Processing submodule."""

from __future__ import annotations

import os
import typing
from pathlib import Path

import click
import h5py
import numpy as np

import mesoscopy.process.metrics as pm
import mesoscopy.process.perievent as pev
from mesoscopy import io
from mesoscopy import timer

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd


@click.group("process")
def process_cmd(): ...


@process_cmd.command("smooth")
@click.argument(
    "path",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--out_dir",
    type=click.Path(dir_okay=True),
    default="./",
    help="Output directory for smoothed recording.",
)
@click.option(
    "-s",
    "--sigma",
    type=int,
    default=2,
    help="Standard deviation of the Gaussian kernel, in pixels.",
)
def smooth_cmd(path: str, out_dir: str, sigma: int = 2) -> None:
    """Generate a smoothed DeltaF/F recording using a Laplace of Gaussian filter."""
    import mesoscopy.process.smooth as psm

    if not Path(out_dir).exists():
        click.echo(f"Creating output directory {out_dir}...")
        Path(out_dir).mkdir(parents=True)

    click.echo(f"Loading preprocessed recording from {path}...")
    # Determine whether we're working with an NWB file
    nwb = bool(path.endswith(".nwb"))
    session_id, deltaf_series, timestamps = io.load_deltaf(path, nwb=nwb)
    aligned = io.read_timestamps_aligned(path)

    outpath = out_dir + os.sep + session_id + "_smoothed.h5"

    with timer.Timer(message="Smoothing with LoG"):
        smoothed_deltaf = psm.laplace_gaussian(deltaf_series, sigma=sigma)

        outpath = io.write_h5(
            path=outpath,
            data={
                "/F": smoothed_deltaf,
                "/timestamps": timestamps,
            },
        )
        if aligned is not None:
            io.write_timestamps_aligned(outpath, *aligned)
    click.echo(f"Saved smoothed recording at {outpath}")


@process_cmd.command("zscore")
@click.argument(
    "path",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--out_dir",
    type=click.Path(dir_okay=True),
    default="./",
    help="Output directory for z-scored recording.",
)
def zscore_cmd(path: str, out_dir: str) -> None:
    """Pixel-wise z-score ∆F/F signal."""
    from pynwb.image import ImageSeries

    import mesoscopy.process.zscore as pzs

    if not Path(out_dir).exists():
        click.echo(f"Creating output directory {out_dir}...")
        Path(out_dir).mkdir(parents=True)

    click.echo(f"Loading preprocessed recording from {path}...")
    # Determine whether we're working with an NWB file
    nwb = bool(path.endswith(".nwb"))
    session_id, deltaf_series, timestamps = io.load_deltaf(path, nwb=nwb)
    aligned = io.read_timestamps_aligned(path)

    h5_outpath = out_dir + os.sep + session_id + "_zscored.h5"
    with timer.Timer(message="Z-scoring DeltaF/F"):
        zscored = pzs.zscore_deltaf(deltaf_series)
        h5_outpath = io.write_h5(
            path=h5_outpath,
            data={
                "/F": zscored,
                "/timestamps": timestamps,
            },
        )
        if aligned is not None:
            io.write_timestamps_aligned(h5_outpath, *aligned)

    click.echo(f"Saved z-scored recording at {h5_outpath}")

    # Append to NWB file
    if nwb:
        click.echo("Appending to NWB file...")
        nwbfile, nwbio = io.read_nwb(path, return_io=True)
        f = io.read_h5(h5_outpath)
        try:
            ophys_module = nwbfile.create_processing_module(
                name="ophys", description="optical physiology processed data"
            )
        except ValueError:
            click.echo("Processing module already exists...")
            ophys_module = nwbfile.processing["ophys"]

        zscored_series = ImageSeries(
            name="zScoredDeltaF",
            data=f["/F"],
            timestamps=f["/timestamps"],
            unit="df/f",
            description="z-scored dF/F widefield cortical imaging series",
        )

        ophys_module.add(zscored_series)

        io.write_nwb(path, nwbfile, io=nwbio)


@process_cmd.command("regions")
@click.argument(
    "path",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--out_dir",
    type=click.Path(dir_okay=True),
    default="./",
    help="Output directory for region activity file.",
)
@click.option(
    "-m",
    "--mask",
    "mask_paths",
    type=click.Path(exists=True),
    multiple=True,
    help=(
        "Path to a custom region mask file (NPY, NPZ or TIFF), drawn in registered frame coordinates. A boolean mask"
        " is extracted as a single region, named after the file (or, for NPZ files, after each key); an"
        " integer-labelled mask is extracted as one region per label, named '<name>_<label>'. Masks are used as"
        " drawn and are not mirrored across hemispheres. May be passed multiple times to combine masks from several"
        " files. Supplying a mask replaces the ABA region extraction unless --include-aba is given."
    ),
)
@click.option(
    "--include-aba",
    is_flag=True,
    default=False,
    help="Extract ABA-defined regions alongside any custom masks. Has no effect if no mask is supplied.",
)
# A click command's docstring doubles as its --help text, so it deliberately carries no Raises: section.
def regions_cmd(path: str, out_dir: str, mask_paths: tuple[str, ...], include_aba: bool) -> None:
    """Extract ∆F signal averages from ABA-defined regions, custom region masks, or both."""  # noqa: DOC501
    import pandas as pd

    import mesoscopy.process.region as pr

    if not Path(out_dir).exists():
        click.echo(f"Creating output directory {out_dir}...")
        Path(out_dir).mkdir(parents=True)

    masks: dict[str, np.ndarray] = {}
    for mask_path in mask_paths:
        click.echo(f"Loading region masks from {mask_path}...")
        try:
            file_masks = io.read_mask(mask_path)
        except ValueError as err:
            msg = f"Could not read region masks from {mask_path}: {err}"
            raise click.ClickException(msg) from err

        duplicates = [name for name in file_masks if name in masks]
        if duplicates:
            msg = (
                f"Region name(s) {', '.join(duplicates)} from {mask_path} are already defined by an earlier mask"
                " file. Rename the mask file, or its NPZ keys, so that every region name is unique."
            )
            raise click.ClickException(msg)

        masks.update(file_masks)

    if masks:
        click.echo(f"Loaded {len(masks)} region mask(s): {', '.join(masks)}")

    click.echo(f"Loading preprocessed recording from {path}...")
    # Determine whether we're working with an NWB file
    nwb = bool(path.endswith(".nwb"))
    session_id, deltaf_series, timestamps = io.load_deltaf(path, nwb=nwb)
    aligned = io.read_timestamps_aligned(path)

    outpath = out_dir + os.sep + session_id + "_regions.csv"

    with timer.Timer(message="Extracting region activity"):
        activity = []
        # Custom masks stand in for the ABA regions unless the ABA regions are explicitly asked for as well.
        if include_aba or not masks:
            activity.append(pd.DataFrame(pr.extract_all_regions(deltaf_series, as_dataframe=True)))
        if masks:
            try:
                activity.append(pd.DataFrame(pr.extract_all_masks(deltaf_series, masks, as_dataframe=True)))
            except ValueError as err:
                msg = str(err)
                raise click.ClickException(msg) from err

        region_activity = pd.concat(activity, ignore_index=True)
        time_idx = region_activity["time_idx"].to_numpy()
        region_activity["time_idx"] = [str(timestamp, encoding="utf-8") for timestamp in timestamps[time_idx]]
        region_activity.rename(columns={"time_idx": "timestamp"}, inplace=True)
        if aligned is not None:
            region_activity.insert(
                region_activity.columns.get_loc("timestamp") + 1, "time_aligned", aligned[0][time_idx]
            )
        region_activity.to_csv(outpath, index=False)

    click.echo(f"Saved region activity at {outpath}")


@process_cmd.command("regression")
@click.argument(
    "recording_path",
    type=click.Path(exists=True),
)
@click.argument(
    "regressor_path",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--out_dir",
    type=click.Path(dir_okay=True),
    default="./",
    help="Output directory for regression results.",
)
@click.option(
    "-a",
    "--alpha",
    type=float,
    default=1.0,
    help="Ridge regularisation strength. Defaults to 1.0.",
)
@click.option(
    "-n",
    "--nuisance-regressors",
    "nuisance_regressor_paths",
    type=click.Path(exists=True),
    multiple=True,
    help=(
        "Path to an external nuisance regressor file (NPZ or HDF5), e.g. behavioural motion energy. Every"
        " array/dataset in the file other than 'timestamps' is treated as one nuisance regressor and interpolated"
        " onto the recording's own timestamps before being z-scored and appended to the regressor matrix. May be"
        " passed multiple times to add nuisance regressors from several files."
    ),
)
@click.option(
    "-f",
    "--fast",
    is_flag=True,
    default=False,
    help=(
        "Use fast vectorised implementation of ridge regression. This is an experimental feature and may not work for"
        " all datasets. Use with caution. Defaults to False."
    ),
)
@click.option(
    "--npz",
    "file_format",
    flag_value="npz",
    default="npz",
    help="Save regression results as a compressed NumPy .npz file. Defaults to True.",
)
@click.option(
    "--h5",
    "file_format",
    flag_value="h5",
    help="Save regression results as an HDF5 file. Defaults to False.",
)
def regression_cmd(
    recording_path: str,
    regressor_path: str,
    out_dir: str,
    alpha: float,
    nuisance_regressor_paths: tuple[str, ...],
    fast: bool,
    file_format: str,
) -> None:
    """Perform pixel-wise ridge regression on a preprocessed ∆F/F recording."""  # noqa: DOC501
    import mesoscopy.process.regression as regr

    if not Path(out_dir).exists():
        click.echo(f"Creating output directory {out_dir}...")
        Path(out_dir).mkdir(parents=True)

    click.echo(f"Loading preprocessed recording from {recording_path}...")
    # Determine whether we're working with an NWB file
    nwb = bool(recording_path.endswith(".nwb"))
    session_id, deltaf_series, timestamps = io.load_deltaf(recording_path, nwb=nwb)

    click.echo(f"Loading regressors from {regressor_path}...")
    regressors, labels, trial_idx, regressor_alignment = io.read_regressors(regressor_path)
    labels = list(labels)

    recording_alignment = io.read_timestamps_aligned(recording_path)
    try:
        for warning in regr.check_alignment(recording_alignment, regressor_alignment):
            click.echo(f"WARNING: {warning}")
    except ValueError as err:
        raise click.ClickException(str(err)) from err

    alignment_attrs: dict[str, str] = {}
    if recording_alignment is not None and regressor_alignment is not None:
        alignment_attrs = {
            "session_start_time": recording_alignment[1]["session_start_time"],
            "behaviour_session": recording_alignment[1]["behaviour_session"],
        }

    if nuisance_regressor_paths:
        # Nuisance regressors are recorded on their own clock (e.g. a behavioural camera), so both series are
        # anchored to elapsed seconds since their own first sample before interpolating one onto the other -- see
        # `regr.elapsed_seconds`.
        target_timestamps = regr.elapsed_seconds(timestamps)

    for nuisance_path in nuisance_regressor_paths:
        click.echo(f"Loading nuisance regressors from {nuisance_path}...")
        nuisance, nuisance_labels, nuisance_timestamps = io.read_nuisance_regressors(nuisance_path)
        nuisance = regr.interpolate_regressors(nuisance, regr.elapsed_seconds(nuisance_timestamps), target_timestamps)
        if trial_idx is not None:
            nuisance = nuisance[trial_idx]
        regressors, labels = regr.append_nuisance_regressors(regressors, labels, nuisance, nuisance_labels)
        click.echo(f"Added nuisance regressors: {nuisance_labels}")

    outpath = out_dir + os.sep + session_id + f"_regression.{file_format}"

    trial_idx_used = False
    if trial_idx is not None:
        deltaf_series = deltaf_series[trial_idx]
        trial_idx_used = True

    with timer.Timer(message="Running regression"):
        if fast:
            coefs, r2, mse = regr.ridge_regression_fast(deltaf_series, regressors, alpha=alpha)
        else:
            coefs, r2, mse = regr.ridge_regression(deltaf_series, regressors)

        if file_format == "npz":
            outpath = io.write_npz(
                path=outpath,
                data={
                    "coefficients": coefs,
                    "r2": r2,
                    "mse": mse,
                    "labels": labels,
                    "trial_idx": trial_idx if trial_idx_used else [],
                    **alignment_attrs,
                },
            )
        elif file_format == "h5":
            outpath = io.write_h5(
                path=outpath,
                data={
                    "/coefficients": coefs,
                    "/r2": r2,
                    "/mse": mse,
                    "/labels": np.array(labels).astype("S"),
                    "/trial_idx": trial_idx if trial_idx_used else [],
                },
                attributes=alignment_attrs,
            )
    click.echo(f"Saved regression results at {outpath}")


def _metrics_options(command: Callable[..., None]) -> Callable[..., None]:
    """Options shared by `process metrics` and `process peri-event --with-metrics`.

    Args:
        command (Callable[..., None]): Command callback to decorate.

    Returns:
        Callable[..., None]: The callback with the options attached.
    """
    options = [
        click.option(
            "--response",
            type=(float, float),
            default=None,
            help="Response window START END, seconds relative to the event, end inclusive. Defaults to all"
            " post-event samples.",
        ),
        click.option(
            "--onset",
            type=click.Choice(pm.ONSET_METHODS),
            default="sd",
            show_default=True,
            help="Onset by threshold at a multiple of the baseline SD or a fraction of the amplitude, or by"
            " extrapolating a line fitted to the rise back to baseline.",
        ),
        click.option(
            "--onset-sd",
            type=float,
            default=2.0,
            show_default=True,
            help="Baseline SD multiple for --onset sd.",
        ),
        click.option(
            "--onset-fraction",
            type=float,
            default=0.2,
            show_default=True,
            help="Amplitude fraction for --onset peak.",
        ),
        click.option(
            "--onset-min-samples",
            type=int,
            default=3,
            show_default=True,
            help="Consecutive samples above threshold required for an onset.",
        ),
        click.option(
            "--extrapolate-range",
            type=(float, float),
            default=(0.2, 0.8),
            show_default=True,
            help="Amplitude fractions LOW HIGH bounding the rise fitted for --onset extrapolate.",
        ),
        click.option(
            "--smooth",
            "smoothing",
            type=int,
            default=1,
            show_default=True,
            help="Moving-average window in samples, odd, for peak, onset, decay and offset detection. 1 disables.",
        ),
        click.option(
            "--decay-fraction",
            type=float,
            default=0.5,
            show_default=True,
            help="Amplitude fraction the trace must fall to after the peak.",
        ),
    ]
    for option in reversed(options):
        command = option(command)
    return command


def _validate_metrics_options(smoothing: int, extrapolate_range: tuple[float, float]) -> None:
    """Reject metric options that `trial_metrics` would refuse.

    Args:
        smoothing (int): `--smooth` value.
        extrapolate_range (tuple[float, float]): `--extrapolate-range` value.

    Raises:
        click.BadParameter: If `smoothing` is not a positive odd integer or the range is not ordered within (0, 1).
    """
    if smoothing < 1 or smoothing % 2 == 0:
        msg = "must be a positive odd integer."
        raise click.BadParameter(msg, param_hint="--smooth")
    if not 0 < extrapolate_range[0] < extrapolate_range[1] < 1:
        msg = "must satisfy 0 < LOW < HIGH < 1."
        raise click.BadParameter(msg, param_hint="--extrapolate-range")


def _write_metrics(trial_table: pd.DataFrame, session_table: pd.DataFrame, out_dir: str, stem: str) -> None:
    """Write the per-trial and per-session metric tables and echo their paths.

    Args:
        trial_table (pd.DataFrame): Per-trial metrics, from `metrics.metrics_tables`.
        session_table (pd.DataFrame): Per-session metrics, from `metrics.metrics_tables`.
        out_dir (str): Output directory.
        stem (str): Output filename stem, without `_perievent`.
    """
    trial_path = out_dir + os.sep + stem + "_metrics.csv"
    session_path = out_dir + os.sep + stem + "_metrics-session.csv"
    trial_table.to_csv(trial_path, index=False)
    session_table.to_csv(session_path, index=False)
    click.echo(f"Saved per-trial metrics at {trial_path}")
    click.echo(f"Saved per-session metrics at {session_path}")


@process_cmd.command("peri-event")
@click.argument(
    "recording_path",
    type=click.Path(exists=True, dir_okay=False),
)
@click.argument(
    "trials_path",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "-o",
    "--out_dir",
    type=click.Path(dir_okay=True),
    default="./",
    help="Output directory for peri-event windows.",
)
@click.option(
    "--event",
    type=click.Choice(pev.EVENTS),
    default="cue_onset",
    show_default=True,
    help="Behavioural event to align windows to.",
)
@click.option(
    "--pre",
    type=float,
    default=1.0,
    show_default=True,
    help="Seconds before the event.",
)
@click.option(
    "--post",
    type=float,
    default=3.0,
    show_default=True,
    help="Seconds after the event.",
)
@click.option(
    "--fs",
    type=float,
    default=25.0,
    show_default=True,
    help="Sampling rate of the output windows, in Hz.",
)
@click.option(
    "--method",
    type=click.Choice(["interp", "nearest"]),
    default="interp",
    show_default=True,
    help="Linear interpolation, or the nearest recorded sample.",
)
@click.option(
    "--baseline",
    type=(float, float),
    default=None,
    help="Baseline window START END, seconds relative to the event; its mean is subtracted per trial.",
)
@click.option(
    "--with-metrics",
    is_flag=True,
    default=False,
    help="Also write the response metrics of `process metrics`, joined with TRIALS_PATH. CSV input only.",
)
@_metrics_options
def perievent_cmd(
    recording_path: str,
    trials_path: str,
    out_dir: str,
    event: str,
    pre: float,
    post: float,
    fs: float,
    method: str,
    baseline: tuple[float, float] | None,
    with_metrics: bool,
    response: tuple[float, float] | None,
    onset: str,
    onset_sd: float,
    onset_fraction: float,
    onset_min_samples: int,
    extrapolate_range: tuple[float, float],
    smoothing: int,
    decay_fraction: float,
) -> None:
    """Extract per-trial windows around a behavioural event from a behaviour-aligned recording.

    RECORDING_PATH is an HDF5 recording with /F and /timestamps_aligned, or a _regions.csv with a time_aligned
    column. TRIALS_PATH is the *_trials.csv written by visiomode-analysis session. HDF5 input gives HDF5 output;
    CSV input gives long-format CSV output. --with-metrics also writes the tables of `process metrics` from the
    CSV output, using --baseline as the metrics baseline window.
    """  # noqa: DOC501
    import pandas as pd

    is_csv = recording_path.endswith(".csv")
    if with_metrics:
        if not is_csv:
            msg = "needs a _regions.csv recording; metrics are not computed for HDF5 windows."
            raise click.BadParameter(msg, param_hint="--with-metrics")
        _validate_metrics_options(smoothing, extrapolate_range)

    if not Path(out_dir).exists():
        click.echo(f"Creating output directory {out_dir}...")
        Path(out_dir).mkdir(parents=True)

    click.echo(f"Loading trials from {trials_path}...")
    trials = pd.read_csv(trials_path)
    events, trial_index = pev.event_times(trials, event)
    grid = pev.window_grid(pre, post, fs)

    click.echo(f"Loading recording from {recording_path}...")
    session_id = Path(recording_path).stem
    outpath = out_dir + os.sep + session_id + f"_event-{pev.event_name(event)}_perievent." + ("csv" if is_csv else "h5")

    with timer.Timer(message="Extracting peri-event windows"):
        if is_csv:
            kept, windows = _perievent_csv(recording_path, outpath, events, trial_index, grid, method, baseline)
        else:
            attrs = {
                "event": event,
                "pre": pre,
                "post": post,
                "fs": fs,
                "method": method,
                "baseline": np.array(baseline if baseline is not None else [], dtype=np.float64),
                "trials_path": str(trials_path),
            }
            kept = _perievent_h5(recording_path, outpath, events, trial_index, grid, method, baseline, attrs)

    n_kept = int(kept.sum())
    click.echo(f"Kept {n_kept} trials, dropped {len(trials) - n_kept}.")
    click.echo(f"Saved peri-event windows at {outpath}")

    if with_metrics and n_kept == 0:
        click.echo("No trials kept, skipping metrics.")
    elif with_metrics:
        with timer.Timer(message="Extracting metrics"):
            tables = pm.metrics_tables(
                windows,
                baseline=baseline,
                response=response,
                trials=trials,
                onset=onset,
                onset_sd=onset_sd,
                onset_fraction=onset_fraction,
                onset_min_samples=onset_min_samples,
                extrapolate_range=extrapolate_range,
                decay_fraction=decay_fraction,
                smoothing=smoothing,
            )
        _write_metrics(*tables, out_dir, Path(outpath).stem.removesuffix("_perievent"))


def _perievent_h5(
    recording_path: str,
    outpath: str,
    events: np.ndarray,
    trial_index: np.ndarray,
    grid: np.ndarray,
    method: str,
    baseline: tuple[float, float] | None,
    attrs: dict,
) -> np.ndarray:
    """Write peri-event windows from an HDF5 recording, one trial at a time.

    Args:
        recording_path (str): HDF5 recording with `/F` and `/timestamps_aligned`.
        outpath (str): Output HDF5 path.
        events (np.ndarray): Event times, seconds from behaviour start.
        trial_index (np.ndarray): Trials CSV row index per event.
        grid (np.ndarray): Sample times relative to the event.
        method (str): `interp` or `nearest`.
        baseline (tuple[float, float] | None): Baseline window, or None.
        attrs (dict): Attributes to write on `/traces`.

    Returns:
        np.ndarray: Boolean mask over `events` marking the kept trials.

    Raises:
        click.ClickException: If the recording lacks `/F` or `/timestamps_aligned`, or their lengths differ.
    """
    aligned = io.read_timestamps_aligned(recording_path)
    if aligned is None:
        msg = f"{recording_path} has no /timestamps_aligned dataset; run `mesoscopy align` first."
        raise click.ClickException(msg)
    time_aligned, aligned_attrs = aligned

    with h5py.File(recording_path, "r") as f:
        if "/F" not in f:
            msg = f"{recording_path} has no /F dataset."
            raise click.ClickException(msg)
        deltaf_series = f["/F"][:]

    if len(time_aligned) != deltaf_series.shape[0]:
        msg = f"{recording_path} has {len(time_aligned)} aligned timestamps but {deltaf_series.shape[0]} frames."
        raise click.ClickException(msg)

    kept = (events + grid[0] >= time_aligned[0]) & (events + grid[-1] <= time_aligned[-1])
    frame_shape = deltaf_series.shape[1:]
    n_kept = int(kept.sum())

    with h5py.File(outpath, "w") as f:
        traces = f.create_dataset(
            "/traces",
            shape=(n_kept, len(grid), *frame_shape),
            chunks=(1, len(grid), *frame_shape),
            dtype=np.float32,
            compression="lzf",
        )
        for i, event_time in enumerate(events[kept]):
            trace, _ = pev.extract(deltaf_series, time_aligned, np.array([event_time]), grid, method=method)
            if baseline is not None:
                trace = pev.apply_baseline(trace, grid, *baseline)
            traces[i] = trace[0]

        f.create_dataset("/time", data=grid.astype(np.float64))
        f.create_dataset("/trial_index", data=trial_index[kept])
        f.create_dataset("/event_time", data=events[kept].astype(np.float64))
        traces.attrs.update(
            {
                **attrs,
                "session_start_time": aligned_attrs["session_start_time"],
                "behaviour_session": aligned_attrs["behaviour_session"],
            }
        )

    return kept


def _perievent_csv(
    recording_path: str,
    outpath: str,
    events: np.ndarray,
    trial_index: np.ndarray,
    grid: np.ndarray,
    method: str,
    baseline: tuple[float, float] | None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Write peri-event windows from a long-format regions CSV.

    Args:
        recording_path (str): Regions CSV with a `time_aligned` column.
        outpath (str): Output CSV path.
        events (np.ndarray): Event times, seconds from behaviour start.
        trial_index (np.ndarray): Trials CSV row index per event.
        grid (np.ndarray): Sample times relative to the event.
        method (str): `interp` or `nearest`.
        baseline (tuple[float, float] | None): Baseline window, or None.

    Returns:
        tuple[np.ndarray, pd.DataFrame]: Boolean mask over `events` marking the kept trials, and the written table.

    Raises:
        click.ClickException: If the CSV has no `time_aligned` column.
    """
    import pandas as pd

    regions = pd.read_csv(recording_path)
    if "time_aligned" not in regions.columns:
        msg = f"{recording_path} has no time_aligned column; run `mesoscopy align` before `process regions`."
        raise click.ClickException(msg)

    # Long to (n_frames, n_regions), keeping the input's region order.
    wide = regions.pivot(index="time_aligned", columns="region", values="F").sort_index()
    wide = wide[regions["region"].unique()]
    time_aligned = wide.index.to_numpy(dtype=np.float64)

    traces, kept = pev.extract(wide.to_numpy(), time_aligned, events, grid, method=method, dtype=np.float64)
    if baseline is not None:
        traces = pev.apply_baseline(traces, grid, *baseline)

    n_kept, n_samples, n_regions = traces.shape
    out = pd.DataFrame(
        {
            "trial_index": np.repeat(trial_index[kept], n_samples * n_regions),
            "event_time": np.repeat(events[kept], n_samples * n_regions),
            "time": np.tile(np.repeat(grid, n_regions), n_kept),
            "region": np.tile(wide.columns.to_numpy(), n_kept * n_samples),
            "F": traces.reshape(-1),
        }
    )
    out.to_csv(outpath, index=False)

    return kept, out


@process_cmd.command("metrics")
@click.argument(
    "path",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "-o",
    "--out_dir",
    type=click.Path(dir_okay=True),
    default="./",
    help="Output directory for metrics tables.",
)
@click.option(
    "--baseline",
    type=(float, float),
    default=None,
    help="Baseline window START END, seconds relative to the event, end exclusive. Defaults to all pre-event samples.",
)
@click.option(
    "-t",
    "--trials",
    "trials_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Trials CSV to join onto the per-trial table by trial_index.",
)
@_metrics_options
def metrics_cmd(
    path: str,
    out_dir: str,
    baseline: tuple[float, float] | None,
    trials_path: str | None,
    response: tuple[float, float] | None,
    onset: str,
    onset_sd: float,
    onset_fraction: float,
    onset_min_samples: int,
    extrapolate_range: tuple[float, float],
    smoothing: int,
    decay_fraction: float,
) -> None:
    """Extract per-trial response metrics and per-session variability from peri-event traces.

    PATH is the long-format *_perievent.csv written by `process peri-event`. Writes <stem>_metrics.csv with onset
    time, peak time, amplitude, area under the curve, decay time, offset time and duration per trial per region,
    and <stem>_metrics-session.csv with the mean, SD and CV of each across trials per region, plus the mean
    pairwise trial-trace correlation.
    """  # noqa: DOC501
    import pandas as pd

    _validate_metrics_options(smoothing, extrapolate_range)

    if not Path(out_dir).exists():
        click.echo(f"Creating output directory {out_dir}...")
        Path(out_dir).mkdir(parents=True)

    click.echo(f"Loading peri-event traces from {path}...")
    perievent = pd.read_csv(path)
    if perievent.empty:
        msg = f"{path} has no rows; `process peri-event` kept no trials."
        raise click.ClickException(msg)
    trials = None
    if trials_path is not None:
        click.echo(f"Loading trials from {trials_path}...")
        trials = pd.read_csv(trials_path)

    with timer.Timer(message="Extracting metrics"):
        try:
            tables = pm.metrics_tables(
                perievent,
                baseline=baseline,
                response=response,
                trials=trials,
                onset=onset,
                onset_sd=onset_sd,
                onset_fraction=onset_fraction,
                onset_min_samples=onset_min_samples,
                extrapolate_range=extrapolate_range,
                decay_fraction=decay_fraction,
                smoothing=smoothing,
            )
        except ValueError as error:
            msg = f"{path}: {error} Expected the columns written by `process peri-event`."
            raise click.ClickException(msg) from error

    _write_metrics(*tables, out_dir, Path(path).stem.removesuffix("_perievent"))
