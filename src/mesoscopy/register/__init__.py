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
import json
import os
import pathlib

import click
import h5py
import numpy as np
from dask import array as da
from pynwb import NWBFile
from pynwb import TimeSeries
from pynwb.image import ImageSeries
from pynwb.ophys import CorrectedImageStack

import mesoscopy.preprocess.compute as preproc_compute
import mesoscopy.register.landmarks_gui as reg_gui
import mesoscopy.register.transform as trf
import mesoscopy.resources as res
from mesoscopy import io
from mesoscopy import timer as timer


@click.group("register")
def register_cmd() -> None:
    """Register recordings to a template."""


@register_cmd.command("label")
@click.argument(
    "path",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--out_dir",
    type=click.Path(dir_okay=True),
    default="./",
    help="Output directory for registered recording.",
)
@click.option(
    "-t",
    "--template-points",
    type=click.Path(dir_okay=False),
    help="Path to template landmark points in CSV or Fiji XML points format",
)
@click.option(
    "--session-id",
    type=str,
    help="Session ID for the recording.",
)
def label_cmd(path, out_dir, template_points, session_id) -> dict:
    """Mark landmarks on a recording for registration to a template using the landmarks GUI.

    Args:
        path (str): Path to preprocessed HDF5 file or NWB file.
        out_dir (str): Output directory for registration landmarks file.
        template_points (str): Path to template landmark points in CSV or Fiji XML points format.
        session_id (str): Session ID for the recording.

    Returns:
        dict: Dictionary with the landmarks and their x-y coordinates.
              Dictionary keys are landmark names, while x-y coordinates are stored as an (x, y)
              tuple, i.e. (column, row).
    """
    click.echo("Loading imaging data...")
    nwb = bool(path.endswith(".nwb"))

    if not session_id:
        session_id = session_id_from_path(path)

    maxip = None
    isosb_maxip = None

    # Prefer the projections written by preprocessing: they are in the same pixel space as the dF/F
    # series, which is the space the landmarks have to be marked in. An NWB file links its dF/F
    # series to the preprocessed HDF5 file, so the projections can be read from there.
    source = linked_preprocessed_path(path) if nwb else path
    if source and pathlib.Path(source).exists():
        maxip, isosb_maxip = load_maxips(source)

    if maxip is None:
        # Fall back to projecting the dF/F series itself. It is a poorer anatomical image than the
        # gcamp projection, but it is guaranteed to be in the same pixel space as the data being
        # registered - projecting the raw frames instead would be off by the preprocessing crop and
        # binning factor, silently scaling the transform.
        click.echo("⚠️ No preprocessing maximum intensity projection found, projecting the ∆F/F series instead.")
        with timer.Timer("Generating maximum intensity projection"):
            _, deltaf_series, _ = io.load_deltaf(path, nwb=nwb)
            maxip = preproc_compute.projections(da.from_array(deltaf_series))["maxip"]

    click.echo("Loading template landmarks...")
    template_landmarks = res.get_default_landmarks()
    template_shape = res.get_atlas()[0].shape
    if template_points:
        template_landmarks = io.read_points(template_points)
        # The image a user-supplied template was marked on is unknown, so seed points can't be scaled.
        template_shape = None

    click.echo("Launching landmark identification GUI...")
    recording_landmarks = reg_gui.mark_landmarks(maxip, isosb_maxip, template_landmarks, template_shape=template_shape)

    click.echo("Saving recording landmarks...")
    outpath = out_dir + os.sep + session_id + "_landmarks.csv"
    io.write_points(outpath, recording_landmarks)

    click.echo(f"Recording landmarks saved at {outpath}.")

    return recording_landmarks


@register_cmd.command("landmarks")
@click.argument(
    "path",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--out_dir",
    type=click.Path(dir_okay=True),
    default="./",
    help="Output directory for registered recording.",
)
@click.option(
    "-r",
    "--recording-points",
    type=click.Path(dir_okay=False),
    help="Path to recording landmark points in Fiji XML points format",
)
@click.option(
    "-t",
    "--template-points",
    type=click.Path(dir_okay=False),
    help="Path to template landmark points in Fiji XML points format",
)
@click.option(
    "--output-width",
    type=int,
    default=None,
    help="Width of the registered frames. Defaults to the width of the Allen CCF template.",
)
@click.option(
    "--output-height",
    type=int,
    default=None,
    help="Height of the registered frames. Defaults to the height of the Allen CCF template.",
)
def landmarks_cmd(
    path: str,
    out_dir: str,
    recording_points: str,
    template_points: str,
    output_width: int | None = None,
    output_height: int | None = None,
) -> str:
    """Register a recording to a template based on defined landmarks.

    Args:
        path (str): Path to preprocessed recording HDF5 or NWB file.
        out_dir (str): Output directory for registered recording.
        recording_points (str, optional): Path to recording landmark points in CSV or Fiji XML points format.
        template_points (str, optional): Path to template landmark points in CSV or Fiji XML points format.
        output_width (int, optional): Width of the registered frames. Defaults to the Allen CCF template width.
        output_height (int, optional): Height of the registered frames. Defaults to the Allen CCF template height.

    Returns:
        str: Path to the registered recording file.

    Raises:
        ValueError: If the path to recording landmarks cannot be inferred.
    """
    click.echo(f"Registering recording {path} to template.")

    os.makedirs(out_dir, exist_ok=True)

    click.echo("Loading imaging data...")

    # Determine whether we're working with an NWB file
    nwb = True if path.endswith(".nwb") else False

    session_id, deltaf_series, timestamps = io.load_deltaf(path, nwb)

    click.echo("Loading landmarks...")
    template_landmarks = res.get_default_landmarks()
    if template_points:
        template_landmarks = io.read_points(template_points)

    if not recording_points:
        candidates = landmarks_path_candidates(path, out_dir)
        found = next((candidate for candidate in candidates if candidate.exists()), None)
        if found is None:
            searched = "\n  ".join(str(candidate) for candidate in candidates)
            msg = (
                "Path to recording landmarks could not be inferred. Searched:\n  "
                f"{searched}\nPlease supply a recording landmarks file with -r/--recording-points."
            )
            raise ValueError(msg)
        recording_points = str(found)
        click.echo(f"Using recording landmarks at {recording_points}")
    recording_landmarks = io.read_points(recording_points)

    # Registered frames land in template space, so default their shape to that of the CCF atlas.
    output_shape = None
    if output_width or output_height:
        atlas_height, atlas_width = res.get_atlas()[0].shape
        output_shape = (output_height or atlas_height, output_width or atlas_width)

    warped, tform = trf.landmarks_affine(
        deltaf_series,
        recording_landmarks,
        template_landmarks,
        output_shape=output_shape,
    )

    # Save warped frames and timestamps
    outpath = out_dir + os.sep + session_id + "_registered.h5"
    outpath = io.write_h5(
        path=outpath,
        data={
            "/F": warped,
            "/timestamps": timestamps,
            "/tform": tform.params,
            "/qa/recording_landmarks": np.array(list(recording_landmarks.values())),
            "/qa/template_landmarks": np.array(list(template_landmarks.values())),
            "/qa/registered_landmarks": tform.inverse(np.array(list(recording_landmarks.values()))),
        },
    )
    click.echo(f"Saved registered frames at {outpath}")

    if nwb:
        click.echo("Updating NWB file...")
        update_nwb(path, outpath, tform.params)
        click.echo(f"Updated NWB file at {path}")

    return outpath


def session_id_from_path(path: str) -> str:
    """Derive a session identifier from a recording path.

    Args:
        path (str): Path to a recording file.

    Returns:
        str: The file name without its extension or the "_preprocessed" suffix.
    """
    return pathlib.Path(path).stem.replace("_preprocessed", "")


def landmarks_path_candidates(path: str, out_dir: str | None = None) -> list[pathlib.Path]:
    """List the paths a recording's landmarks file may have been written to.

    ``register label`` names its output after the session ID, which drops the "_preprocessed"
    suffix, so the landmarks file rarely sits at ``<recording>_landmarks.csv``.

    Args:
        path (str): Path to the recording file.
        out_dir (str, optional): Output directory the landmarks may have been written to.

    Returns:
        list[pathlib.Path]: Candidate landmark file paths, in search order, without duplicates.
    """
    recording = pathlib.Path(path)
    names = [f"{session_id_from_path(path)}_landmarks.csv", f"{recording.stem}_landmarks.csv"]
    directories = [recording.parent]
    if out_dir:
        directories.append(pathlib.Path(out_dir))

    candidates = [directory / name for directory in directories for name in names]
    return list(dict.fromkeys(candidates))


def load_maxips(path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load maximum intensity projections from a preprocessed HDF5 file.

    Args:
        path (str): Path to the preprocessed HDF5 file.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]: Maximum intensity projection for the gcamp and
            isosb channels, or (None, None) if the file holds no gcamp projection.
    """
    if not h5py.is_hdf5(path):
        return None, None

    with h5py.File(path, "r") as f_preproc:
        if "/qa/gcamp_maxip_projection" not in f_preproc:
            return None, None

        gcamp_maxip_projection = np.array(f_preproc["/qa/gcamp_maxip_projection"])
        isosb_maxip_projection = None
        if "/qa/isosb_maxip_projection" in f_preproc:
            isosb_maxip_projection = np.array(f_preproc["/qa/isosb_maxip_projection"])

    return gcamp_maxip_projection, isosb_maxip_projection


def linked_preprocessed_path(nwb_path: str) -> str | None:
    """Resolve the preprocessed HDF5 file that an NWB file links its dF/F series to.

    Args:
        nwb_path (str): Path to the NWB file.

    Returns:
        str | None: Path to the linked HDF5 file, or None if the dF/F series is not an external link.
    """
    with h5py.File(nwb_path, "r") as f:
        link = f.get("processing/ophys/DeltaFSeries/data", getlink=True)

    if not isinstance(link, h5py.ExternalLink):
        return None

    # External link targets are stored relative to the NWB file (absolute paths pass through).
    return str(pathlib.Path(nwb_path).parent / link.filename)


def update_nwb(nwb_path: str, h5_path: str, tform_params: np.ndarray) -> NWBFile:
    """Update an NWB file with registered imaging data stored in an HDF5 file.

    Creates a link between the NWB file and the HDF5 file. See https://pynwb.readthedocs.io/en/stable/tutorials/advanced_io/linking_data.html.

    The registration is a single global affine, so the same 3x3 matrix is stored for every frame of
    the xy_translation series, giving it a shape of (n_timestamps, 3, 3).

    Args:
        nwb_path (str): Path to the NWB file.
        h5_path (str): Path to the HDF5 file containing the registered images.
        tform_params (np.ndarray): Affine transformation matrix, as a 3x3 array.

    Returns:
        NWBFile: The updated NWB file object. Note that its link to the HDF5 file is closed on
            return, so the registered image data is only readable by re-opening nwb_path.
    """
    nwbfile, nwbio = io.read_nwb(nwb_path, return_io=True)

    with h5py.File(h5_path, "r") as f:
        try:
            ophys_module = nwbfile.create_processing_module(
                name="ophys", description="optical physiology processed data"
            )
        except ValueError:
            click.echo("Processing module already exists...")
            ophys_module = nwbfile.processing["ophys"]

        registered_series = ImageSeries(
            name="corrected",
            data=f["/F"],
            timestamps=f["/timestamps"],
            unit="df/f",
            description="dF/F widefield cortical imaging series.",
            comments="This is the haemodynamic corrected series registered to the Allen Brain Atlas CCFv3.",
        )

        xy_translation = TimeSeries(
            name="xy_translation",
            data=np.tile(tform_params, (len(f["/timestamps"]), 1, 1)),
            unit="pixels",
            timestamps=f["/timestamps"],
            description="Affine transformation parameters for image registration to the ABA CCFv3.",
        )

        corrected_image_stack = CorrectedImageStack(
            name="CCFRegisteredSeries",
            corrected=registered_series,
            original=nwbfile.acquisition["DualChannelImagingSeries"],
            xy_translation=xy_translation,
        )

        ophys_module.add(corrected_image_stack)

        io.write_nwb(nwb_path, nwbfile, io=nwbio)

    return nwbfile
