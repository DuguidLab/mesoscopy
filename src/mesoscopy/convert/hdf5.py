#  Copyright (c) 2022-2025 Constantinos Eleftheriou <Constantinos.Eleftheriou@ed.ac.uk>.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this
#   software and associated documentation files (the "Software"), to deal in the
#   Software without restriction, including without limitation the rights to use, copy,
#   modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#   and to permit persons to whom the Software is furnished to do so, subject to the
#   following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies
#  or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#  HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
#  IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
#  IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

"""HDF5 conversion utilities."""

import os
from datetime import datetime

import click
from pynwb import NWBFile
from pynwb.file import Subject
from pynwb.ophys import OnePhotonSeries
from pynwb.ophys import OpticalChannel

import mesoscopy.convert.metadata as mtd
from mesoscopy import io


def to_nwb(
    input_path: str,
    out_dir: str,
    link_only: bool = False,
    frames_group: str = "frames",
    timestamps_group: str = "timestamps",
    meta_path: str | None = None,
    subject_id: str | None = None,
    sex: str | None = None,
    genotype: str | None = None,
    species: str | None = None,
    strain: str | None = None,
    dob: str | None = None,
    session_description: str | None = None,
    experimenter: str | None = None,
    lab: str | None = None,
    institution: str | None = None,
    **kwargs,
) -> str:
    """Convert a mesoscale recording session from HDF5 to NWB. Optionally add session metadata.

    Metadata can be parsed from a compatible YAML or JSON file. Individual metadata arguments passed to this function (e.g. subject ID) will take precedence over the contents of a metadata file, if both are provided.

    Args:
        input_path (str): Path to raw HDF5 file.
        out_dir (str): Output directory.
        link_only (bool): Create links to HDF5 datasets instead of copying them over. Defaults to False.
        frames_group (str): HDF group path under which frame data is stored. Defaults to "frames".
        timestamps_group (str): HDF group path under which timestamp data is stored. Defaults to "timestamps".
        meta_path (typing.Optional[str], optional): Path to metadata file in JSON or YAML format. Defaults to None.
        subject_id (typing.Optional[str], optional): Subject ID. Defaults to None.
        sex (typing.Optional[str], optional): Subject sex. Defaults to None.
        genotype (typing.Optional[str], optional): Subject genotype. Defaults to None.
        species (typing.Optional[str], optional): Subject species. Defaults to None.
        strain (typing.Optional[str], optional): Subject strain. Defaults to None.
        dob (typing.Optional[str], optional): Subject date of birth. Defaults to None.
        session_description (typing.Optional[str], optional): Session description. Defaults to None.
        experimenter (typing.Optional[str], optional): Experimenter name. Defaults to None.
        lab (typing.Optional[str], optional): Laboratory experiment was done in. Defaults to None.
        institution (typing.Optional[str], optional): Institution experiment was done in. Defaults to None

    Raises:
        ValueError: Raised if declared frames or timestamp groups don't exist in the HDF5 file.

    Returns:
        str: Path to new NWB file.
    """
    h5file = io.read_h5(input_path)

    # Validate frame and timestamp groups actually exist.
    if (frames_group not in h5file) or (timestamps_group not in h5file):
        raise ValueError("Could not find frame or timestamp data, did you use the right group names?")

    session_start_time = datetime.fromisoformat(
        h5file[f"/{timestamps_group}"][0].decode("utf-8")
    )  # Use first timestamp as session start

    session_identifier = input_path.split(os.sep)[-1].replace(".h5", "")
    subject_meta = mtd.DEFAULT_METADATA.copy()

    if meta_path:
        if meta_path.endswith(".yaml") or meta_path.endswith(".yml"):
            subject_meta = mtd.read_yaml(meta_path)
        elif meta_path.endswith(".json"):
            subject_meta = mtd.read_json(meta_path)
        else:
            click.echo("WARNING - Invalid metadata file provided, skipping...")

    # Infer subject_id if not provided.
    if not subject_id and not meta_path:
        try:
            # Check if NWB-style file-naming
            if "sub-" in input_path.split(os.sep)[-1]:
                subject_meta["subject_id"] = (
                    input_path.split(os.sep)[-1].split("_")[0].replace("sub-", "").replace(".h5", "")
                )
        except IndexError:
            click.echo(
                "WARNING - Subject ID not provided and could not be inferred, using a default placeholder value. This might get confusing!"
            )

    if subject_id:
        subject_meta["subject_id"] = subject_id
    if sex:
        subject_meta["sex"] = sex
    if genotype:
        subject_meta["genotype"] = genotype
    if species:
        subject_meta["species"] = species
    if strain:
        subject_meta["strain"] = strain
    if dob:
        subject_meta["dob"] = dob
    if session_description:
        subject_meta["session_description"] = session_description
    if experimenter:
        subject_meta["experimenter"] = experimenter
    if lab:
        subject_meta["lab"] = lab
    if institution:
        subject_meta["institution"] = institution

    nwbfile = NWBFile(
        session_description=subject_meta.get("session_description"),
        identifier=session_identifier,
        session_start_time=session_start_time,  # Use first timestamp as session start
        experimenter=subject_meta.get("experimenter"),
        lab=subject_meta.get("lab"),
        institution=subject_meta.get("institution"),
    )

    nwbfile.subject = Subject(
        subject_id=subject_meta.get("subject_id"),
        species=subject_meta.get("species"),
        strain=subject_meta.get("strain"),
        sex=subject_meta.get("sex"),
        date_of_birth=datetime.fromisoformat(subject_meta.get("dob", "1900-01-01")),
        genotype=subject_meta.get("genotype"),
    )

    device = nwbfile.create_device(
        name="Mesoscope",
        description="Single-photon widefield imaging scope.",
        manufacturer="",
    )

    optical_channel = OpticalChannel(
        name="DualAcquisitionChannel",
        description="Acquisition channel for GCaMP6s excited at 470 and 405 nm.",
        emission_lambda=500.0,
    )

    imaging_plane = nwbfile.create_imaging_plane(
        name="DualChannelImagingPlane",
        optical_channel=optical_channel,
        # imaging_rate=50.,
        description="Dual excitation through the skull at 470 and 405 nm (25Hz per LED).",
        device=device,
        excitation_lambda=470.0,
        indicator="GCaMP6s",
        location="dorsal cortex",
        grid_spacing=[20.0, 20.0],
        grid_spacing_unit="micrometers",
    )

    source_data = h5file[f"/{frames_group}"]
    source_timestamps = h5file[f"/{timestamps_group}"]

    # recalculate timestamps to relative from session start
    timestamps = [
        (datetime.fromisoformat(timestamp.decode("utf-8")) - session_start_time).total_seconds()
        for timestamp in source_timestamps
    ]

    imaging_series = OnePhotonSeries(
        name="DualChannelImagingSeries",
        imaging_plane=imaging_plane,
        data=source_data,
        timestamps=timestamps,
        unit="pixel_intensity",
        binning=2,
        power=10.0,
        exposure_time=0.01,
        pmt_gain=10.0,
        description="Widefield cortical images acquired at alternating 470 and 405 nm excitation.",
        comments="This imaging series requires channel separation and haemodynamic subtraction before use.",
    )

    nwbfile.add_acquisition(imaging_series)

    nwb_path = out_dir + os.sep + session_identifier + ".nwb"
    io.write_nwb(
        path=nwb_path,
        nwbfile=nwbfile,
        link_data=link_only,
    )

    return nwb_path
