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

"""File conversion CLI."""

import typing

import click

from mesoscopy import docs_epilog


@click.group("convert")
def convert_cmd() -> None:
    """Convert raw recordings to NWB or HDF5."""


@convert_cmd.command("h5", epilog=docs_epilog("typical-workflow/"))
@click.argument(
    "input_path",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--out-dir",
    type=click.Path(),
    default="./",
    help="Output directory, created if missing.",
)
@click.option(
    "-l",
    "--link-only",
    is_flag=True,
    show_default=True,
    default=False,
    help="Link the raw frames instead of copying them. The NWB file then needs the raw HDF5 file to stay in place.",
)
@click.option(
    "-f", "--frames-group", type=str, default="frames", show_default=True, help="HDF5 dataset holding the frames."
)
@click.option(
    "-t",
    "--timestamps-group",
    type=str,
    default="timestamps",
    show_default=True,
    help="HDF5 dataset holding the timestamps.",
)
@click.option(
    "-m",
    "--meta",
    "meta_path",
    type=click.Path(exists=True),
    help="Subject and session metadata file in YAML or JSON format.",
)
@click.option("--subject-id", type=str, help="Subject identifier.")
@click.option("--sex", type=click.Choice(["M", "F"], case_sensitive=False), help="Subject sex.")
@click.option("--genotype", type=str, help="Subject genotype.")
@click.option("--species", type=str, help="Subject species.")
@click.option("--strain", type=str, help="Subject strain.")
@click.option("--dob", type=str, help="Subject date of birth as YYYY-MM-DD.")
@click.option("--description", "session_description", type=str, help="Session description.")
@click.option("--experimenter", type=str, help="Experimenter name.")
@click.option("--lab", type=str, help="Lab the session was recorded in.")
@click.option("--institution", type=str, help="Institution the session was recorded in.")
def convert_h5_cmd(**kwargs: typing.Any) -> None:
    """Convert a raw HDF5 recording to NWB.

    INPUT_PATH is a raw dual-channel HDF5 recording. Fill in the subject and session metadata with --meta or the
    metadata options. Writes an NWB file named after INPUT_PATH.
    """
    import mesoscopy.convert.hdf5 as conv_h5

    conv_h5.to_nwb(**kwargs)


@convert_cmd.command("video")
@click.argument(
    "input_path",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--out-dir",
    type=click.Path(),
    default="./",
    help="Output directory, created if missing.",
)
@click.option(
    "-t",
    "--ts-path",
    type=click.Path(exists=True),
    help="Frame timestamps file, plain text or delimited.",
)
@click.option(
    "-d",
    "--ts-delimiter",
    type=click.Choice([",", "\t", " "], case_sensitive=False),
    default=",",
    help="Delimiter of the timestamps file.",
)
@click.option(
    "-c",
    "--ts-column",
    type=int,
    help="Column of the timestamps file to read, needed when it has more than one.",
)
@click.option(
    "--ts-has-header",
    type=bool,
    is_flag=True,
    default=False,
    help="The timestamps file has a header row.",
)
@click.option(
    "-l",
    "--hdf5-only",
    is_flag=True,
    show_default=True,
    default=False,
    help="Write the HDF5 file only, without converting it to NWB.",
)
@click.option(
    "-m",
    "--meta",
    "meta_path",
    type=click.Path(exists=True),
    help="Subject and session metadata file in YAML or JSON format.",
)
@click.option("--subject-id", type=str, help="Subject identifier.")
@click.option("--sex", type=click.Choice(["M", "F"], case_sensitive=False), help="Subject sex.")
@click.option("--genotype", type=str, help="Subject genotype.")
@click.option("--species", type=str, help="Subject species.")
@click.option("--strain", type=str, help="Subject strain.")
@click.option("--dob", type=str, help="Subject date of birth as YYYY-MM-DD.")
@click.option("--description", "session_description", type=str, help="Session description.")
@click.option("--experimenter", type=str, help="Experimenter name.")
@click.option("--lab", type=str, help="Lab the session was recorded in.")
@click.option("--institution", type=str, help="Institution the session was recorded in.")
def convert_video_cmd(**kwargs: typing.Any) -> None:
    """Convert a raw video recording to NWB.

    INPUT_PATH is an AVI or MP4 recording and --ts-path its frame timestamps file. Fill in the subject and session
    metadata with --meta or the metadata options. Writes an HDF5 file and an NWB file named after INPUT_PATH.
    """
    import mesoscopy.convert.video as conv_vid

    if not kwargs.get("ts_path"):
        click.echo(
            "⚠️ WARNING - No timestamps file provided, using autogenerated timestamps. This might mess things up "
            "downstream."
        )
    if kwargs.pop("hdf5_only"):
        conv_vid.to_hdf5(**kwargs)
    else:
        conv_vid.to_nwb(**kwargs)
