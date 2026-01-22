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

import mesoscopy.convert.hdf5 as conv_h5
import mesoscopy.convert.video as conv_vid


@click.group("convert")
def convert_cmd(): ...


@convert_cmd.command("h5")
@click.argument(
    "input_path",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--out-dir",
    type=click.Path(),
    default="./",
    help="Output directory for converted file, defaults to current working directory. Will be created if it doesn't exist.",
)
@click.option(
    "-l",
    "--link-only",
    is_flag=True,
    show_default=True,
    default=False,
    help="Create a link to the HDF5 dataset instead of copying it over. If this is enabled, you MUST keep the raw HDF5 file - if you delete it, the data will also be gone.",
)
@click.option(
    "-f", "--frames-group", type=str, default="frames", help="HDF group path under which frame data is stored."
)
@click.option(
    "-t",
    "--timestamps-group",
    type=str,
    default="timestamps",
    help="HDF group path under which timestamp data is stored.",
)
@click.option(
    "-m",
    "--meta",
    "meta_path",
    type=click.Path(exists=True),
    help="Path to animal metadata file. Must be YAML or JSON format.",
)
@click.option("--subject-id", type=str, help="Metadata field - subject identifier.")
@click.option("--sex", type=click.Choice(["M", "F"], case_sensitive=False), help="Metadata field - subject sex.")
@click.option("--genotype", type=str, help="Metadata field - subject genotype.")
@click.option("--species", type=str, help="Metadata field - subject species.")
@click.option("--strain", type=str, help="Metadata field - subject strain.")
@click.option("--dob", type=str, help="Metadata field - subject date of birth in YYYY-MM-DD format (i.e. 1900-01-31).")
@click.option("--description", "session_description", type=str, help="Metadata field - session description.")
@click.option("--experimenter", type=str, help="Metadata field - experimenter name.")
@click.option("--lab", type=str, help="Metadata field - lab experiment was done in.")
@click.option("--institution", type=str, help="Metadata field - institution experiment was done in.")
def convert_h5_cmd(**kwargs: typing.Any) -> None:
    """Convert a raw mesoscale calcium recording session from HDF5 to an NWB file compatible with mesoscopy."""
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
    help="Output directory for converted file, defaults to current working directory. Will be created if it doesn't exist.",
)
@click.option(
    "-t",
    "--ts-path",
    type=click.Path(exists=True),
    help="Path to timestamps file. Must be plain text or delimited.",
)
@click.option(
    "-d",
    "--ts-delimiter",
    type=click.Choice([",", "\t", " "], case_sensitive=False),
    default=",",
    help="Timestamp file delimiter, if file is delimited.",
)
@click.option(
    "-c",
    "--ts-column",
    type=int,
    help="Column in timestamp file to be used as timestamps. Must be provided if file is delimited and has more than one column.",
)
@click.option(
    "--ts-has-header",
    type=bool,
    is_flag=True,
    default=False,
    help="Flag signifying whether or not the timestamps file has a header row.",
)
@click.option(
    "-l",
    "--hdf5-only",
    is_flag=True,
    show_default=True,
    default=False,
    help="Generate an HDF5 linker file only without converting to NWB.",
)
@click.option(
    "-m",
    "--meta",
    "meta_path",
    type=click.Path(exists=True),
    help="Path to animal metadata file. Must be YAML or JSON format.",
)
@click.option("--subject-id", type=str, help="Metadata field - subject identifier.")
@click.option("--sex", type=click.Choice(["M", "F"], case_sensitive=False), help="Metadata field - subject sex.")
@click.option("--genotype", type=str, help="Metadata field - subject genotype.")
@click.option("--species", type=str, help="Metadata field - subject species.")
@click.option("--strain", type=str, help="Metadata field - subject strain.")
@click.option("--dob", type=str, help="Metadata field - subject date of birth in YYYY-MM-DD format (i.e. 1900-01-31).")
@click.option("--description", "session_description", type=str, help="Metadata field - session description.")
@click.option("--experimenter", type=str, help="Metadata field - experimenter name.")
@click.option("--lab", type=str, help="Metadata field - lab experiment was done in.")
@click.option("--institution", type=str, help="Metadata field - institution experiment was done in.")
def convert_video_cmd(**kwargs: typing.Any) -> None:
    """Convert a raw mesoscale calcium recording session from video to an NWB file compatible with mesoscopy."""
    if not kwargs.get("ts_path"):
        click.echo(
            "⚠️ WARNING - No timestamps file provided, using autogenerated timestamps. This might mess things up "
            "downstream."
        )
    if kwargs.pop("hdf5_only"):
        conv_vid.to_hdf5(**kwargs)
    else:
        conv_vid.to_nwb(**kwargs)
