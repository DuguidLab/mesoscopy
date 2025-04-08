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

import os

import click
import typing

import mesoscopy.convert.metadata as mtd
import mesoscopy.io as io


@click.command("convert")
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
    "-m", "--meta", type=click.Path(exists=True), help="Path to animal metadata file. Must be YAML or JSON format."
)
@click.option("--subject-id", type=str, help="Metadata field - subject identifier.")
@click.option("--sex", type=click.Choice(["M", "F"]), case_sensitive=False, help="Metadata field - subject sex.")
@click.option("--genotype", type=str, help="Metadata field - subject genotype.")
@click.option("--species", type=str, help="Metadata field - subject species.")
@click.option("--strain", type=str, help="Metadata field - subject strain.")
@click.option("--dob", type=str, help="Metadata field - subject date of birth in YYYY-MM-DD format (i.e. 1900-01-31).")
@click.option("--description", type=str, help="Metadata field - session description.")
@click.option("--experimenter", type=str, help="Metadata field - experimenter name.")
@click.option("--lab", type=str, help="Metadata field - lab experiment was done in.")
@click.option("--institution", type=str, help="Metadata field - institution experiment was done in.")
def convert_cmd(**kwargs: typing.Any) -> None:
    """Convert a raw mesoscale calcium recording session to an NWB file compatible with mesoscopy."""
    convert(**kwargs)


def convert(
    input_path: str,
    out_dir: str,
    meta_path: typing.Optional[str] = None,
    subject_id: typing.Optional[str] = None,
    sex: typing.Optional[str] = None,
    genotype: typing.Optional[str] = None,
    species: typing.Optional[str] = None,
    strain: typing.Optional[str] = None,
    dob: typing.Optional[str] = None,
    session_description: typing.Optional[str] = None,
    experimenter: typing.Optional[str] = None,
    lab: typing.Optional[str] = None,
    institution: typing.Optional[str] = None,
) -> str:
    """Convert a mesoscale recording session from HDF5 to NWB. Optionally add subject metadata.

    Metadata can be parsed from a compatible YAML or JSON file. Individual metadata arguments passed to this function (e.g. subject ID) will take precedence over the contents of a metadata file, if both are provided.

    Args:
        input_path (str): Path to raw HDF5 file.
        out_dir (str): Output directory.
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

    Returns:
        str: Path to new NWB file.
    """
    subject_metadata = mtd.DEFAULT_METADATA

    if meta_path:
        if meta_path.endswith(".yaml") or meta_path.endswith(".yml"):
            subject_metadata = mtd.read_yaml(meta_path)
        elif meta_path.endswith(".json"):
            subject_metadata = mtd.read_json(meta_path)
        else:
            click.echo("WARNING - Invalid metadata file provided, skipping...")

    # Infer subject_id if not provided.
    if not subject_id:
        try:
            # Assume NWB-style file-naming.
            subject_metadata["subject_id"] = input_path.split(os.sep)[-1].split("_")[-1]
        except IndexError:
            click.echo(
                "WARNING - Subject ID not provided and could not be inferred, using a default placeholder value. This might get confusing!"
            )

    if sex:
        subject_metadata["sex"] = sex
    if genotype:
        subject_metadata["genotype"] = genotype
    if species:
        subject_metadata["species"] = species
    if strain:
        subject_metadata["strain"] = strain
    if dob:
        subject_metadata["dob"] = dob
    if session_description:
        subject_metadata["session_description"] = session_description

    return ""
