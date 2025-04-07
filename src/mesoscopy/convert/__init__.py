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

import click
import typing


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
@click.option("--description", type=str, help="Metadata field - session description")
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
): ...
