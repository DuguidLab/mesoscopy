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
@click.argument(
    "-o",
    "--out-dir",
    type=click.Path(),
    default="./",
    help="Output directory for converted file, defaults to current working directory. Will be created if it doesn't exist.",
)
@click.argument(
    "-m", "--meta", type=click.Path(exists=True), help="Path to animal metadata file. Must be YAML or JSON format."
)
def convert_cmd(**kwargs: typing.Any) -> None:
    """Convert a raw mesoscale calcium recording session to an NWB file compatible with mesoscopy."""


def convert(): ...
