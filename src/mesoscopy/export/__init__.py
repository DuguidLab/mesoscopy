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
"""Export functions for mesoscopy-generated files."""

import typing

import click

import mesoscopy.export.nwb as exp_nwb


@click.group("export")
def export_cmd() -> None:
    """Export mesoscopy-generated files."""


@export_cmd.command("nwb")
@click.argument("nwb_path", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option(
    "--out-path",
    type=click.Path(dir_okay=False, path_type=str),
    default="",
    help="Path to save the exported NWB file. If not provided, the output file will be named as the input file with '_export.nwb' appended.",
)
def export_nwb_cmd(**kwargs: typing.Any) -> None:
    """Create a sharable copy of an NWB file by resolving external data links."""
    click.echo("Exporting NWB file...")
    export_path = export_nwb(**kwargs)
    click.echo(f"NWB file exported to {export_path}")


def export_nwb(nwb_path: str, out_path: str = "") -> str:
    """Create a sharable copy of an NWB file by resolving external data links.

    Args:
        nwb_path (str): Path to the source NWB file.
        out_path (str, optional): Path to save the exported NWB file. If not provided, the output file will be named as the input file with '_export.nwb' appended. Defaults to "".

    Returns:
        str: Path to the exported NWB file.
    """
    return exp_nwb.export_standalone(nwb_path, out_path)
