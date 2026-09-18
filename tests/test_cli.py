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

"""Tests for the root CLI group."""

import subprocess
import sys

import click
import pytest
from click.testing import CliRunner

import mesoscopy

HEAVY_MODULES = ["napari", "sklearn", "scipy", "pims", "matplotlib", "pynwb", "dask", "h5py"]


def test_help_lists_every_subcommand():
    result = CliRunner().invoke(mesoscopy.cli, ["--help"])

    assert result.exit_code == 0
    for name in mesoscopy.LAZY_SUBCOMMANDS:
        assert name in result.output


def test_help_does_not_import_stage_modules():
    code = "import sys, mesoscopy; from click.testing import CliRunner; "
    code += "CliRunner().invoke(mesoscopy.cli, ['--help']); "
    code += f"print(sorted(m for m in sys.modules if m.split('.')[0] in {HEAVY_MODULES!r}))"
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout  # noqa: S603

    assert output.strip() == "[]"


@pytest.mark.parametrize("name", list(mesoscopy.LAZY_SUBCOMMANDS))
def test_every_subcommand_resolves(name):
    ctx = click.Context(mesoscopy.cli)
    cmd = mesoscopy.cli.get_command(ctx, name)

    assert isinstance(cmd, click.Command)
    assert cmd.name == name


def test_no_eager_subcommands():
    # mkdocs-click ignores lazy subcommands when any are registered with `cli.add_command`.
    assert mesoscopy.cli.commands == {}


def test_unknown_subcommand_is_none():
    assert mesoscopy.cli.get_command(click.Context(mesoscopy.cli), "nope") is None
