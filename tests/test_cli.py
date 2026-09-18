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

# Modules a stage's `--help` must not import; grows as each command module defers its imports. Exact names:
# `scipy` itself is cheap and comes in through dask, `scipy.stats` is not.
STAGE_HEAVY_MODULES = {
    "process": ["napari", "sklearn", "scipy.stats", "scipy.ndimage", "pims", "matplotlib", "plotly", "diptest"],
    "align": ["napari", "sklearn", "scipy.stats", "pims", "matplotlib", "plotly", "diptest", "skimage"],
    "convert": ["napari", "sklearn", "scipy.stats", "pims", "av", "matplotlib", "plotly", "diptest", "skimage"],
    "export": ["napari", "sklearn", "scipy.stats", "pims", "matplotlib", "plotly", "diptest", "skimage", "tqdm"],
    "inspect": ["napari", "sklearn", "scipy.stats", "pims", "matplotlib", "plotly", "diptest", "skimage"],
    "preprocess": ["napari", "sklearn", "scipy.stats", "pims", "matplotlib", "plotly", "diptest"],
    "register": ["napari", "magicgui", "skimage", "sklearn", "scipy.stats", "pims", "matplotlib", "plotly", "diptest"],
    "report": ["napari", "magicgui", "skimage", "sklearn", "scipy.stats", "pims", "matplotlib", "plotly", "diptest"],
}


def _imported_heavy_modules(args: list[str], heavy: list[str]) -> str:
    code = "import sys, mesoscopy; from click.testing import CliRunner; "
    code += f"CliRunner().invoke(mesoscopy.cli, {args!r}); "
    code += f"print(sorted(set(sys.modules) & set({heavy!r})))"
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603


def test_help_lists_every_subcommand():
    result = CliRunner().invoke(mesoscopy.cli, ["--help"])

    assert result.exit_code == 0
    for name in mesoscopy.LAZY_SUBCOMMANDS:
        assert name in result.output


def test_help_does_not_import_stage_modules():
    assert _imported_heavy_modules(["--help"], HEAVY_MODULES) == "[]"


@pytest.mark.parametrize(("stage", "heavy"), list(STAGE_HEAVY_MODULES.items()))
def test_stage_help_does_not_import_heavy_modules(stage, heavy):
    assert _imported_heavy_modules([stage, "--help"], heavy) == "[]"


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
