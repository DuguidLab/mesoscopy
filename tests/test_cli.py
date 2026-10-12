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
from pathlib import Path

import click
import h5py
import pytest
from click.testing import CliRunner

import mesoscopy

HEAVY_MODULES = ["napari", "sklearn", "scipy", "pims", "matplotlib", "pynwb", "dask", "h5py"]

# Modules no stage's `--help` may import, plus per-stage additions. Exact names, so `scipy.stats` is listed
# rather than `scipy`.
COMMON_HEAVY_MODULES = ["pynwb", "hdmf", "dask", "zarr", "pandas", "scipy.stats", "sklearn", "napari", "magicgui"]
COMMON_HEAVY_MODULES += ["pims", "av", "matplotlib", "plotly", "diptest", "skimage"]
STAGE_HEAVY_MODULES = {
    "process": [*COMMON_HEAVY_MODULES, "scipy.ndimage"],
    "align": [*COMMON_HEAVY_MODULES],
    "convert": [*COMMON_HEAVY_MODULES],
    "export": [*COMMON_HEAVY_MODULES, "tqdm"],
    "inspect": [*COMMON_HEAVY_MODULES],
    "preprocess": [*COMMON_HEAVY_MODULES],
    "register": [*COMMON_HEAVY_MODULES],
    "report": [*COMMON_HEAVY_MODULES],
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


def _walk(path: str, cmd: click.Command) -> list[tuple[str, click.Command]]:
    found = [(path, cmd)]
    if isinstance(cmd, click.Group):
        ctx = click.Context(cmd)
        for name in cmd.list_commands(ctx):
            sub = cmd.get_command(ctx, name)
            assert sub is not None
            found += _walk(f"{path} {name}", sub)
    return found


@pytest.mark.parametrize("name", list(mesoscopy.LAZY_SUBCOMMANDS))
def test_subcommand_summary_matches_help(name):
    cmd = mesoscopy.cli.get_command(click.Context(mesoscopy.cli), name)

    assert cmd.help.splitlines()[0] == mesoscopy.LAZY_SUBCOMMANDS[name][1]


def test_summaries_fit_group_listings():
    # Click cuts a summary off with "..." when it overruns the 80-column command listing.
    for path, cmd in _walk("mesoscopy", mesoscopy.cli):
        if not isinstance(cmd, click.Group):
            continue
        ctx = click.Context(cmd)
        names = cmd.list_commands(ctx)
        limit = 78 - 6 - max(len(name) for name in names)
        for name in names:
            assert not cmd.get_command(ctx, name).get_short_help_str(limit).endswith("..."), f"{path} {name}"


def test_docs_epilogs_point_at_docs_pages():
    docs = Path(__file__).parents[1] / "docs"
    epilogs = {path: cmd.epilog for path, cmd in _walk("mesoscopy", mesoscopy.cli) if cmd.epilog}

    assert "mesoscopy" in epilogs
    for path, epilog in epilogs.items():
        page = epilog.split(mesoscopy.DOCS_URL)[1].strip("/")
        assert (docs / f"{page or 'index'}.md").exists(), path


def test_start_time_prints_first_timestamp(tmp_path):
    path = str(tmp_path / "rec.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("/timestamps", data=[b"2025-05-20T14:42:39.6422784", b"2025-05-20T14:42:39.6622784"])

    result = CliRunner().invoke(mesoscopy.cli, ["start-time", path])

    assert result.exit_code == 0
    assert result.output == "2025-05-20T14:42:39.6422784\n"


def test_start_time_nwb(nwbfile):
    result = CliRunner().invoke(mesoscopy.cli, ["start-time", nwbfile])

    assert result.exit_code == 0
    assert result.output.startswith("2024-01-01T14:00:00")


def test_start_time_missing_timestamps_fails(tmp_path):
    path = str(tmp_path / "rec.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("/F", data=[0.0])

    result = CliRunner().invoke(mesoscopy.cli, ["start-time", path])

    assert result.exit_code == 1
    assert "no /timestamps" in result.output
