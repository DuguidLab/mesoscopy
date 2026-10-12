#  Copyright (c) 2022 Constantinos Eleftheriou <Constantinos.Eleftheriou@ed.ac.uk>.
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

"""Main entry point to the mesoscopy CLI"""

import importlib
import os
import typing
from importlib.metadata import version

import click

__version__ = version("mesoscopy")

DOCS_URL = "https://docs.mesoscopy.org"


def docs_epilog(page: str = "") -> str:
    """Help epilog pointing at the docs.

    Args:
        page: Page path under the docs site, such as "how-to/metrics/". Empty for the home page.

    Returns:
        The epilog, marked so that click does not rewrap the URL.
    """
    return f"\b\nDocumentation at {DOCS_URL}/{page}" if page else f"\b\nDocumentation at {DOCS_URL}"


# Lazy subcommands are imported only when they are looked up, to avoid importing heavy dependencies unnecessarily.
LAZY_SUBCOMMANDS = {
    "align": ("mesoscopy.align:align_cmd", "Align frame timestamps to a behaviour session."),
    "convert": ("mesoscopy.convert:convert_cmd", "Convert raw recordings to NWB or HDF5."),
    "export": ("mesoscopy.export:export_cmd", "Export mesoscopy files to other formats."),
    "inspect": ("mesoscopy.inspect:inspect_cmd", "Inspect an NWB recording and its processing progress."),
    "postprocess": ("mesoscopy.postprocess:postprocess_cmd", "Postprocess extracted activity."),
    "preprocess": ("mesoscopy.preprocess:preprocess_cmd", "Extract haemodynamics-corrected ∆F/F from a recording."),
    "process": ("mesoscopy.process:process_cmd", "Analyse preprocessed or registered recordings."),
    "register": ("mesoscopy.register:register_cmd", "Register recordings to an anatomical template."),
    "report": ("mesoscopy.report:report_cmd", "Generate an HTML report for a pipeline stage."),
    "sample": ("mesoscopy:sample", "Save one frame of an HDF5 recording as a PNG."),
    "start-time": ("mesoscopy:start_time", "Print a recording's start time."),
}


class LazyGroup(click.Group):
    """Click group that imports a subcommand's module only when that subcommand is looked up."""

    def __init__(
        self, *args: typing.Any, lazy_subcommands: dict[str, tuple[str, str]] | None = None, **kwargs: typing.Any
    ) -> None:
        """Initialise the group.

        Args:
            *args: Passed to `click.Group`.
            lazy_subcommands: Mapping of subcommand name to `("module:attribute", summary)` of its click command.
            **kwargs: Passed to `click.Group`.
        """
        super().__init__(*args, **kwargs)
        self.lazy_subcommands = lazy_subcommands or {}

    def list_commands(self, ctx: click.Context) -> list[str]:
        """List registered and lazy subcommand names without importing the lazy ones.

        Args:
            ctx: Click context.

        Returns:
            Sorted subcommand names.
        """
        return sorted([*super().list_commands(ctx), *self.lazy_subcommands])

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Return the subcommand, importing its module if it is lazy.

        Args:
            ctx: Click context.
            cmd_name: Subcommand name.

        Returns:
            The click command, or `None` if no such subcommand exists.
        """
        if cmd_name in self.lazy_subcommands:
            return self._load_lazy_command(cmd_name)
        return super().get_command(ctx, cmd_name)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Write the subcommand listing, using stored summaries for lazy subcommands instead of importing them.

        Args:
            ctx: Click context.
            formatter: Help formatter to write to.
        """
        rows = []
        for name in self.list_commands(ctx):
            if name in self.lazy_subcommands:
                rows.append((name, self.lazy_subcommands[name][1]))
                continue
            cmd = super().get_command(ctx, name)
            if cmd is not None and not cmd.hidden:
                rows.append((name, cmd.get_short_help_str(limit=formatter.width - 6 - len(name))))
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)

    def _load_lazy_command(self, cmd_name: str) -> click.Command:
        module_name, attr_name = self.lazy_subcommands[cmd_name][0].split(":")
        cmd = getattr(importlib.import_module(module_name), attr_name)
        if not isinstance(cmd, click.Command):
            msg = f"Lazy subcommand {cmd_name!r} resolved to {cmd!r}, not a click command."
            raise TypeError(msg)
        return cmd


@click.group(cls=LazyGroup, lazy_subcommands=LAZY_SUBCOMMANDS, epilog=docs_epilog())
@click.version_option(__version__)
def cli():
    """Widefield calcium imaging analysis pipeline."""


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.argument("out_dir", type=click.Path(dir_okay=True))
@click.option("--index", default=0, show_default=True, help="Index of the frame to save.")
@click.option("--crop", default=0, show_default=True, help="Pixels to crop from each edge.")
@click.option("--vmin", type=float, default=0, show_default=True, help="Lower end of the colour scale.")
@click.option("--vmax", type=float, default=255, show_default=True, help="Upper end of the colour scale.")
@click.option("--key", type=str, default=None, help="Dataset holding the frames, such as F or frames.")
def sample(path, out_dir, index, crop=0, vmin=0, vmax=255, key="frames"):
    """Save one frame of an HDF5 recording as a PNG.

    PATH is an HDF5 recording and OUT_DIR the directory for the image. Pick the frame with --index and its dataset
    with --key. Writes <name>_sample-<index>.png.
    """
    import h5py
    from matplotlib import pyplot as plt

    click.echo(f"Sampling {path} at index {index}.")

    f = h5py.File(path)
    d = f[f"/{key}"]

    os.makedirs(out_dir, exist_ok=True)
    outpath = out_dir + os.sep + path.split("/")[-1].replace(".h5", f"_sample-{index}.png")

    if crop > 0:
        plt.imsave(outpath, d[index, crop:-crop, crop:-crop], vmin=vmin, vmax=vmax)
    else:
        plt.imsave(outpath, d[index], vmin=vmin, vmax=vmax, cmap="jet")
    click.echo(f"Saved sample at {outpath}")


@click.command("start-time")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
def start_time(path: str) -> None:
    """Print a recording's start time.

    PATH is an HDF5 recording or an NWB file. Prints the first frame timestamp of an HDF5 file, or the session start
    time of an NWB file.
    """  # noqa: DOC501
    from mesoscopy import io

    try:
        click.echo(io.read_start_time(path))
    except ValueError as err:
        raise click.ClickException(str(err)) from err
