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

# Lazy subcommands are imported only when they are looked up, to avoid importing heavy dependencies unnecessarily.
LAZY_SUBCOMMANDS = {
    "align": ("mesoscopy.align:align_cmd", "Write behaviour-aligned frame timestamps to a recording."),
    "convert": ("mesoscopy.convert:convert_cmd", "Convert raw recordings to NWB or HDF5."),
    "export": ("mesoscopy.export:export_cmd", "Export mesoscopy-generated files to other formats."),
    "inspect": ("mesoscopy.inspect:inspect_cmd", "Inspect a recording session and associated preprocessing output."),
    "postprocess": ("mesoscopy.postprocess:postprocess_cmd", "Postprocess extracted activity."),
    "preprocess": ("mesoscopy.preprocess:preprocess_cmd", "Extract haemodynamics-corrected deltaF from a recording."),
    "process": ("mesoscopy.process:process_cmd", "Extract activity from preprocessed or registered recordings."),
    "register": ("mesoscopy.register:register_cmd", "Register recordings to an anatomical template."),
    "report": ("mesoscopy.report:report_cmd", "Generate an HTML report for a mesoscopy processing step."),
    "sample": ("mesoscopy:sample", "Sample an image frame from an HDF5 file and export it as a PNG."),
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


@click.group(cls=LazyGroup, lazy_subcommands=LAZY_SUBCOMMANDS)
@click.version_option(__version__)
def cli():
    """Widefield calcium imaging analysis pipeline."""


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.argument("out_dir", type=click.Path(dir_okay=True))
@click.option("--index", default=0, show_default=True, help="Index of frame to sample.")
@click.option("--crop", default=0)
@click.option("--vmin", type=float, default=0)
@click.option("--vmax", type=float, default=255)
@click.option("--key", type=str, default=None, help="Activity column")
def sample(path, out_dir, index, crop=0, vmin=0, vmax=255, key="frames"):
    """Sample an image frame from an HDF5 file and export it as a PNG."""
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
