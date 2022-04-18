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
import os
import click
import tables
import mesoscopy.preprocess as preprocess
import mesoscopy.process as process
import mesoscopy.postprocess as postprocess

from matplotlib import pyplot as plt

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution("visiomode").version
except DistributionNotFound:
    # package is not installed
    __version__ = "debug"


@click.group()
def cli():
    """Widefield calcium imaging analysis pipeline."""
    click.echo("Mesoscopy version {}".__version__)


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.argument("out_dir", type=click.Path(dir_okay=True))
@click.option("--index", default=0, show_default=True, help="Index of frame to sample.")
@click.option("--crop", default=0)
def sample(path, out_dir, index, crop=0):
    """Sample an image frame from an HDF5 file and export it as a PNG."""
    click.echo("Sampling {} at index {}.".format(path, index))

    data = tables.open_file(path, "r")
    os.makedirs(out_dir, exist_ok=True)
    outpath = (
        out_dir
        + os.sep
        + path.split("/")[-1].replace(".h5", "_sample-{}.png".format(index))
    )

    plt.imsave(outpath, data.root.frames[index, crop:-crop, crop:-crop])
    click.echo("Saved sample at {}".format(outpath))


cli.add_command(preprocess.preprocess)
cli.add_command(process.process)
cli.add_command(postprocess.postprocess)
