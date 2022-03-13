#  Copyright (c) 2022 Constantinos Eleftheriou <Constantinos.Eleftheriou@ed.ac.uk>.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this
#   software and associated documentation files (the "Software"), to deal in the
#   Software without restriction, including without limitation the rights to use, copy,
#   modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#   and to permit persons to whom the Software is furnished to do so, subject to the
#  following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies
#  or substantial portions of the Software
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
#  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
#  IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
#  IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

"""Preprocessing submodule."""
import os
import click
import h5py
import dask

from timeit import default_timer as timer  # Debugging performance


@click.command()
@click.argument("raw_path", type=click.Path(exists=True))
@click.argument("out_dir", type=click.Path(dir_okay=True))
def preprocess(raw_path, out_dir, channels):
    """Preprocessing to extract deltaF from a single session.

    Preprocessing separates the two channels, applies the haemodynamic correction,
    realigns the images to a common coordinate space (ABA) and extracts the delta F signal.

    Args:
        raw: Path to raw HDF5 file
        out_dir: Path to output directory for preprocessed data. This directory doesn't have to exist.

    """
    click.echo("Preprocessing file {}.".format(raw_path))

    session_id = raw_path.split("/")[-1].replace(".h5", "")
    os.makedirs(out_dir, exist_ok=True)


def separate_channels():
    pass


def mean_frame():
    pass


def haemodynamic_correction():
    pass


def register_to_frame():
    pass


def realign():
    pass


def extract_signal():
    pass
