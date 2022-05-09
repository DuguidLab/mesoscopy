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

import os
import click
import imageio
import numpy as np


@click.group()
def utils():
    """General utilities that might come in handy when registering brains to an atlas."""
    pass


@utils.command()
@click.argument("images", type=click.Path(exists=True), nargs=-1)
@click.option("-o", "--out_dir", type=click.Path(dir_okay=True), default="./")
def average_anatomy(images, out_dir):
    click.echo("Generating average anatomy...")
    image_array = np.array([imageio.imread(image, as_gray=True) for image in images])

    outpath = out_dir + os.sep + "average_anatomy.png"
    imageio.imsave(outpath, image_array.mean(axis=0).astype(np.uint8))
