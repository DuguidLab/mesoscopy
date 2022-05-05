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
"""Points registration CLI."""
import click
import xmltodict
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from skimage import transform as trf


@click.command()
@click.argument("recording_path", type=click.Path(exists=True))
@click.argument("out_dir", type=click.Path(dir_okay=True))
def landmarks(recording_path, out_dir, recording_points, template_points):
    """Register a recording to a template based on manually predefined landmarks."""
    template_landmarks = get_landmarks(template_points)
    recording_landmarks = get_landmarks(recording_points)


def get_landmarks(points_path):
    with open(points_path, "r") as fp:
        pts = xmltodict.parse(fp.read())
        pts = OrderedDict(
            {
                point["@name"]: (point["@x"], point["@y"])
                for point in pts["namedpointset"]["pointworld"]
            }
        )

    return np.array(list(pts.values()), dtype=np.float32)
