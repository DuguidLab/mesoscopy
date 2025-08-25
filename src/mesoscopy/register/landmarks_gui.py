#  Copyright (c) 2024 Constantinos Eleftheriou <Constantinos.Eleftheriou@ed.ac.uk>.
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
from collections import OrderedDict

import magicgui.widgets as mgw
import napari
import numpy as np
import numpy.typing as npt
from dask import array as da

COLOR_CYCLE = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
]


def mark_landmarks(
    maxip_image: npt.NDArray | da.Array, alt_image: npt.NDArray | da.Array | None, template_landmarks: dict = {}
) -> dict:
    """Launch the napari viewer to identify anatomical landmarks on a maximum intensity projection image.

    The following landmarks are identified for registration:
    - bregma: Just plain ol' bregma.
    - cFP: Frontal pole center.
    - rFP: Rightmost aspect of the frontal pole.
    - lFP: Leftmost aspect of the frontal pole.
    - rPB: Right lateral edge of the parietal bone.
    - lPB: Left lateral edge of the parietal bone.
    - lpRSP: Left posterior aspect of the retrosplenial cortex.
    - rpRSP: Right posterior aspect of the retrosplenial cortex.
    - aIPB: Anterior aspect of the interparietal bone.

    Args:
        maxip_image (npt.NDArray): Maximum intensity projection image. Could be either channel.
        alt_image (npt.NDArray): Alternative image to be displayed alongside the maximum intensity projection image. Usually a second channel.
        template_landmarks (dict, optional): Dictionary with the landmarks and their x-y coordinates. Dictionary keys are landmark names, while x-y coordinates are stored as an (y, x) tuple. Defaults to {}.

    Returns:
        dict: Dictionary with the landmarks and their x-y coordinates. Dictionary keys are landmark names, while x-y coordinates are stored as an (y, x) tuple.

    Raises:
        ValueError: If the maximum intensity projection and alternative image do not have the same dimensions.
    """
    if len(maxip_image.shape) == 3:
        maxip_height, maxip_width, _ = maxip_image.shape
    else:
        maxip_height, maxip_width = maxip_image.shape

    viewer = napari.view_image(maxip_image)

    if alt_image is not None:
        if len(alt_image.shape) == 3:
            alt_height, alt_width, _ = alt_image.shape
        else:
            alt_height, alt_width = alt_image.shape
        if (maxip_height, maxip_width) != (alt_height, alt_width):
            msg = "Maximum intensity projection and alternative image must have the same dimensions."
            raise ValueError(msg)

        viewer.add_image(alt_image, name="alt_maxip")

    default_landmark_locations = template_landmarks or {
        "bregma": (maxip_height / 2, maxip_width / 2),
        "cFP": (maxip_height / 7, maxip_width / 2),
        "rFP": (maxip_height / 7, maxip_width / 1.5),
        "lFP": (maxip_height / 7, maxip_width / 3),
        "rPB": (maxip_height / 4, maxip_width / 1.25),
        "lPB": (maxip_height / 4, maxip_width / 5),
        "lpRSP": (maxip_height / 1.25, maxip_width / 2.25),
        "rpRSP": (maxip_height / 1.25, maxip_width / 1.75),
        "aIPB": (maxip_height / 1.4, maxip_width / 2),
    }

    landmarks = list(default_landmark_locations.keys())

    points_layer = viewer.add_points(
        data=np.array(list(default_landmark_locations.values())),
        name="landmarks",
        ndim=2,
        properties={"label": landmarks},
        property_choices={"label": landmarks},
        symbol="o",
        face_color="label",
        face_color_cycle=COLOR_CYCLE,
        border_width=0,  # fraction of point size
        size=5,
    )
    points_layer.face_color_mode = "cycle"

    # add the label menu widget to the viewer
    label_widget = _create_label_menu(points_layer, landmarks)
    viewer.window.add_dock_widget(label_widget)

    points_layer.mode = "select"

    napari.run()

    return OrderedDict(zip(landmarks, points_layer.data, strict=True))


def _create_label_menu(points_layer, labels):
    """Create a label menu widget that can be added to the napari viewer dock.

    Borrowed from https://napari.org/stable/tutorials/annotation/annotate_points.html.

    Args:
        points_layer (napari.layers.Points): A napari points layer.
        labels (List[str]): List of the labels for each keypoint to be annotated (e.g., the body parts to be labeled).

    Returns:
        Container: the magicgui Container with our dropdown menu widget
    """
    # Create the label selection menu
    label_menu = mgw.ComboBox(label="feature_label", choices=labels)
    info_text = mgw.Label(value="When you're done annotating,\nclose the viewer window.")
    close_button = mgw.Button(label="Save and Close")
    label_widget = mgw.Container(widgets=[label_menu, info_text, close_button])

    def close_viewer(event) -> None:
        napari.current_viewer().close()

    close_button.changed.connect(close_viewer)

    def update_label_menu(event) -> None:
        """Update the label menu when the point selection changes."""
        new_label = str(points_layer.current_properties["label"][0])
        if new_label != label_menu.value:
            label_menu.value = new_label

    points_layer.events.current_properties.connect(update_label_menu)

    def label_changed(new_label) -> None:
        """Update the Points layer when the label menu selection changes."""
        current_properties = points_layer.current_properties
        current_properties["label"] = np.asarray([new_label])
        points_layer.current_properties = current_properties
        points_layer.refresh_colors()

    label_menu.changed.connect(label_changed)

    return label_widget
