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

"""Animal metadata readers."""

import json

import yaml

DEFAULT_METADATA = {
    "subject_id": "unknown",
    "sex": "Unknown",
    "genotype": "Unknown",
    "species": "Unknown",
    "strain": "Unknown",
    "dob": "1900-01-01",
    "session_description": "No description",
    "experimenter": "Unknown",
    "lab": "Unknown",
    "institution": "Unknown",
}


def read_yaml(path: str) -> dict:
    """Read a yaml metadata file.

    Args:
        path (str): Path to file.

    Raises:
        ValueError: Raised if file contains fields not in DEFAULT_METADATA.

    Returns:
        dict: Subject metadata.
    """
    with open(path) as fp:
        metadata = yaml.safe_load(fp)

        for key in metadata.keys():
            if key not in list(DEFAULT_METADATA.keys()):
                raise ValueError(f"Invalid file contents - unrecognised key <{key}>.")

        for field in DEFAULT_METADATA:
            if field not in metadata.keys():
                metadata[field] = DEFAULT_METADATA.get(field)

        return metadata


def read_json(path: str) -> dict:
    """Read a json metadata file.

    Args:
        path (str): Path to file.

    Raises:
        ValueError: Raised if file contains fields not in DEFAULT_METADATA.

    Returns:
        dict: Subject metadata.
    """
    with open(path) as fp:
        metadata = json.load(fp)

        for key in metadata.keys():
            if key not in list(DEFAULT_METADATA.keys()):
                raise ValueError(f"Invalid file contents - unrecognised key <{key}>.")

        for field in DEFAULT_METADATA:
            if field not in metadata.keys():
                metadata[field] = DEFAULT_METADATA.get(field)

        return metadata
