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

from pynwb import NWBHDF5IO


def export_standalone(nwb_path: str, out_path: str = "") -> str:
    """Create a sharable copy of an NWB file by resolving external data links.

    This function reads an NWB file, generates a new unique identifier for the NWBFile,
    and exports it to a new file with all external data links resolved (i.e., all data
    is contained within the new file). This is useful for sharing NWB files without
    dependencies on external files (e.g. for uploading on DANDI).

    Args:
        nwb_path (str): Path to the source NWB file.
        out_path (str, optional): Path to save the exported NWB file. If not provided,
            the output file will be named as the input file with '_export.nwb' appended.

    Returns:
        str: Path to the exported NWB file.
    """
    with NWBHDF5IO(nwb_path, mode="r") as read_io:
        nwbfile = read_io.read()
        nwbfile.generate_new_id()

        if not out_path:
            out_path = nwb_path.replace(".nwb", "_export.nwb")
        with NWBHDF5IO(out_path, mode="w") as export_io:
            export_io.export(src_io=read_io, nwbfile=nwbfile, write_args={"link_data": False})

    return out_path
