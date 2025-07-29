import os
import pathlib

from click.testing import CliRunner
from pynwb import NWBHDF5IO

import mesoscopy
import mesoscopy.convert.hdf5 as conv_h5
import mesoscopy.convert.metadata as mtd
import mesoscopy.convert.video as conv_vid


def test_metadata_yaml(meta_yaml):
    expected_keys = list(mtd.DEFAULT_METADATA.keys())
    meta = mtd.read_yaml(meta_yaml)
    assert sorted(expected_keys) == sorted(list(meta.keys()))
    assert meta.get("subject_id") == "testyaml"
    assert meta.get("experimenter") == "John Doe"


def test_metadata_json(meta_json):
    expected_keys = list(mtd.DEFAULT_METADATA.keys())
    meta = mtd.read_json(meta_json)
    assert sorted(expected_keys) == sorted(list(meta.keys()))
    assert meta.get("subject_id") == "testjson"
    assert meta.get("experimenter") == "John Doe"


def test_convert_h5_linkonly(raw_h5, output_dir):
    expected_outpath = output_dir + os.sep + raw_h5.split(os.sep)[-1].replace(".h5", ".nwb")
    nwb_outpath = conv_h5.to_nwb(raw_h5, out_dir=output_dir, link_only=True)
    assert expected_outpath == nwb_outpath
    assert pathlib.Path(nwb_outpath).is_file()
    assert pathlib.Path(raw_h5).stat().st_size > pathlib.Path(nwb_outpath).stat().st_size


def test_convert_h5_eagercopy(raw_h5, output_dir):
    expected_outpath = output_dir + os.sep + raw_h5.split(os.sep)[-1].replace(".h5", ".nwb")
    nwb_outpath = conv_h5.to_nwb(raw_h5, out_dir=output_dir)
    assert expected_outpath == nwb_outpath
    assert pathlib.Path(nwb_outpath).is_file()
    assert pathlib.Path(raw_h5).stat().st_size <= pathlib.Path(nwb_outpath).stat().st_size


def test_convert_h5_metadata_yaml(raw_h5, output_dir, meta_yaml):
    nwb_outpath = conv_h5.to_nwb(raw_h5, out_dir=output_dir, meta_path=meta_yaml)

    io = NWBHDF5IO(nwb_outpath, mode="r")
    nwbfile = io.read()
    assert nwbfile.subject.subject_id == "testyaml"


def test_convert_h5_metadata_json(raw_h5, output_dir, meta_json):
    nwb_outpath = conv_h5.to_nwb(raw_h5, out_dir=output_dir, meta_path=meta_json)

    io = NWBHDF5IO(nwb_outpath, mode="r")
    nwbfile = io.read()
    assert nwbfile.subject.subject_id == "testjson"


def test_convert_h5_metadata_args(raw_h5, output_dir):
    nwb_outpath = conv_h5.to_nwb(
        raw_h5,
        out_dir=output_dir,
        subject_id="sometest",
        sex="F",
        genotype="Wt",
        species="Mus",
        strain="c57",
        dob="1999-01-02",
        session_description="somedesc",
        experimenter="Python Test",
        lab="well funded lab",
        institution="University of Great Science",
    )

    io = NWBHDF5IO(nwb_outpath, mode="r")
    nwbfile = io.read()
    assert nwbfile.subject.subject_id == "sometest"
    assert nwbfile.session_description == "somedesc"
    assert nwbfile.experimenter == ("Python Test",)


def test_convert_h5_metadata_args_file_mixed(raw_h5, output_dir): ...


def test_h5_cmd_parity(raw_h5, output_dir):
    """Check if click convert hdf5 cmd caller works"""
    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli,
        args=f"convert h5 --out-dir {output_dir} {raw_h5}",
    )
    assert result.exit_code == 0


def test_avi_h5_converter(raw_avi, output_dir):
    h5_outpath = conv_vid.to_hdf5(raw_avi, out_dir=output_dir)
    assert pathlib.Path(h5_outpath).is_file()


def test_avi_h5_converter_with_timestamps(raw_avi, raw_timestamps, output_dir):
    h5_outpath = conv_vid.to_hdf5(raw_avi, output_dir, ts_path=raw_timestamps)
    assert pathlib.Path(h5_outpath).is_file()


def test_avi_nwb_converter(raw_avi, output_dir):
    h5_outpath = conv_vid.to_nwb(raw_avi, out_dir=output_dir)
    assert pathlib.Path(h5_outpath).is_file()
