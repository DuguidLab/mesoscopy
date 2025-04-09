import pytest

import mesoscopy.convert as conv
import mesoscopy.convert.metadata as mtd


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


def test_convert_h5_linkonly(raw_h5, output_dir): ...


def test_convert_h5_eagercopy(raw_h5, output_dir): ...


def test_convert_h5_metadata_args(raw_h5, output_dir): ...


def test_convert_h5_metadata_args_file_mixed(raw_h5, output_dir): ...


def test_convert_h5_cli(raw_h5, output_dir): ...
