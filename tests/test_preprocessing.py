import pytest

import mesoscopy.preprocess as preprocess
import mesoscopy.preprocess.calculations as calculations


def test_load_raw_h5(raw_h5):
    preprocess._load_raw(raw_h5, nwb=False)


def test_load_raw_nwb(nwbfile):
    preprocess._load_raw(nwbfile, nwb=True)


def test_channel_qa():
    pass


def test_nwb_link(nwbfile, preproc_h5):
    preprocess._update_nwb(nwbfile, preproc_h5)


def test_preprocess_h5(raw_h5):
    ...


def test_preprocess_h5_crop(raw_h5):
    ...


def test_preprocess_h5_frameskip(raw_h5):
    ...


def test_preprocess_h5_bin(raw_h5):
    ...

def test_preprocess_nwb(nwbfile):
    ...


def test_preprocess_nwb_crop(nwbfile):
    ...


def test_preprocess_nwb_frameskip(nwbfile):
    ...


def test_preprocess_nwb_bin(nwbfile):
    ...


def test_preprocess_nwb_channel_means(nwbfile):
    ...


def test_preprocess_nwb_use_means(nwbfile):
    ...


def test_preprocess_nwb_flip_channels(nwbfile):
    ...


def test_preprocess_nwb_use_means_flip_channels(nwbfile):
    ...


def test_preprocess_nwb_interim_dir(nwbfile):
    ...


def test_preprocess_nwb_chunksize(nwbfile):
    ...
