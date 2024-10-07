import pytest

import mesoscopy.preprocess as preprocess
import mesoscopy.preprocess.calculations as calculations


def test_load_raw_h5(raw_h5):
    preprocess._load_raw(raw_h5, nwb=False)


def test_load_raw_nwb(nwbfile):
    preprocess._load_raw(nwbfile, nwb=True)


def test_update_nwb(nwbfile, preproc_h5):
    pass


def test_channel_qa():
    pass


def test_nwb_link(nwbfile, preproc_h5):
    preprocess._update_nwb(nwbfile, preproc_h5)
