import pytest

import mesoscopy.preprocess as preprocess


def test_nwb_link(nwbfile, preproc_h5):
    preprocess.update_nwb(nwbfile, preproc_h5)
