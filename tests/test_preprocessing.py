import pytest

import mesoscopy.preprocess as preprocess


def test_nwb_link(clean_nwbfile, preproc_h5):
    preprocess.update_nwb(clean_nwbfile, preproc_h5)
