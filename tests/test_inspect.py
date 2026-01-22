from click.testing import CliRunner

import mesoscopy
import mesoscopy.inspect as insp


def test_inspect_meta(preproc_nwb):
    insp.inspect(input_path=preproc_nwb, info_level="meta_only")


def test_inspect_meta_noprocessing(nwbfile):
    insp.inspect(input_path=nwbfile, info_level="meta_only")


# def test_inspect_acquisition(nwbfile):
#     insp.inspect(input_path=nwbfile, info_level="acquisition")


# def test_inspect_acquisition_notfound(nwbfile_noacquisition):
#     insp.inspect(input_path=nwbfile_noacquisition, info_level="acquisition")


# def test_inspect_deltaf(preproc_nwb):
#     insp.inspect(input_path=preproc_nwb, info_level="deltaf")


# def test_inspect_deltaf_notfound(nwbfile):
#     insp.inspect(input_path=nwbfile, info_level="deltaf")


# def test_inspect_registration(): ...


# def test_inspect_registration_notfound(nwbfile): ...


def test_cmd_parity(nwbfile):
    """Check if click preprocess cmd caller works"""
    runner = CliRunner()
    result = runner.invoke(
        mesoscopy.cli,
        args=f"inspect {nwbfile}",
    )
    assert result.exit_code == 0
