#  Copyright (c) 2026 Constantinos Eleftheriou <Constantinos.Eleftheriou@ed.ac.uk>.
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

"""Tests for the peri-event report."""

import json
import pathlib
import shutil

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

import mesoscopy
from mesoscopy import resources
from mesoscopy.report import perievent as pevreport

REGIONS = ["L_MOp1", "R_MOp1", "L_VISp"]
STEM = "sub-01_exp-a_ses-20240101_regions_event-cueonset"


@pytest.fixture(scope="module")
def perievent_dir(tmp_path_factory):
    """A `_perievent.csv` with trials columns and its `_metrics` tables, from `process peri-event --with-metrics`."""
    data = tmp_path_factory.mktemp("perievent")
    time = np.arange(0, 30, 0.04)
    regions = pd.concat(
        [
            pd.DataFrame(
                {
                    "region": region,
                    "time_aligned": time,
                    "F": offset + np.sin(2 * np.pi * 0.5 * time),
                }
            )
            for offset, region in enumerate(REGIONS)
        ],
        ignore_index=True,
    )
    regions_path = data / "sub-01_exp-a_ses-20240101_regions.csv"
    regions.to_csv(regions_path, index=False)
    trials_path = data / "sub-01_exp-a_ses-20240101_trials.csv"
    pd.DataFrame(
        {
            "start_time": [4.0, 9.0, 14.0, 19.0, 24.0],
            "cue_onset": [4.5, 9.5, 14.5, 19.5, 24.5],
            "stop_time": [7.0, 12.0, 17.0, 22.0, 27.0],
            "response_time": [0.6, np.nan, 0.8, 0.7, 0.5],
            "sdt_type": ["hit", "miss", "hit", "correct_rejection", None],
        }
    ).to_csv(trials_path, index=False)
    result = CliRunner().invoke(
        mesoscopy.cli,
        args=f"process peri-event {regions_path} {trials_path} -o {data} --with-metrics --pre 1 --post 2",
    )
    assert result.exit_code == 0, result.output
    return data


@pytest.fixture(scope="module")
def perievent_csv(perievent_dir):
    return str(perievent_dir / f"{STEM}_perievent.csv")


class TestPerieventCube:
    def test_orders_and_fills_missing(self):
        table = pd.DataFrame(
            {
                "trial_index": [7, 7, 7, 7, 2, 2, 2],
                "time": [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5],
                "region": ["B", "A", "B", "A", "B", "A", "B"],
                "F": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            }
        )
        regions, time, trial_index, cube = pevreport.perievent_cube(table)

        assert regions == ["B", "A"]
        assert time.tolist() == [0.0, 0.5]
        assert trial_index.tolist() == [7, 2]
        assert cube.shape == (2, 2, 2)
        assert cube.dtype == np.float32
        assert cube[0].tolist() == [[1.0, 2.0], [3.0, 4.0]]
        assert cube[1, 0].tolist() == [5.0, 6.0]
        assert cube[1, 1, 0] == 7.0
        assert np.isnan(cube[1, 1, 1])

    def test_matches_written_file(self, perievent_csv):
        table = pd.read_csv(perievent_csv)
        regions, time, trial_index, cube = pevreport.perievent_cube(table)

        assert regions == REGIONS
        assert cube.shape == (5, len(time), 3)
        row = table.iloc[10]
        t = trial_index.tolist().index(row["trial_index"])
        k = time.tolist().index(row["time"])
        assert cube[t, k, regions.index(row["region"])] == pytest.approx(row["F"], abs=1e-6)


class TestTrialTypes:
    def test_from_table(self, perievent_csv):
        table = pd.read_csv(perievent_csv)
        assert pevreport.trial_types(table, table["trial_index"].unique()) == [
            "hit",
            "miss",
            "hit",
            "correct_rejection",
            "unlabelled",
        ]

    def test_from_trials_fallback(self, perievent_dir, perievent_csv):
        table = pd.read_csv(perievent_csv).drop(columns=["sdt_type"])
        trials = pd.read_csv(perievent_dir / "sub-01_exp-a_ses-20240101_trials.csv")
        assert pevreport.trial_types(table, np.array([3, 0]), trials) == ["correct_rejection", "hit"]

    def test_none_without_source(self, perievent_csv):
        table = pd.read_csv(perievent_csv).drop(columns=["sdt_type"])
        assert pevreport.trial_types(table, table["trial_index"].unique()) is None
        assert pevreport.trial_types(table, table["trial_index"].unique(), pd.DataFrame({"x": [1]})) is None


def test_pack_float32_roundtrip():
    cube = np.array([[[1.5, np.nan], [-2.0, 3.0]]], dtype=np.float64)
    text = pevreport.pack_float32(cube)
    back = pevreport.unpack_float32(text, cube.shape)

    assert back.dtype == np.float32
    np.testing.assert_array_equal(back, cube.astype(np.float32))


def test_atlas_outlines_cover_every_region():
    outlines = pevreport.atlas_outlines()
    left, _ = resources.get_atlas()
    acronyms = set(resources.get_atlas_annotations()["acronym"])
    n_regions = len(set(np.unique(left)) - {0})

    assert len(outlines) == 2 * n_regions
    assert {key[:2] for key in outlines} == {"L_", "R_"}
    assert all(key[2:] in acronyms for key in outlines)
    assert all(path.startswith("M") and path.endswith("Z") for path in outlines.values())
    assert "L_MOp1" in outlines


def test_metrics_paths():
    trial, session = pevreport.metrics_paths("/data/ses-01_regions_event-cueonset_perievent.csv")

    assert trial == pathlib.Path("/data/ses-01_regions_event-cueonset_metrics.csv")
    assert session == pathlib.Path("/data/ses-01_regions_event-cueonset_metrics-session.csv")


def _payload(html: str) -> dict:
    start = html.index('<script id="pev-data" type="application/json">') + len(
        '<script id="pev-data" type="application/json">'
    )
    return json.loads(html[start : html.index("</script>", start)])


def test_report_cmd_writes_perievent_report(perievent_csv, output_dir):
    result = CliRunner().invoke(mesoscopy.cli, args=f"report {perievent_csv} -o {output_dir}")

    assert result.exit_code == 0, result.output
    assert "Warning" not in result.output
    out_path = pathlib.Path(output_dir) / f"{STEM}_perievent_report.html"
    html = out_path.read_text(encoding="utf-8")
    payload = _payload(html)

    assert "sub-01_exp-a_ses-20240101_regions_event-cueonset" in html
    assert "Peri-event Report" in html
    assert 'id="pev-metric"' in html  # metrics section rendered
    assert payload["regions"] == REGIONS
    assert payload["shape"] == [5, len(payload["time"]), 3]
    assert payload["event"] == "cueonset"
    assert payload["sdt_type"] == ["hit", "miss", "hit", "correct_rejection", "unlabelled"]
    assert payload["trial_index"] == [0, 1, 2, 3, 4]
    assert payload["event_time"] == [4.5, 9.5, 14.5, 19.5, 24.5]
    assert len(payload["session_metrics"]) == 3
    assert {row["region"] for row in payload["trial_metrics"]} == set(REGIONS)
    assert set(payload["trial_metrics"][0]) == set(pevreport.TRIAL_METRIC_COLUMNS)
    assert "L_MOp1" in payload["atlas"]["paths"]

    cube = pevreport.unpack_float32(payload["traces"], tuple(payload["shape"]))
    table = pd.read_csv(perievent_csv)
    expected = table[(table["trial_index"] == 2) & (table["region"] == "L_VISp")].sort_values("time")["F"]
    np.testing.assert_allclose(cube[2, :, 2], expected.to_numpy(), atol=1e-6)


def test_report_without_metrics_warns_and_omits_section(perievent_csv, tmp_path):
    alone = tmp_path / pathlib.Path(perievent_csv).name
    shutil.copy(perievent_csv, alone)

    result = CliRunner().invoke(mesoscopy.cli, args=f"report {alone} -o {tmp_path}")

    assert result.exit_code == 0, result.output
    assert "No metrics tables" in result.output
    html = (tmp_path / f"{STEM}_perievent_report.html").read_text(encoding="utf-8")
    payload = _payload(html)
    assert payload["trial_metrics"] is None
    assert payload["session_metrics"] is None
    assert 'id="pev-metric"' not in html
    assert 'id="pev-markers"' not in html


def test_report_without_sdt_type_warns_unless_trials_given(perievent_dir, perievent_csv, tmp_path):
    alone = tmp_path / pathlib.Path(perievent_csv).name
    pd.read_csv(perievent_csv).drop(columns=["sdt_type"]).to_csv(alone, index=False)
    trials = perievent_dir / "sub-01_exp-a_ses-20240101_trials.csv"

    result = CliRunner().invoke(mesoscopy.cli, args=f"report {alone} -o {tmp_path}")
    assert result.exit_code == 0, result.output
    assert "No sdt_type column" in result.output
    html = (tmp_path / f"{STEM}_perievent_report.html").read_text(encoding="utf-8")
    assert _payload(html)["sdt_type"] is None
    assert 'id="pev-split"' not in html

    result = CliRunner().invoke(mesoscopy.cli, args=f"report {alone} -o {tmp_path} -t {trials}")
    assert result.exit_code == 0, result.output
    assert "No sdt_type column" not in result.output
    html = (tmp_path / f"{STEM}_perievent_report.html").read_text(encoding="utf-8")
    assert _payload(html)["sdt_type"] == ["hit", "miss", "hit", "correct_rejection", "unlabelled"]
    assert 'id="pev-split"' in html


def test_report_cmd_rejects_other_files(tmp_path):
    other = tmp_path / "ses-01_regions.csv"
    other.write_text("region,F\n")

    result = CliRunner().invoke(mesoscopy.cli, args=f"report {other}")

    assert result.exit_code == 1
    assert "not a _preprocessed.h5, _registered.h5 or _perievent.csv file" in result.output


def test_report_payload_rejects_empty_file(tmp_path):
    empty = tmp_path / "ses-01_perievent.csv"
    empty.write_text("trial_index,event_time,time,region,F\n")

    with pytest.raises(ValueError, match="no trials"):
        pevreport.report_payload(str(empty))
