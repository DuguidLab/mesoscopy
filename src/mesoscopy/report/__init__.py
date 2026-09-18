#  Copyright (c) 2025 Constantinos Eleftheriou <Constantinos.Eleftheriou@ed.ac.uk>.
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
from pathlib import Path

import click
import numpy as np
from jinja2 import Environment
from jinja2 import PackageLoader
from jinja2 import select_autoescape

from mesoscopy import io

PREPROCESSING_REPORT_TEMPLATE = "preprocessing.html"
REGISTRATION_REPORT_TEMPLATE = "registration.html"
PERIEVENT_REPORT_TEMPLATE = "perievent.html"

env = Environment(loader=PackageLoader("mesoscopy.report", "templates"), autoescape=select_autoescape())


@click.command(name="report")
@click.argument("path")
@click.option(
    "-o",
    "--out_dir",
    type=click.Path(dir_okay=True),
    default=".",
    help="Output directory for the report.",
)
@click.option(
    "-t",
    "--trials",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Trials CSV giving sdt_type per trial, for peri-event files written without it.",
)
def report_cmd(path: str, out_dir: str, trials: str | None) -> str:
    """Generate an HTML report for a mesoscopy processing step.

    PATH is a *_preprocessed.h5, *_registered.h5 or *_perievent.csv file. Peri-event reports pick up the
    *_metrics.csv and *_metrics-session.csv written by `process metrics` next to the input when present.

    Args:
        path (str): Path to the input file.
        out_dir (str): Directory where the report will be saved. Defaults to the current directory
        trials (str, optional): Trials CSV for peri-event files without an sdt_type column.

    Returns:
        str: Path to the generated report.
    """  # noqa: DOC501
    if path.endswith("_preprocessed.h5"):
        return generate_preprocessing_report(path, out_dir=out_dir)
    elif path.endswith("_registered.h5"):
        return generate_registration_report(path, out_dir=out_dir)
    elif path.endswith("_perievent.csv"):
        return generate_perievent_report(path, out_dir=out_dir, trials_path=trials)
    else:
        msg = f"{path} is not a _preprocessed.h5, _registered.h5 or _perievent.csv file."
        raise click.ClickException(msg)


def generate_preprocessing_report(path: str, out_dir: str = ".") -> str:
    """Generate a preprocessing report for a mesoscopy recording.

    Args:
        path (str): Path to the preprocessed recording file.
        out_dir (str): Directory where the report will be saved. Defaults to the current directory.

    Returns:
        str: Path to the generated report.
    """
    import mesoscopy.preprocess.qa as preqa

    preproc = io.read_h5(path)
    session_id = Path(path).name.split("_preprocessed")[0]

    template = env.get_template(PREPROCESSING_REPORT_TEMPLATE)

    template_identifiers = {
        "session_id": session_id,
        "animal_id": session_id.split("_")[0].replace("sub-", ""),
        "session_date": session_id.split("_date-")[-1].replace("_", ":"),
        "experiment_id": session_id.split("_exp-")[-1].split("_")[0],
        "duration": str(
            (np.datetime64(preproc.get("timestamps")[-1]) - np.datetime64(preproc.get("timestamps")[0])).astype(  # type: ignore[attr-defined]
                "timedelta64[m]"
            )
        ),
        "frame_num": len(preproc.get("F")),  # type: ignore[attr-defined]
        "filesize": preproc.id.get_filesize() / (1024 * 1024),
        "qa_histogram_separation": preproc.attrs.get("/qa/checks/histogram_separation"),
        "qa_timestamp_consistency": preproc.attrs.get("/qa/checks/timestamp_consistency"),
        "qa_timestamp_jump": preproc.attrs.get("/qa/checks/timestamp_jump"),
        "qa_check_noise": preproc.attrs.get("/qa/checks/noise_check"),
        "qa_median_noise": np.median(np.array(preproc.get("/qa/noise_levels", 0))),
        "qa_check_snr": preproc.attrs.get("/qa/checks/snr_check"),
        "qa_snr": preproc.attrs.get("/qa/checks/snr"),
        "qa_check_bleaching": preproc.attrs.get("/qa/checks/bleaching_check"),
        "qa_bleaching_factor": preproc.attrs.get("/qa/checks/bleaching_factor"),
        "fig_integrity_timestamps": preqa.plot_timestamps(
            [np.datetime64(ts) for ts in preproc.get("timestamps")],  # type: ignore[attr-defined]
            as_html=True,
        ),
        "fig_integrity_timestamp_jumps": preqa.plot_timestamp_jumps(
            preproc.get("timestamps"),  # type: ignore[attr-defined]
            as_html=True,
        ),
        "fig_separation_pre_timeseries_mean": preqa.plot_raw_timeseries(
            preproc.get("qa").get("frame_means_timeseries"),  # type: ignore[attr-defined]
            as_html=True,
        ),
        "fig_separation_pre_hist_mean": preqa.plot_raw_histogram(
            preproc.get("qa").get("frame_means_timeseries"),  # type: ignore[attr-defined]
            as_html=True,
        ),
        "fig_separation_pre_hist_std": preqa.plot_raw_histogram(
            preproc.get("qa").get("frame_stds_timeseries"),  # type: ignore[attr-defined]
            as_html=True,
        ),
        "fig_separation_post_timeseries": preqa.plot_channels_timeseries(
            gcamp_channel=preproc.get("qa").get("gcamp_mean_timeseries"),  # type: ignore[attr-defined]
            isosb_channel=preproc.get("qa").get("isosb_mean_timeseries"),  # type: ignore[attr-defined]
            as_html=True,
        ),
        "fig_separation_post_gcamp_maxip": preqa.plot_projection(
            preproc.get("qa").get("gcamp_maxip_projection"),  # type: ignore[attr-defined]
            as_html=True,
        ),
        "fig_separation_post_gcamp_stdp": preqa.plot_projection(
            preproc.get("qa").get("gcamp_std_projection"),  # type: ignore[attr-defined]
            as_html=True,
        ),
        "fig_separation_post_isosb_maxip": preqa.plot_projection(
            preproc.get("qa").get("isosb_maxip_projection"),  # type: ignore[attr-defined]
            as_html=True,
        ),
        "fig_separation_post_isosb_stdp": preqa.plot_projection(
            preproc.get("qa").get("isosb_std_projection"),  # type: ignore[attr-defined]
            as_html=True,
        ),
        "fig_separation_post_filter_idx": preqa.plot_frame_ids(
            gcamp_ids=preproc.get("qa").get("gcamp_filter"),  # type: ignore[attr-defined]
            isosb_ids=preproc.get("qa").get("isosb_filter"),  # type: ignore[attr-defined]
            as_html=True,
        ),
        "fig_separation_post_filter_pie": preqa.plot_filter_pie(
            gcamp_ids=preproc.get("qa").get("gcamp_filter"),  # type: ignore[attr-defined]
            isosb_ids=preproc.get("qa").get("isosb_filter"),  # type: ignore[attr-defined]
            as_html=True,
        ),
        "fig_channel_dff": preqa.plot_channels_dff(
            gcamp_channel=preproc.get("qa").get("gcamp_dff_timeseries"),  # type: ignore[attr-defined]
            isosb_channel=preproc.get("qa").get("isosb_dff_timeseries"),  # type: ignore[attr-defined]
            as_html=True,
        ),
        "fig_corrected_dff": preqa.plot_corrected_dff(
            preproc.get("qa").get("f_mean_timeseries"),  # type: ignore[attr-defined]
            as_html=True,
        ),
        "fig_corrected_example": preqa.plot_frame(
            preproc.get("F")[100],  # type: ignore[attr-defined]
            as_html=True,
        ),
        "fig_corrected_maxip": preqa.plot_projection(preproc.get("qa").get("f_maxip"), as_html=True),  # type: ignore[attr-defined]
    }

    out_path = Path(out_dir) / Path(path).name.replace(".h5", "_report.html")
    out_path.write_text(template.render(template_identifiers), encoding="utf-8")
    return str(out_path)


def generate_registration_report(path: str, out_dir: str = ".") -> str:
    """Generate a registration report for a mesoscopy recording.

    Args:
        path (str): Path to the registered recording file.
        out_dir (str): Directory where the report will be saved. Defaults to the current directory.

    Returns:
        str: Path to the generated report.
    """
    import mesoscopy.register.qa as regqa

    registered = io.read_h5(path)
    session_id = Path(path).name.split("_preprocessed_registered")[0]

    template = env.get_template(REGISTRATION_REPORT_TEMPLATE)

    template_identifiers = {
        "session_id": session_id,
        "animal_id": session_id.split("_")[0].replace("sub-", ""),
        "session_date": session_id.split("_date-")[-1].replace("_", ":"),
        "experiment_id": session_id.split("_exp-")[-1].split("_")[0],
        "duration": str(
            (np.datetime64(registered.get("timestamps")[-1]) - np.datetime64(registered.get("timestamps")[0])).astype(  # type: ignore[attr-defined]
                "timedelta64[m]"
            )
        ),
        "frame_num": len(registered.get("F")),  # type: ignore[attr-defined]
        "filesize": registered.id.get_filesize() / (1024 * 1024),
        "fig_landmarks_unregistered": regqa.plot_landmarks(
            source_landmarks=registered.get("/qa/template_landmarks"),
            target_landmarks=registered.get("/qa/recording_landmarks"),
            as_html=True,
        ),
        "fig_landmarks_registered": regqa.plot_landmarks(
            source_landmarks=registered.get("/qa/template_landmarks"),
            target_landmarks=registered.get("/qa/registered_landmarks"),
            as_html=True,
        ),
        "fig_frame_example": regqa.plot_frame(
            registered.get("F")[1], landmarks=registered.get("/qa/template_landmarks"), as_html=True
        ),
    }

    out_path = Path(out_dir) / Path(path).name.replace(".h5", "_report.html")
    out_path.write_text(template.render(template_identifiers), encoding="utf-8")
    return str(out_path)


def generate_perievent_report(path: str, out_dir: str = ".", trials_path: str | None = None) -> str:
    """Generate a peri-event report from a `*_perievent.csv` and any metrics tables next to it.

    Args:
        path (str): Path to the peri-event CSV.
        out_dir (str): Directory where the report will be saved. Defaults to the current directory.
        trials_path (str, optional): Trials CSV giving `sdt_type` when the peri-event file lacks it.

    Returns:
        str: Path to the generated report.
    """
    import mesoscopy.report.perievent as pevreport

    payload, warnings = pevreport.report_payload(path, trials_path)
    for warning in warnings:
        click.echo(f"Warning: {warning}", err=True)

    name = Path(path).name
    session_id = name.split("_preprocessed")[0] if "_preprocessed" in name else name.removesuffix("_perievent.csv")
    template = env.get_template(PERIEVENT_REPORT_TEMPLATE)
    template_identifiers = {
        "session_id": session_id,
        "animal_id": session_id.split("_")[0].replace("sub-", ""),
        "session_date": session_id.split("_ses-")[-1].split("_")[0] if "_ses-" in session_id else None,
        "experiment_id": session_id.split("_exp-")[-1].split("_")[0] if "_exp-" in session_id else None,
        "event": payload["event"],
        "n_trials": payload["shape"][0],
        "n_samples": payload["shape"][1],
        "n_regions": payload["shape"][2],
        "window": (payload["time"][0], payload["time"][-1]),
        "has_types": payload["sdt_type"] is not None,
        "has_metrics": payload["session_metrics"] is not None,
        "payload": payload,
    }

    out_path = Path(out_dir) / name.replace(".csv", "_report.html")
    out_path.write_text(template.render(template_identifiers), encoding="utf-8")
    return str(out_path)
