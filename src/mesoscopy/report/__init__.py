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
from jinja2 import Environment
from jinja2 import PackageLoader
from jinja2 import select_autoescape

import plotly.express as px

PREPROCESSING_REPORT_TEMPLATE = "preprocessing.html"
REGISTRATION_REPORT_TEMPLATE = "registration.html"

env = Environment(loader=PackageLoader("mesoscopy.report", "templates"), autoescape=select_autoescape())


@click.command(name="report")
@click.argument("path")
@click.option(
    "-o",
    "--out_dir",
    type=click.Path(dir_okay=True),
    default=".",
    help="Output directory for preprocessed recording.",
)
def report_cmd(path: str, out_dir: str) -> str:
    """Generate a report for a mesoscopy processing step."""
    return generate_preprocessing_report(path)
    # if path.endswith("_preprocessed.h5"):
    #     return generate_preprocessing_report(path)
    # elif path.endswith("_registered.h5"):
    #     return generate_registration_report(path)
    # else:
    #     raise ValueError


def generate_preprocessing_report(path: str, out_dir: str = ".") -> str:
    template = env.get_template(PREPROCESSING_REPORT_TEMPLATE)

    data_canada = px.data.gapminder().query("country == 'Canada'")
    fig = px.bar(data_canada, x='year', y='pop') 

    out_path = out_dir / Path("test.html")
    out_path.write_text(template.render({
        "fig": fig.to_html(full_html=False)
    }), encoding="utf-8")
    return ""


def generate_registration_report(path: str) -> str:
    template = env.get_template(REGISTRATION_REPORT_TEMPLATE)
    return ""
