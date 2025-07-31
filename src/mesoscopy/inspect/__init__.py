#  Copyright (c) 2022-2025 Constantinos Eleftheriou <Constantinos.Eleftheriou@ed.ac.uk>.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this
#   software and associated documentation files (the "Software"), to deal in the
#   Software without restriction, including without limitation the rights to use, copy,
#   modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#   and to permit persons to whom the Software is furnished to do so, subject to the
#   following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies
#  or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#  HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
#  IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
#  IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

"""File inspection CLI & GUI."""

import click

import mesoscopy.inspect.data_viewers as dvs
from mesoscopy import io


@click.command("inspect")
@click.argument(
    "input_path",
    type=click.Path(exists=True),
)
@click.option(
    "--meta-only",
    "info_level",
    flag_value="meta_only",
    default=True,
)
@click.option(
    "--acquisition",
    "info_level",
    flag_value="acquisition",
)
@click.option(
    "--deltaf",
    "info_level",
    flag_value="deltaf",
)
@click.option(
    "--registered",
    "info_level",
    flag_value="registered",
)
def inspect_cmd(input_path: str, info_level: str) -> None:
    """Inspect a calcium recording session and associated preprocessing output. Files must be in NWB format."""
    inspect(input_path=input_path, info_level=info_level)


def inspect(input_path: str, info_level: str = "meta_only") -> None:  # noqa: PLR0911
    """Inspect a mesoscale recording NWB file, alongside any associated preprocessing data, using the inspection GUI.

    Args:
        input_path (str): Path to NWB file.
        info_level (str): Inspection level. Defaults to metadata only.
    """
    nwbfile = io.read_nwb(input_path)

    click.echo("-" * (len(input_path) + 14))
    click.secho(f"Metadata for {input_path}", bold=True)
    click.echo("-" * (len(input_path) + 14))
    click.echo("")
    click.echo(f"File identifier: \t {nwbfile.identifier}")
    click.echo(f"Session date: \t\t {nwbfile.session_start_time.date()}")
    click.echo(f"Session description: \t {nwbfile.session_description}")
    click.echo(
        f"Experimenter: \t\t {nwbfile.experimenter[0] if nwbfile.experimenter else 'Unknown'} ({nwbfile.lab}, {
            nwbfile.institution
        })"
    )
    click.echo("")
    click.echo(f"Subject ID: \t\t {nwbfile.subject.subject_id if nwbfile.subject else 'Unknown'}")
    click.echo(f"Subject DOB: \t\t {nwbfile.subject.date_of_birth.date() if nwbfile.subject else 'Unknown'}")
    click.echo(f"Subject sex: \t\t {nwbfile.subject.sex if nwbfile.subject else 'Unknown'}")
    click.echo(f"Subject species: \t {nwbfile.subject.species if nwbfile.subject else 'Unknown'}")
    click.echo(f"Subject strain: \t {nwbfile.subject.strain if nwbfile.subject else 'Unknown'}")
    click.echo(f"Subject genotype: \t {nwbfile.subject.genotype if nwbfile.subject else 'Unknown'}")
    click.echo("")
    click.echo(f"Behavior trials: \t {f'{len(nwbfile.trials.id)} trials' if nwbfile.trials else 'None found'}")
    click.echo(
        f"Acquisition modules: \t {', '.join(list(nwbfile.acquisition.keys())) if nwbfile.acquisition else 'None found'}"
    )
    click.echo(
        f"Processing modules: \t {', '.join(list(nwbfile.processing.keys())) if nwbfile.processing else 'None found'}"
    )
    click.echo(f"Analysis modules: \t {', '.join(list(nwbfile.analysis.keys())) if nwbfile.analysis else 'None found'}")
    click.echo("")
    if nwbfile.processing and nwbfile.processing.get("ophys"):
        click.echo(click.style("Mesoscopy processing progress:", bold=True))
        click.echo(
            f"\t{
                f'{click.style("✅ Preprocessed", fg="green")}'
                if nwbfile.processing.get('ophys').get('DeltaFSeries')
                else f'{click.style("❌ Not preprocessed", fg="red")}'
            }"
        )
        click.echo(
            f"\t{
                f'{click.style("✅ Registered", fg="green")}'
                if 'CCFRegisteredSeries' in nwbfile.processing.get('ophys').data_interfaces
                else f'{click.style("❌ Not registered", fg="red")}'
            }"
        )
        click.echo(
            f"\t{
                f'{click.style(f"✅ Analysed ({', '.join(list(nwbfile.analysis.keys()))})", fg="green")}'
                if nwbfile.analysis
                else f'{click.style("❌ No analysis found", fg="red")}'
            }"
        )
    else:
        click.echo(click.style("No mesoscopy processing found.", bold=True))

    click.echo("")

    if info_level == "meta_only":
        return None
    if info_level == "acquisition":
        if not nwbfile.acquisition.get("DualChannelImagingSeries"):
            return click.secho("⚠️  No acquisition data found!", fg="red", bold=True)
        click.echo("Launching acquisition viewer...")
        return dvs.acquisition_viewer(nwbfile)
    if info_level == "deltaf":
        if not nwbfile.processing.get("ophys").get("DeltaFSeries"):
            return click.secho("⚠️  No DeltaFSeries found!", fg="red", bold=True)
        click.echo("Launching ∆F viewer...")
        return dvs.deltaf_viewer(nwbfile)
    if info_level == "registered":
        if not nwbfile.processing.get("ophys").get("CCFRegisteredSeries"):
            return click.secho("⚠️  No CCFRegisteredSeries found!", fg="red", bold=True)
        click.echo("Launching CCF registered ∆F viewer...")
        return dvs.ccfregistered_viewer(nwbfile)
    return None
