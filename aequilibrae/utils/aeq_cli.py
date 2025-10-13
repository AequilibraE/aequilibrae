from pathlib import Path
import os
import enum
import yaml

import click

from aequilibrae.project import Project
from aequilibrae.parameters import Parameters


@click.group(invoke_without_command=False)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo("You can only run procedures from the 'run' module.")


def available_parameters():
    current_dir = os.getcwd()

    p = Parameters(Path(current_dir))
    return p.parameters["run"]


@cli.command(help="Load and return the AequilibraE run module")  # type: ignore
@click.option("--function", type=click.Choice(list(available_parameters().keys())), required=False)
def run(function):
    current_dir = Path.cwd()

    project = Project()
    project.open(current_dir)

    func = getattr(project.run, function)
    result = func()

    click.echo(result)

    project.close()


if __name__ == "__main__":
    cli()  # type: ignore
