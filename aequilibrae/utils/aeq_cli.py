from pathlib import Path
import os
from inspect import signature

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


@cli.command(help="Load and return the AequilibraE run module")
@click.option("-d", "--project_dir", required=True, help="Project directory", default=os.getcwd)
@click.option("-f", "--function", type=click.Choice(list(available_parameters().keys())), required=True)
@click.argument("params", nargs=-1, required=False)
def run(project_dir, function, params):
    project = Project()
    project.open(project_dir)

    func = getattr(project.run, function)
    sig = signature(func)
    kwargs = {}
    if sig.parameters:
        keys = list(sig.parameters.keys())
        for i, par in enumerate(params):
            kwargs[keys[i]] = par
    result = func(**kwargs)

    click.echo(result)

    project.close()


@cli.command(help="Update the AequilibraE command line interface")
def update_cli():
    pass


if __name__ == "__main__":
    cli()
