import os
from pathlib import Path
from inspect import signature

import click

from aequilibrae.parameters import Parameters
from aequilibrae.project import Project


@click.group(invoke_without_command=False)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo("You can only invoke commands run or list-functions")


def available_parameters():
    current_dir = os.getcwd()

    p = Parameters(Path(current_dir))
    return p.parameters["run"]


@cli.command(help="Load and return the AequilibraE run module for current directory.")
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


@cli.command(help="List available functions in the run module.")
@click.option("-d", "--project_dir", required=True, help="Project directory", default=os.getcwd)
def list_functions(project_dir):
    p = Parameters(Path(project_dir))

    click.echo("Available functions: ", list(p.parameters["run"].keys()))


if __name__ == "__main__":
    cli()
