import argparse
import functools
import os


def add_subcommand_from_function(subparsers, func, defaults: dict):
    """Create an sub-command from a function's signature."""
    import inspect

    parser = subparsers.add_parser(func.__name__, description=func.__doc__)
    parser.set_defaults(_internal_target_func=func)

    sig = inspect.signature(func)

    for param_name, param in sig.parameters.items():
        # Determine if parameter has a default value
        annotation_type = param.annotation if param.annotation is not inspect.Parameter.empty else str

        param_cli_name = param_name.replace("_", "-")
        if param.default is inspect.Parameter.empty:
            # Required positional argument
            if default := defaults.get(param_name):
                parser.add_argument(param_cli_name, type=annotation_type, default=default)
            else:
                parser.add_argument(param_cli_name, type=annotation_type)
        else:
            # Optional argument with default
            parser.add_argument(
                f"--{param_cli_name}", default=defaults.get(param_name, param.default), type=annotation_type
            )

        # No handling of POSITIONAL_ONLY, VAR_POSITIONAL or VAR_KEYWORD, inspect docs say we could check the argument
        # kind via .kind https://docs.python.org/3/library/inspect.html#inspect.Parameter.kind but I'm not sure what we
        # would do with that information.


def list_functions(parser, args, unparsed_args):
    """
    List functions present in the run module.
    """
    from pprint import pprint

    from aequilibrae.project import Project

    # We attempt to parse the remaining arguments to provide a good error message in case something was provided.
    args = parser.parse_args(args=unparsed_args, namespace=args)

    project = Project()
    project.open(args.project)

    pprint(list(project.parameters["run"].keys()))

    project.close()


def run(args, unparsed_args):
    """
    Execute a function from the run module with argument parsing inferred from the function signature.
    """
    from pprint import pprint

    from aequilibrae.project import Project

    project = Project()
    project.open(args.project)

    run_module = project.run

    # We create a new parser because we don't want any of the old arguments to pollute the unparsed_args with their
    # default values (specifically "no_defaults").
    new_parser = argparse.ArgumentParser(prog="aeq run", description="Run module functions")
    subparsers = new_parser.add_subparsers(help="Available functions to run", required=True)

    # For each function we'll inspect the signature and create a sort of "best guess" set of arguments to accept. Does
    # not support POSITIONAL_ONLY, VAR_POSITIONAL or VAR_KEYWORD arguments because we supply everything as keyword
    # arguments.
    for func in run_module:
        add_subcommand_from_function(subparsers, func.func, func.keywords if not args.no_defaults else {})

    args = vars(new_parser.parse_args(args=unparsed_args))

    # Because of the .set_defaults trick we need to remove this "_internal_target_func" argument from the parsed
    # arguments. "_internal_target_func" corresponds to the run module function we should run. It does not have the
    # parameters.yml arguments applied via functools.partial like project.run does, but they should have been set as the
    # defaults
    target_function = args.pop("_internal_target_func")

    try:
        res = target_function(**args)
        if res is not None:
            pprint(res)
    finally:
        project.close()


def cli():
    # Create a parser that just parses the project dir. This is global and required by every sub-command.
    parser = argparse.ArgumentParser(description="Load and return AequilibraE run module")
    parser.add_argument(
        "-p", "--project", default=os.getcwd(), type=str, help="Path to the project folder", required=True
    )

    # We'll add out sub-commands via a sub-parser. The function correspond to the sub-command is set via the
    # .set_defaults trick given in the docs
    # https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_subparsers
    subparsers = parser.add_subparsers(title="commands", help="Available run commands", required=True)

    run_parser = subparsers.add_parser("run", help="Run project commands")
    run_parser.add_argument(
        "--no-defaults", action="store_true", help="do no use default arguments from parameters.yml"
    )
    run_parser.set_defaults(_internal_func=run)

    list_functions_parser = subparsers.add_parser("list", help="List project commands")
    list_functions_parser.set_defaults(_internal_func=functools.partial(list_functions, list_functions_parser))

    args, unparsed_args = parser.parse_known_args()

    # args now contains the project and the "_internal_func" to run from .set_defaults. unparsed_args should contain
    # everything else.
    args._internal_func(args, unparsed_args)
