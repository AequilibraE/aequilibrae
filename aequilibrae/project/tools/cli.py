import argparse
import functools
import inspect
import os

# Dests used internally via the .set_defaults trick. The prefix keeps them from colliding with
# real function parameters.
_VAR_KEYWORDS = "_aeq_internal_var_keywords"


def _parse_value(text: str):
    """Parse a command line value as a Python literal when possible (numbers, booleans, lists, ...)
    so values round-trip with the types found in parameters.yml, otherwise keep it as a string.

    Quoting a value forces it to be a string, e.g. '"101"'.
    """
    import ast

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _keyword_pair(text: str):
    """
    Parse a 'key=value' string into a (key, value) tuple.
    """
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected 'key=value', got '{text}'")

    return key, _parse_value(value)


def _annotation_help(param):
    """
    Type annotations are not enforced, they are only surfaced in the help text.
    """
    if param.annotation is inspect.Parameter.empty:
        return None
    return f"type: {inspect.formatannotation(param.annotation)}"


def _extract_target_call(args: dict):
    """
    Split a parsed argument dict into the target function and the keyword arguments to call it with.
    """
    target_function = args.pop("_aeq_internal_target_func")
    var_keywords = dict(args.pop(_VAR_KEYWORDS, []))

    return target_function, {**args, **var_keywords}


def add_subcommand_from_function(subparsers, func, defaults: dict):
    """
    Create a sub-command from a function's signature.
    """
    doc = inspect.getdoc(func)
    parser = subparsers.add_parser(func.__name__, description=doc, help=doc.split("\n")[0] if doc else None)
    parser.set_defaults(_aeq_internal_target_func=functools.partial(func, **defaults) if defaults else func)

    var_keyword = None
    for param_name, param in inspect.signature(func).parameters.items():
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise ValueError(f"positional-only parameters ('{param_name}') are not supported by the aeq CLI")
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            var_keyword = param_name
            continue
        elif param.kind is inspect.Parameter.VAR_POSITIONAL:
            # We call the function with keyword arguments only, so *args can never be supplied.
            continue

        param_cli_name = param_name.replace("_", "-")
        help_text = _annotation_help(param)

        if param.default is inspect.Parameter.empty and param_name not in defaults:
            # Required positional argument. dest must remain the parameter name so the parsed
            # value can be passed back as a keyword argument.
            parser.add_argument(param_name, metavar=param_cli_name, type=_parse_value, help=help_text)
        elif param.default is inspect.Parameter.empty:
            # Required in the signature, but parameters.yml provides a default
            parser.add_argument(
                param_name,
                metavar=param_cli_name,
                nargs="?",
                default=argparse.SUPPRESS,
                type=_parse_value,
                help=help_text,
            )
        elif isinstance(default := defaults.get(param_name, param.default), bool):
            # Boolean defaults become a --flag/--no-flag pair rather than taking a value.
            parser.add_argument(
                f"--{param_cli_name}", action=argparse.BooleanOptionalAction, default=default, help=help_text
            )
        else:
            # Optional argument with default
            parser.add_argument(f"--{param_cli_name}", default=argparse.SUPPRESS, type=_parse_value, help=help_text)

    if var_keyword is not None:
        # Arbitrary keyword arguments are accepted as trailing 'key=value' pairs. parameters.yml
        # defaults that don't match a named parameter reach **kwargs through the partial, and the
        # command line can override them since later keyword arguments win.
        parser.add_argument(
            _VAR_KEYWORDS,
            nargs="*",
            metavar="key=value",
            type=_keyword_pair,
            default=[],
            help=f"additional keyword arguments passed as **{var_keyword}",
        )


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

    try:
        run_module = project.run

        # We create a new parser because we don't want any of the old arguments to pollute the unparsed_args with
        # their default values (specifically "no_defaults").
        new_parser = argparse.ArgumentParser(prog="aeq run", description="Run module functions", allow_abbrev=False)
        subparsers = new_parser.add_subparsers(title="functions", metavar="function", required=True)

        # For each function we'll inspect the signature and create a sort of "best guess" set of arguments to accept.
        # POSITIONAL_ONLY and VAR_POSITIONAL arguments are not supported because we supply everything as keyword
        # arguments. VAR_KEYWORD arguments are accepted as trailing 'key=value' pairs.
        for func in run_module:
            add_subcommand_from_function(subparsers, func.func, func.keywords if not args.no_defaults else {})

        # The parsed arguments contain internal bookkeeping entries from the .set_defaults trick: the run module
        # function to call and its **kwargs handling. _extract_target_call() separates those from the real keyword
        # arguments, which contain only what the user typed. parameters.yml defaults are partially applied to the
        # target function itself, just like project.run does.
        target_function, kwargs = _extract_target_call(vars(new_parser.parse_args(args=unparsed_args)))

        res = target_function(**kwargs)
        if res is not None:
            pprint(res)
    finally:
        project.close()


def cli(argv=None):
    """Entry point for the aeq command. argv defaults to sys.argv, it is a parameter for testing."""
    # Create a parser that just parses the project dir. This is global and used by every sub-command.
    # allow_abbrev is disabled here and below because unrecognised arguments are handed over to the per-function
    # parsers built in run(); prefix matching would let e.g. "--proj" silently bind to "--project".
    parser = argparse.ArgumentParser(description="AequilibraE project command line tool", allow_abbrev=False)
    parser.add_argument(
        "-p",
        "--project",
        default=os.getcwd(),
        type=str,
        help="Path to the project folder (defaults to the current directory)",
    )

    try:
        from importlib.metadata import version

        aeq_version = version("aequilibrae")
    except Exception:
        aeq_version = "unknown"
    parser.add_argument("--version", action="version", version=f"aeq (AequilibraE {aeq_version})")

    # We'll add our sub-commands via a sub-parser. The function corresponding to the sub-command is set via the
    # .set_defaults trick given in the docs
    # https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_subparsers
    subparsers = parser.add_subparsers(title="commands", help="Available run commands", required=True)

    # add_help is disabled so that a --help after the function name is left in unparsed_args for the per-function
    # parser built in run(), e.g. "aeq -p dir run my_func --help" prints the help of my_func. "aeq -p dir run --help"
    # still works: the parser built in run() handles it and lists the available functions.
    run_parser = subparsers.add_parser("run", help="Run project commands", add_help=False, allow_abbrev=False)
    run_parser.add_argument(
        "--no-defaults", action="store_true", help="do not use default arguments from parameters.yml"
    )
    run_parser.set_defaults(_aeq_internal_func=run)

    list_functions_parser = subparsers.add_parser("list", help="List project commands")
    list_functions_parser.set_defaults(_aeq_internal_func=functools.partial(list_functions, list_functions_parser))

    args, unparsed_args = parser.parse_known_args(argv)

    # args now contains the project and the "_aeq_internal_func" to run from .set_defaults. unparsed_args should
    # contain everything else.
    args._aeq_internal_func(args, unparsed_args)
