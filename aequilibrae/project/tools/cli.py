import argparse
import inspect
import os
import sys


from aequilibrae.project import Project



def cli():
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("-p", "--project", default=os.getcwd(), type=str)

    temp_args, _ = temp_parser.parse_known_args()

    project = Project()
    project.open(temp_args.project)

    global_parser = argparse.ArgumentParser(prog="aeq", description="Load and return AequilibraE run module")
    global_parser.add_argument("-p", "--project", help="Path to the project folder", default=os.getcwd(), type=str)
    subparsers = global_parser.add_subparsers(dest="action", help="Available actions")

    run_parser = subparsers.add_parser("run", help="Run project commands")
    run_subparsers = run_parser.add_subparsers(
        title="commands", dest="command", help="Available run commands", required=False
    )

    for idx, name in enumerate(project.run._fields):
        parser = run_subparsers.add_parser(name, help=f"Run {name}")
        sig = inspect.signature(project.run[idx])

        for param_name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                arg_type = param.annotation
            elif param.default != inspect.Parameter.empty and param.default is not None:
                arg_type = type(param.default)
            else:
                arg_type = str

            if param.default == inspect.Parameter.empty:
                parser.add_argument(param_name, type=arg_type)
            else:
                parser.add_argument(f"--{param_name}", default=param.default, type=arg_type)

        parser.set_defaults(func=project.run[idx])

    args = global_parser.parse_args()

    if args.action == "run" and args.command is not None and hasattr(args, "func"):
        func_sig = inspect.signature(args.func)
        kwargs = {k: v for k, v in vars(args).items() if k in func_sig.parameters}
        result = args.func(**kwargs)

        if result is not None:
            print(result)
    else:
        global_parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    cli()
