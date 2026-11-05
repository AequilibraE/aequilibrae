"""Development commands shared by local workflows and GitHub Actions."""

import os
import re
import sys
from pathlib import Path

import click
from spin import util
from spin.cmds import meson


@util.extend_command(meson.test, doc="Run Python and reStructuredText doctests.", remove_args=("tests",))
def doctest(*, parent_callback, pytest_args, **kwargs):
    """Use the same doctest targets and exclusions as the documentation workflow."""
    args = (
        "--doctest-modules",
        "--doctest-glob=*.rst",
        "--ignore-glob=*setup_ipf.py",
        "--ignore-glob=*setup_assignment.py",
        "--ignore-glob=*database_specification*",
        "src/aequilibrae/distribution",
        "src/aequilibrae/matrix",
        "src/aequilibrae/paths",
        "src/aequilibrae/project",
        "src/aequilibrae/log.py",
        "src/aequilibrae/parameters.py",
        "src/aequilibrae/transit",
        "docs/source/aequilibrae_project/project_components",
        "docs/source/traffic_assignment",
        "docs/source/public_transport",
    )

    parent_callback(pytest_args=args + tuple(pytest_args), tests=None, **kwargs)


def _cython_lint():
    """Keep readable local output and annotate Cython diagnostics on GitHub."""
    github = os.environ.get("GITHUB_ACTIONS") == "true"
    result = util.run(["cython-lint", "./src/"], sys_exit=False, output=not github)

    if github:
        for line in result.stdout.decode("utf-8").splitlines():
            click.echo(line)

            match = re.match(r"^([^:]+):(\d+):(\d+): (.+)$", line)
            if match:
                path, row, col, message = match.groups()

                # GitHub workflow commands require escaping diagnostic text.
                message = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
                path = path.replace("%", "%25").replace(",", "%2C").replace(":", "%3A")
                click.echo(f"::error title=cython-lint,file={path},line={row},col={col}::{message}")

    return result.returncode


@click.command()
@meson.build_dir_option
def lint(*, build_dir):
    """Run Ruff lint/format checks, cython-lint, clang-tidy and clang-format, without fixing files.

    Requires clang-format, and clang-tidy when a compilation database exists.
    Warn and skip clang-tidy if the build directory has no compilation database.
    """
    ruff_args = [sys.executable, "-m", "ruff", "check"]
    ruff_format_args = [sys.executable, "-m", "ruff", "format", "--check"]
    if os.environ.get("GITHUB_ACTIONS") == "true":
        ruff_args.append("--output-format=github")
        ruff_format_args.append("--output-format=github")

    failed = util.run(ruff_args, sys_exit=False).returncode != 0
    failed |= util.run(ruff_format_args, sys_exit=False).returncode != 0
    failed |= _cython_lint() != 0

    sources = sorted(str(path) for path in Path("src").rglob("*") if path.is_file() and path.suffix in (".cpp", ".hpp"))
    if sources:
        compilation_database = Path(build_dir) / "compile_commands.json"
        if compilation_database.is_file():
            result = util.run(
                ["clang-tidy", "-p", build_dir, "--quiet", "--warnings-as-errors=*", *sources], sys_exit=False
            )
            failed |= result.returncode != 0
        else:
            click.secho(
                f"Warning: skipping clang-tidy: {compilation_database} not found. "
                "Run 'spin build' first or use -C to select an existing build.",
                fg="yellow",
                err=True,
            )

        failed |= util.run(["clang-format", "--dry-run", "--Werror", *sources], sys_exit=False).returncode != 0
    raise SystemExit(int(failed))
