import argparse

import pytest

from aequilibrae.project import Project
from aequilibrae.project.tools.cli import _extract_target_call, add_subcommand_from_function, cli


def parse_call(func, argv, defaults=None):
    """Build a parser for func, parse argv, call the resulting target, and return what func received."""
    parser = argparse.ArgumentParser()
    add_subcommand_from_function(parser.add_subparsers(), func, defaults or {})

    target, kwargs = _extract_target_call(vars(parser.parse_args([func.__name__, *argv])))
    return target(**kwargs)


def test_signature_mapping():
    def f(first_arg, second_arg: int = 2):
        return {"first_arg": first_arg, "second_arg": second_arg}

    # Required parameters are positional, defaulted ones are --options, values parse as Python literals
    assert parse_call(f, ["1"]) == {"first_arg": 1, "second_arg": 2}
    assert parse_call(f, ["text", "--second-arg", "5"]) == {"first_arg": "text", "second_arg": 5}
    with pytest.raises(SystemExit):
        parse_call(f, [])  # first_arg is required


def test_parameters_yml_defaults():
    def f(a: int, zone: str = "x"):
        return {"a": a, "zone": zone}

    # A falsy default still makes the positional optional; string defaults must not be re-typed
    assert parse_call(f, [], defaults={"a": 0, "zone": "101"}) == {"a": 0, "zone": "101"}
    assert parse_call(f, ["7"], defaults={"a": 0}) == {"a": 7, "zone": "x"}


def test_bool_defaults_become_flag_pairs():
    def f(overwrite=False):
        return overwrite

    assert parse_call(f, []) is False
    assert parse_call(f, ["--overwrite"]) is True
    assert parse_call(f, ["--no-overwrite"], defaults={"overwrite": True}) is False


def test_var_keyword_pairs():
    def f(a="d", **kwargs):
        return {"a": a, **kwargs}

    assert parse_call(f, ["--a", "v", "x=1", "y=[1,2]", "e=a=b"]) == {"a": "v", "x": 1, "y": [1, 2], "e": "a=b"}
    # yml defaults that match no named parameter flow into **kwargs and are overridable from the command line
    assert parse_call(f, ["extra=6"], defaults={"a": "y", "extra": 5}) == {"a": "y", "extra": 6}
    with pytest.raises(SystemExit):
        parse_call(f, ["novalue"])  # malformed pair


def test_string_escape():
    def f(a: int | str = 1):
        return {"a": a}

    assert parse_call(f, ["--a", "1"]) == {"a": 1}
    assert parse_call(f, ["--a", "'1'"]) == {"a": "1"}


def test_no_positional_only_args():
    def f(a: int = 1, /):
        return {"a": a}

    with pytest.raises(ValueError, match="positional-only parameters"):
        parse_call(f, ["--a", "1"])


def test_functions_without_var_keyword_reject_pairs():
    def f(a="d"):
        return a

    with pytest.raises(SystemExit):
        parse_call(f, ["k=1"])


def test_help_shows_docstring_and_annotations(capsys):
    def f(zone: int = 1):
        """A very identifiable docstring."""
        return zone

    parser = argparse.ArgumentParser()
    add_subcommand_from_function(parser.add_subparsers(), f, {})
    with pytest.raises(SystemExit):
        parser.parse_args(["f", "--help"])

    out = capsys.readouterr().out
    assert "A very identifiable docstring." in out
    assert "type: int" in out


@pytest.fixture(scope="module")
def project_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("cli") / "project"
    project = Project()
    project.new(str(path))
    project.close()
    return str(path)


def test_help_reaches_the_function_parser(project_path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli(["-p", project_path, "run", "example_function_with_kwargs", "--help"])

    assert exc_info.value.code == 0
    assert "An example function to demonstrate" in capsys.readouterr().out


def test_run_help_lists_functions(project_path, capsys):
    with pytest.raises(SystemExit):
        cli(["-p", project_path, "run", "--help"])

    out = capsys.readouterr().out
    assert "example_function_with_kwargs" in out
    assert "matrix_summary" in out


def test_run_function(project_path, capsys):
    cli(["-p", project_path, "run", "example_function_with_kwargs", "--arg1", "hello", "extra=1"])

    out = capsys.readouterr().out
    assert "arg1: hello" in out
    assert "kwargs: {'extra': 1}" in out


def test_run_defaults(project_path, capsys):
    cli(["-p", project_path, "run", "example_function_with_kwargs"])
    assert "arg1: parameters.yml argument" in capsys.readouterr().out

    cli(["-p", project_path, "run", "--no-defaults", "example_function_with_kwargs"])
    assert "arg1: default argument" in capsys.readouterr().out


def test_list(project_path, capsys):
    cli(["-p", project_path, "list"])

    assert "example_function_with_kwargs" in capsys.readouterr().out
