import os
from logging import FileHandler

from aequilibrae.project.project import Project


def test_contents(sioux_falls_test):
    log = sioux_falls_test.log()
    cont = log.contents()
    assert len(cont) == 4, "Returned the wrong amount of data from the log"


def test_clear(sioux_falls_test):
    log = sioux_falls_test.log()
    log.clear()

    proj_dir = sioux_falls_test.project_base_path
    with open(os.path.join(proj_dir, "aequilibrae.log"), "r") as file:
        q = file.readlines()
    assert len(q) == 0, "Failed to clear the log file"


def get_handlers(logger, name="aequilibrae"):
    return [h for h in logger.handlers if h.name == name]


def get_logger_file(logger, name="aequilibrae") -> str:
    handlers = get_handlers(logger, name)
    if not handlers:
        raise ValueError(f"Logger has no handlers named {name}")
    handler = handlers[0]
    if not isinstance(handler, FileHandler):
        raise TypeError(f"Handler must be FileHandler, not {type(handler).__name__}")

    return handlers[0].baseFilename


def test_project_logger(empty_project):
    assert get_logger_file(empty_project.logger).startswith(str(empty_project.project_base_path))


def test_multiple_projects_have_separate_logger(tmp_path):
    a = Project()
    a.new(tmp_path / "a")

    b = Project()
    b.new(tmp_path / "b")
    assert get_logger_file(a.logger) != get_logger_file(b.logger)
