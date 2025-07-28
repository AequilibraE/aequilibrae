import os
from logging import FileHandler

import pytest


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


class TestStartsLogging:
    @pytest.fixture
    def project(self, create_project):
        return create_project()

    @staticmethod
    def get_handlers(logger, name="aequilibrae"):
        return [h for h in logger.handlers if h.name == name]

    @classmethod
    def get_logger_file(cls, logger, name="aequilibrae") -> str:
        handlers = cls.get_handlers(logger, name)
        if not handlers:
            raise ValueError(f"Logger has no handlers named {name}")
        handler = handlers[0]
        if not isinstance(handler, FileHandler):
            raise TypeError(f"Handler must be FileHandler, not {type(handler).__name__}")

        return handlers[0].baseFilename

    def test_project_logger(self, create_project):
        project = create_project()
        assert self.get_logger_file(project.logger).startswith(str(project.project_base_path))

    def test_multiple_projects_have_separate_logger(self, create_project):
        a = create_project()
        b = create_project()
        assert self.get_logger_file(a.logger) != self.get_logger_file(b.logger)
