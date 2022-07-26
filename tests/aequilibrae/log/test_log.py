from logging import FileHandler
import os
import uuid
from shutil import copytree
from tempfile import gettempdir
from unittest import TestCase

import pytest

from ...data import siouxfalls_project
from aequilibrae.project import Project
from aequilibrae import log
from aequilibrae.log import _setup_logger


class TestLog(TestCase):
    def setUp(self) -> None:
        self.proj_dir = os.path.join(gettempdir(), uuid.uuid4().hex)
        copytree(siouxfalls_project, self.proj_dir)

        self.project = Project()
        self.project.open(self.proj_dir)

    def tearDown(self) -> None:
        self.project.close()

    def test_contents(self):
        log = self.project.log()
        cont = log.contents()
        self.assertEqual(len(cont), 4, "Returned the wrong amount of data from the log")

    def test_clear(self):
        log = self.project.log()
        log.clear()

        with open(os.path.join(self.proj_dir, "aequilibrae.log"), "r") as file:
            q = file.readlines()
        self.assertEqual(len(q), 0, "Failed to clear the log file")


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

    def test_logger_has_single_handler_named_aquilibrae(self):
        _setup_logger()
        logger = _setup_logger()
        assert len(self.get_handlers(logger)) == 1

    def test_default_logger_handler(self):
        logger = _setup_logger()
        assert self.get_logger_file(logger).endswith("aequilibrae.log")

    def test_project_logger(self, create_project):
        project = create_project()
        assert self.get_logger_file(project.logger).startswith(project.project_base_path)

    def test_activate_project_leaves_global_logger_intact(self, create_project):
        project = create_project()
        assert self.get_logger_file(log.global_logger) != self.get_logger_file(project.logger)

    def test_multiple_projects_have_separate_logger(self, create_project):
        a = create_project()
        b = create_project()
        assert self.get_logger_file(a.logger) != self.get_logger_file(b.logger)
