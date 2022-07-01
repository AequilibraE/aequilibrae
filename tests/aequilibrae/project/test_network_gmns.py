from unittest import TestCase
import sqlite3
from tempfile import gettempdir
import os
import uuid
from shutil import copytree
import platform
from aequilibrae.project import Project
from aequilibrae.project.network.network import Network
from aequilibrae.parameters import Parameters
from os.path import join, dirname
from warnings import warn
from random import random
from aequilibrae.project.spatialite_connection import spatialite_connection
from ...data import gmns_link, gmns_node, gmns_groups


class TestNetwork(TestCase):

    def test_create_from_gmns(self):

        proj_path = os.path.join(gettempdir(), uuid.uuid4().hex)

        self.project = Project()
        self.project.new(proj_path)
        self.project.network.create_from_gmns(gmns_link, gmns_node, gmns_groups, srid=32619)
