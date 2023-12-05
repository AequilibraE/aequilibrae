import os
import shutil

from aequilibrae.log import logger

from aequilibrae.project.project_creation import initialize_tables
from aequilibrae.reference_files import spatialite_database
from aequilibrae.transit.lib_gtfs import GTFSRouteSystemBuilder
from aequilibrae.transit.transit_graph_builder import SF_graph_builder
from aequilibrae.paths.graph import Graph
from aequilibrae.project.database_connection import database_connection
import sqlite3


class Transit:
    default_capacities = {
        0: [150, 300],  # Tram, Streetcar, Light rail
        1: [280, 560],  # Subway/metro
        2: [700, 700],  # Rail
        3: [30, 60],  # Bus
        4: [400, 800],  # Ferry
        5: [20, 40],  # Cable tram
        11: [30, 60],  # Trolleybus
        12: [50, 100],  # Monorail
        "other": [30, 60],
    }
    graphs: dict[str, Graph] = {}
    pt_con: sqlite3.Connection

    def __init__(self, project):
        """
        :Arguments:
             **project** (:obj:`Project`, optional): The Project to connect to. By default, uses the currently
            active project
        """

        self.project_base_path = project.project_base_path
        self.logger = logger
        self.__transit_file = os.path.join(project.project_base_path, "public_transport.sqlite")

        self.create_transit_database()
        self.pt_con = database_connection("transit")

    def new_gtfs_builder(self, agency, file_path, day="", description="") -> GTFSRouteSystemBuilder:
        """Returns a GTFSRouteSystemBuilder object compatible with the project

        :Arguments:
            **agency** (:obj:`str`): Name for the agency this feed refers to (e.g. 'CTA')

            **file_path** (:obj:`str`): Full path to the GTFS feed (e.g. 'D:/project/my_gtfs_feed.zip')

            **day** (:obj:`str`, *Optional*): Service data contained in this field to be imported (e.g. '2019-10-04')

            **description** (:obj:`str`, *Optional*): Description for this feed (e.g. 'CTA2019 fixed by John Doe')

        :Return:
            **gtfs_feed** (:obj:`StaticGTFS`): A GTFS feed that can be added to this network
        """
        gtfs = GTFSRouteSystemBuilder(
            network=self.project_base_path,
            agency_identifier=agency,
            file_path=file_path,
            day=day,
            description=description,
            default_capacities=self.default_capacities,
        )
        return gtfs

    def create_transit_database(self):
        """Creates the public transport database"""
        if not os.path.exists(self.__transit_file):
            shutil.copyfile(spatialite_database, self.__transit_file)
            initialize_tables(self, "transit")

    def create_graph(self, **kwargs):
        graph = SF_graph_builder(self.pt_con, **kwargs)
        graph.create_vertices()
        graph.create_edges()
        self.graphs[kwargs.get("period_id", 1)] = graph

    def save(self):
        for graph in self.graphs.values():
            graph.save()

    def load(self, period_ids: list[int] = None):
        if period_ids is None:
            period_ids = [period[0] for period in self.pt_con.execute("SELECT period_id FROM periods;").fetchall()]

        for period_id in period_ids:
            self.graphs[period_id] = SF_graph_builder.from_db(period_id=period_id)
