import os
import shutil
import sqlite3
import warnings
from typing import Dict, List

import pandas as pd

from aequilibrae.log import logger
from aequilibrae.paths.graph import TransitGraph
from aequilibrae.project.project_creation import initialize_tables
from aequilibrae.reference_files import spatialite_database
from aequilibrae.transit.lib_gtfs import GTFSRouteSystemBuilder
from aequilibrae.transit.transit_graph_builder import TransitGraphBuilder
from aequilibrae.utils.aeq_signal import SIGNAL
from aequilibrae.utils.db_utils import read_and_close
from aequilibrae.utils.get_table import get_geo_table
from aequilibrae.utils.interface.worker_thread import WorkerThread


class Transit(WorkerThread):
    transit = SIGNAL(object)
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
    default_pces = {0: 5.0, 1: 5.0, 3: 4.0, 5: 4.0, 11: 3.0, "other": 2.0}
    graphs: Dict[str, TransitGraph] = {}
    pt_con: sqlite3.Connection

    def __init__(self, project):
        """
        :Arguments:
            **project** (:obj:`Project`, *Optional*): The Project to connect to. By default, uses the currently
            active project
        """
        super().__init__(None)

        self.project = project
        self.logger = logger
        self.periods = project.network.periods

        self.create_transit_database()

    def get_table(self, table_name) -> pd.DataFrame:
        with self.project.transit_connection as conn:
            return get_geo_table(table_name, conn)

    def new_gtfs_builder(self, agency, file_path, day="", description="") -> GTFSRouteSystemBuilder:
        """Returns a ``GTFSRouteSystemBuilder`` object compatible with the project

        :Arguments:
            **agency** (:obj:`str`): Name for the agency this feed refers to (e.g. 'CTA')

            **file_path** (:obj:`str`): Full path to the GTFS feed (e.g. 'D:/project/my_gtfs_feed.zip')

            **day** (:obj:`str`, *Optional*): Service data contained in this field to be imported (e.g. '2019-10-04')

            **description** (:obj:`str`, *Optional*): Description for this feed (e.g. 'CTA2019 fixed by John Doe')

        :Returns:
            **gtfs_feed** (:obj:`StaticGTFS`): A GTFS feed that can be added to this network
        """
        gtfs = GTFSRouteSystemBuilder(
            network=self.project.project_base_path,
            agency_identifier=agency,
            file_path=file_path,
            day=day,
            description=description,
            capacities=self.default_capacities,
            pces=self.default_pces,
        )

        gtfs.signal = self.transit
        gtfs.gtfs_data.signal = self.transit
        return gtfs

    def create_transit_database(self):
        """Creates the public transport database"""
        if not os.path.exists(self.project._transit_database_path):
            shutil.copyfile(spatialite_database, self.project._transit_database_path)
            with self.project.transit_connection as conn:
                initialize_tables(self.logger, "transit", conn=conn)

    def create_graph(self, **kwargs) -> TransitGraphBuilder:
        """
        Create a transit graph from an existing GTFS import.

        All arguments are forwarded to 'TransitGraphBuilder'.

        A 'period_id' may be specified to select a time period. By default, a whole day is used. See
        'project.network.Periods' for more details.
        """
        period_id = kwargs.pop("period_id", self.periods.default_period.period_id)

        graph = TransitGraphBuilder(self.project, period_id, **kwargs)
        graph.create_graph()
        self.graphs[period_id] = graph
        return graph

    def save_graphs(self, period_ids: List[int] = None, force: bool = False):
        """
        Save the previously build transit graphs to the 'public_transport.sqlite' database. Saving may be filtered
        by 'period_id'.

        :Arguments:
            **period_ids** (:obj:`int`): List of periods of to save. Defaults to 'project.network.periods.default_period.period_id'.
            **force** (:obj:`bool`): Remove the existing graphs before saving the 'period_ids' graphs. Default 'False'.

        """
        if period_ids is None:
            period_ids = self.graphs.keys()

        if force:
            self.remove_graphs(period_ids)

        for period_id in period_ids:
            self.graphs[period_id].save()

    def remove_graphs(self, period_ids: List[int], unload: bool = False):
        """
        Remove the previously saved transit graphs from the 'public_transport.sqlite' database. Removing may be filtered
        by 'period_id'.

        :Arguments:
            **period_ids** (:obj:`int`): List of periods of to save.
            **unload** (:obj:`bool`): Also unload the graph.

        """
        for period_id in period_ids:
            with self.project.transit_connection as pt_conn, self.project.db_connection as project_conn:
                TransitGraphBuilder.remove(pt_conn, project_conn, period_id)
            if unload:
                del self.graphs[period_id]

    def load(self, period_ids: List[int] = None):
        """
        Load the previously saved transit graphs from the 'public_transport.sqlite' database. Loading may be filtered
        by 'period_id'.

        :Arguments:
            **period_ids** (:obj:`int`): List of periods of to load. Defaults to all available graph configurations.

        """
        if period_ids is None:
            with self.project.db_connection as conn:
                res = conn.execute("SELECT period_id FROM transit_graph_configs").fetchall()
            period_ids = [x[0] for x in res]

        for period_id in period_ids:
            self.graphs[period_id] = TransitGraphBuilder.from_db(self.project, period_id)

    def build_pt_preload(self, start: int, end: int, inclusion_cond: str = "start") -> pd.DataFrame:
        """Builds a preload vector for the transit network over the specified time period

        :Arguments:
            **start** (:obj:`int`): The start of the period for which to check pt schedules (seconds from midnight)

            **end** (:obj:`int`): The end of the period for which to check pt schedules, (seconds from midnight)

            **inclusion_cond** (:obj:`str`): Specifies condition with which to include/exclude pt trips from the
            preload.

        :Returns:
            **preloads** (:obj:`pd.DataFrame`): A DataFrame of preload from transit vehicles that can be directly used
            in an assignment

        .. code-block:: python

            >>> project = create_example(project_path, "coquimbo")

            >>> project.network.build_graphs()

            >>> start = int(6.5 * 60 * 60) # 6:30 am
            >>> end = int(8.5 * 60 * 60)   # 8:30 am

            >>> transit = Transit(project)
            >>> preload = transit.build_pt_preload(start, end)

            >>> project.close()
        """
        with self.project.transit_connection as conn:
            return pd.read_sql(self.__build_pt_preload_sql(start, end, inclusion_cond), conn)

    def __build_pt_preload_sql(self, start, end, inclusion_cond):
        probe_point_lookup = {
            "start": "MIN(departure)",
            "end": "MAX(arrival)",
            "midpoint": "(MIN(departure) + MAX(arrival)) / 2",
        }

        def select_trip_ids():
            in_period = f"BETWEEN {start} AND {end}"
            if inclusion_cond == "any":
                return f"SELECT DISTINCT trip_id FROM trips_schedule WHERE arrival {in_period} OR departure {in_period}"
            return f"""
                SELECT trip_id FROM trips_schedule GROUP BY trip_id
                HAVING {probe_point_lookup[inclusion_cond]} {in_period}
            """

        # Convert trip_id's to link/dir's via pattern_id's
        return f"""
            SELECT pm.link as link_id, pm.dir as direction, SUM(r.pce) as preload
            FROM (SELECT pattern_id FROM trips WHERE trip_id IN ({select_trip_ids()})) as p
            INNER JOIN pattern_mapping pm ON p.pattern_id = pm.pattern_id
            INNER JOIN routes r ON p.pattern_id = r.pattern_id
            GROUP BY pm.link, pm.dir
        """
