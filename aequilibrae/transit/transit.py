import os
import shutil
import warnings

from aequilibrae.log import logger
from typing import Dict, List

from aequilibrae.project.project_creation import initialize_tables
from aequilibrae.reference_files import spatialite_database
from aequilibrae.transit.lib_gtfs import GTFSRouteSystemBuilder
from aequilibrae.transit.transit_graph_builder import TransitGraphBuilder
from aequilibrae.paths.graph import TransitGraph
from aequilibrae.project.database_connection import database_connection
from aequilibrae.utils.db_utils import read_and_close
from aequilibrae.utils.signal import SIGNAL
from aequilibrae.utils.interface.worker_thread import WorkerThread
import sqlite3
import pandas as pd


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
        WorkerThread.__init__(self, None)

        self.project_base_path = project.project_base_path
        self.logger = logger
        self.__transit_file = os.path.join(project.project_base_path, "public_transport.sqlite")
        self.periods = project.network.periods

        self.create_transit_database()
        self.pt_con = database_connection("transit")

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
            network=self.project_base_path,
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
        if not os.path.exists(self.__transit_file):
            shutil.copyfile(spatialite_database, self.__transit_file)
            initialize_tables(self, "transit")

    def create_graph(self, **kwargs) -> TransitGraphBuilder:
        period_id = kwargs.get("period_id", self.periods.default_period.period_id)
        graph = TransitGraphBuilder(self.pt_con, period_id, **kwargs)
        graph.create_graph()
        self.graphs[period_id] = graph
        return graph

    def save_graphs(self, period_ids: List[int] = None):
        # TODO: Support multiple graph saving
        warnings.warn(
            "Currently only a single transit graph can be saved and reloaded. Multiple graph support is plan for a future release."
        )
        if period_ids is None:
            period_ids = [self.periods.default_period.period_id]

        if len(period_ids) > 1:
            raise ValueError("Multiple graphs can currently be saved.")

        for period_id in period_ids:
            self.graphs[period_id].save()

    def load(self, period_ids: List[int] = None):
        # TODO: Support multiple graph loading
        warnings.warn(
            "Currently only a single transit graph can be saved and reloaded. Multiple graph support is plan for a future release. `period_ids` argument is currently ignored."
        )

        if period_ids is None:
            period_ids = [self.periods.default_period.period_id]

        if len(period_ids) > 1:
            raise ValueError("Multiple graphs can currently be loaded.")

        self.graphs[period_ids[0]] = TransitGraphBuilder.from_db(self.pt_con, period_ids[0])

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
        """
        return pd.read_sql(self.__build_pt_preload_sql(start, end, inclusion_cond), self.pt_con)

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
