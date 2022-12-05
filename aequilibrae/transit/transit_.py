from os.path import join, dirname, realpath, isfile

import pandas as pd

from polarislib.utils.database.db_utils import run_sql_file
from polarislib.network.database_connection import supply_database_connection, get_srid
from polarislib.network.starts_logging import logger
from polarislib.network.transit.lib_gtfs import GTFSRouteSystemBuilder
from polarislib.network.utils.worker_thread import WorkerThread
from polarislib.utils.signals import SIGNAL
from polarislib.utils.signals import qgis
from . import RouteSystem
from .functions import fix_connections_table
from .functions.del_pattern import delete_pattern
from polarislib.utils.database.db_utils import list_tables_in_db
from polarislib.network.traffic import Intersection
from ..data import DataTableStorage


class Transit(WorkerThread):
    """Polaris network transit class

    One example of how to do a complete network import is below.
    Note that building the active network is necessary after importing transit

    ::

        from time import perf_counter
        from os.path import join
        import sqlite3
        import pandas as pd
        from datetime import timedelta
        from polarislib.network import Network
        from polarislib.utils.database.db_utils import read_and_close


        t = perf_counter()
        root = 'D:/Argonne/GTFS/CHICAGO'
        network_path = join(root, 'chicago2018-Supply.sqlite')

        df = pd.read_csv('D/Argonne/Polaris/Networks/supporting_data/transit_max_speeds.csv')
        max_speeds = df[df.city == 'Chicago']

        my_network = Network()
        my_network.open(network_path)

        transit = my_network.get_transit()

        # True means it will delete the actual transit tables and recreate them according to the standard
        transit.purge(True)

        # You can also disable the Python progress bars in case you are importing too many GTFS feeds
        # at the same time
        transit.set_progress_bar(False)

        my_network.conn.execute('VACUUM;')

        metra = transit.new_gtfs(file_path=join(root, 'METRA', '2019-10-04.zip'),
                                            description='METRA Commuter Rail',
                                            agency_id='METRA')

        pace = transit.new_gtfs(file_path=join(root, 'PACE', '2019-09-12.zip'),
                                           description='PACE Suburban Bus',
                                           agency_id='PACE')

        ssl = transit.new_gtfs(file_path=join(root, 'SSL', '2017-07-01.zip'),
                                          description='South Shore Line',
                                          agency_id='SSL')

        cta = transit.new_gtfs(file_path=join(root, 'CTA', '2019-10-04.zip'),
                                          description='Chicago Transit Authority',
                                          agency_id='CTA')

        for feed in [metra, ssl, cta, pace]:
            feed.set_maximum_speeds(max_speeds)
            feed.set_allow_map_match(False)
            feed.set_date('2019-10-08')
            feed.set_do_raw_shapes(False)
            feed.load_date('2019-10-08')
            feed.save_to_disk()

        # Blanket values is to guarantee that capacities will exist
        for rt in range(13):
            transit.set_capacity_by_route_type(route_type=rt, seated=40, total=80)

        # We can refine for each route type as well
        transit.set_capacity_by_route_type(route_type=1, seated=272, total=984)
        transit.set_capacity_by_route_type(route_type=2, seated=909, total=909)
        transit.set_capacity_by_route_type(route_type=3, seated=36, total=82)

        # Or get the information from another database and insert by route number
        with read_and_close(join(root, 'chicago2018-Supply_base.sqlite')) as conn:
            curr = conn.cursor()
            curr.execute('Select route, seated_capacity, design_capacity, total_capacity from Transit_Routes;')

            for route_id, seated, design, total in curr.fetchall():
                transit.set_capacity_by_route_id(route_id=route_id, seated=seated, total=total, design=design)

        # Now we build the active links network
        active = my_network.get_walk()
        active.build()

        my_network.conn.execute("update transit_stops set has_parking = 1 where stop like 'M-%' or stop like 'S-%';")

        my_network.conn.commit()
        my_network.close()

        print(f"Total run time for this script: {timedelta(seconds=perf_counter() - t)}")
    """

    transit = SIGNAL(object)
    default_capacities = {
        0: [150, 300, 300],  # Tram, Streetcar, Light rail
        1: [280, 560, 560],  # Subway/metro
        2: [700, 700, 700],  # Rail
        3: [30, 60, 60],  # Bus
        4: [400, 800, 800],  # Ferry
        5: [20, 40, 40],  # Cable tram
        11: [30, 60, 60],  # Trolleybus
        12: [50, 100, 100],  # Monorail
        "other": [30, 60, 60],
    }  # Any other mode that is not default GTFS

    def __init__(self, network) -> None:
        """Instantiates a transit class for the network

        Args:
            *network* (:obj:`Network`): Network to which this transit instance refers to
        """
        WorkerThread.__init__(self, None)
        self.srid = get_srid()
        self.network = network

    def set_default_capacities(self, route_type: int, seated=None, design=None, total=None):
        """Allows the user to define the default transit capacities per vehicle/route type prior to import

        Args:
            *route_type* (:obj:`int`): Route mode as defined in GTFS (e.g. Bus=3)
            *seated* (:obj:`int`, *Optional*): Number of seated passengers possible
            *design* (:obj:`int`, *Optional*): Number of passengers possible by design
            *total* (:obj:`int`, *Optional*): Number of TOTAL passengers possible
        """

        self.default_capacities[route_type] = self.default_capacities.get(route_type, self.default_capacities["other"])
        for i, val in enumerate([seated, design, total]):
            self.default_capacities[route_type][i] = self.default_capacities[route_type][i] if val is None else val

    def purge(self, hard_reset=False) -> None:
        """We remove all the transit service information from the database

        **The tables cleared are**: transit_links, transit_walk, transit_stops,
        transit_fare_attributes, transit_fare_rules, transit_pattern_mapping,
        Transit_Pattern_Links, transit_patterns, transit_routes,transit_trips,
        transit_trips_schedule, transit_zones,transit_agencies, TRANSIT_RAW_SHAPES

        Args:
            *hard_reset* (:obj:`bool`, *Optional*): Excludes tables from database and recreates them according to the
                                                    standard (Default is False)
        """
        to_clear = [
            "transit_links",
            "transit_walk",
            "transit_fare_attributes",
            "transit_fare_rules",
            "transit_zones",
            "transit_agencies",
            "transit_pattern_mapping",
            "transit_pattern_links",
            "transit_patterns",
            "transit_routes",
            "transit_trips",
            "transit_trips_schedule",
            "transit_bike",
            "transit_stops",
            "TRANSIT_RAW_SHAPES",
        ]

        with supply_database_connection() as conn:
            curr = conn.cursor()
            table_list = list_tables_in_db(curr)
            if hard_reset:
                logger.warning("Performing a table hard-reset")
                curr.execute("PRAGMA foreign_keys=OFF;")
                for table in to_clear:
                    logger.warning(f"    {table}")
                    if table.lower() in table_list:
                        curr.execute(f"DROP Table if exists {table};")
                        curr.execute(f'DELETE FROM geometry_columns where f_table_name="{table}";')
                    logger.debug(f"    Creating table: {table}")
                    qry_file = join(dirname(dirname(realpath(__file__))), "database/sql_schema", f"{table}.sql")
                    if isfile(qry_file):
                        run_sql_file(qry_file=qry_file, curr=curr, replacements={"SRID_PARAMETER": self.srid})
                    else:
                        logger.warning(f"    Table {table} was not re-created for lack of specification")

                curr.execute("PRAGMA foreign_keys=ON;")
            else:
                logger.warning("Clearing transit tables")
                for table in to_clear:
                    logger.warning(f"    {table}")
                    if table.lower() in table_list:
                        curr.execute(f'DELETE from "{table}"')
                        logger.debug(f'Table "{table}" cleared')
                    else:
                        logger.error(f'Table "{table}" does not exist in the network')

        self.__purge_added_connections(conn)
        conn.close()

    def new_gtfs(self, agency, file_path, day="", description="") -> GTFSRouteSystemBuilder:
        """Returns a GTFSRouteSystemBuilder object compatible with the project

        Args:
            *agency* (:obj:`str`): Name for the agency this feed refers to (e.g. 'CTA')

            *file_path* (:obj:`str`): Full path to the GTFS feed (e.g. 'D:/project/my_gtfs_feed.zip')

            *day* (:obj:`str`, *Optional*): Service data contained in this field to be imported (e.g. '2019-10-04')

            *description* (:obj:`str`, *Optional*): Description for this feed (e.g. 'CTA2019 fixed by John Doe')

        Return:
            *gtfs_feed* (:obj:`StaticGTFS`): A GTFS feed that can be added to this network
        """
        gtfs = GTFSRouteSystemBuilder(
            network=self.network,
            agency_identifier=agency,
            file_path=file_path,
            day=day,
            description=description,
            default_capacities=self.default_capacities,
        )
        if not qgis:
            gtfs.signal = self.transit
            gtfs.gtfs_data.signal = self.transit
        return gtfs

    def set_capacity_by_route_type(self, route_type: int, seated: int, total: int, design=None):
        """Sets vehicle capacities for routes with a certain route type/mode

        Args:
            *route_type* (:obj:`int`): Route mode as defined in GTFS (e.g. Bus=3)

            *seated* (:obj:`int`): Number of seated passengers possible

            *total* (:obj:`int`): Number of TOTAL passengers possible

            *design* (:obj:`int`, *Optional*): Number of passengers possible by design. Defaults to total
        """
        conn = supply_database_connection()
        if design is None:
            design = total
        conn.execute(
            """UPDATE  Transit_Routes  set seated_capacity=?, design_capacity=?, total_capacity=?
                        where "type"=?""",
            [seated, design, total, route_type],
        )
        conn.commit()
        conn.close()

    def set_capacity_by_route_id(self, route: str, seated: int, total: int, design=None, number_of_cars=0):
        """Sets vehicle capacities for an specific route_id

        Args:
            *route* (:obj:`str`): Route_id as present in the GTFS databasez

            *seated* (:obj:`int`): Number of seated passengers possible

            *total* (:obj:`int`): Number of TOTAL passengers possible

            *design* (:obj:`int`, *Optional*): Number of passengers possible by design. Defaults to total

            *number_of_cars* (:obj:`int`, *Optional*): Number of cars operating the route (applies to trains)
        """
        conn = supply_database_connection()
        if design is None:
            design = total
        conn.execute(
            """UPDATE  Transit_Routes  set seated_capacity=?, design_capacity=?, total_capacity=?,
                        number_of_cars=? where "route"=?""",
            [seated, design, total, number_of_cars, route],
        )
        conn.commit()
        conn.close()

    def delete_route(self, route_id: int):
        """Deletes all information regarding one specific route

        Args:
            *route_id* (:obj:`str`): Route_id as present in the database
        """
        conn = supply_database_connection()
        patterns = [x[0] for x in conn.execute("Select pattern_id from Transit_Patterns where route_id=?", [route_id])]
        for pat in patterns:
            delete_pattern(pat)

    def delete_pattern(self, pattern_id: int):
        """Deletes all information regarding one specific transit_pattern

        Args:
            *pattern_id* (:obj:`str`): pattern_id as present in the database
        """
        delete_pattern(pattern_id)

    def fix_connections_table(self):
        """Adds connections that do not exist"""
        with supply_database_connection() as conn:
            data_tables = DataTableStorage()
            connections = data_tables.get_table("Connection", conn)
            map_matching = data_tables.get_table("Transit_Pattern_Mapping", conn).reset_index()
            bus_pat_sql = """SELECT pattern_id from Transit_Patterns tp
                             INNER JOIN transit_routes tr ON tp.route_id=tr.route_id
                             WHERE tr.type=3"""
            patterns = pd.read_sql(bus_pat_sql, conn)
            map_matching = map_matching[map_matching.pattern_id.isin(patterns.pattern_id)]
            fix_connections_table(connections, map_matching, conn)

    def export_gtfs(self, output_folder: str):
        """Exports transit system to GTFS output

        Args:
            *output_folder* (:obj:`str`): Folder where the GTFS files will be created
        """
        rs = RouteSystem(self.network.path_to_file)
        rs.load_route_system()
        rs.write_GTFS(output_folder)

    def set_progress_bar(self, progress_bar: bool):
        """Controls display of progress bars in Python

        Args:
            *progress_bar* (:obj:`bool`): Removes progress bars from Python when set to False
        """
        self.transit.deactivate = not progress_bar

    def __purge_added_connections(self, conn):
        nodes_qry = "SELECT distinct(node) from Turn_Overrides where notes='required_by_pt_map_matching'"
        nodes = [x[0] for x in conn.execute(nodes_qry).fetchall()]
        conn.execute("Delete from Turn_Overrides where notes='required_by_pt_map_matching'")

        for n in nodes:
            intersec = Intersection(data_tables=DataTableStorage(), conn=conn)
            intersec.load(n)
            intersec.rebuild_intersection()

        conn.commit()
