from os.path import join

from aequilibrae.project.database_connection import database_connection
from aequilibrae.transit.lib_gtfs import GTFSRouteSystemBuilder


class Transit:
    def __init__(self, project):
        self.conn = database_connection(join(project.project_base_path, "public_transport.sqlite"))
        self.project_base_path = project.project_base_path

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
            network=self.project_base_path,
            agency_identifier=agency,
            file_path=file_path,
            day=day,
            description=description,
            default_capacities=self.default_capacities,
        )
        return gtfs