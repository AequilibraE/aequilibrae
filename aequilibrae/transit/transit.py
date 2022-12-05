from os.path import join

from aequilibrae.project.database_connection import database_connection


class Transit:
    def __init__(self, project):
        self.conn = database_connection(join(project.project_base_path, "public_transport.sqlite"))
