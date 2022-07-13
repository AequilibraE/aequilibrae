from unittest import TestCase
from tempfile import gettempdir
import os
import uuid
import pandas as pd
from aequilibrae.project import Project
from aequilibrae.parameters import Parameters
from warnings import warn
import random
from ...data import gmns_link, gmns_node, gmns_groups


class TestNetwork(TestCase):
    def test_create_from_gmns(self):

        proj_path = os.path.join(gettempdir(), uuid.uuid4().hex)
        self.project = Project()
        self.project.new(proj_path)

        new_link_fields = {
            'bridge': {'description': 'bridge flag', 'type': 'text'},
            'tunnel': {'description': 'tunnel flag', 'type': 'text'}
        }
        new_node_fields = {
            'port': {'description': 'port flag', 'type': 'text'},
            'hospital': {'description': 'hoospital flag', 'type': 'text'}
        }
        par = Parameters()
        par.parameters['network']['gmns']['other_link_fields'].update(new_link_fields)
        par.parameters['network']['gmns']['other_node_fields'].update(new_node_fields)
        par.write_back()

        self.project.network.create_from_gmns(gmns_link, gmns_node, gmns_groups, srid=32619)

        gmns_node_df = pd.read_csv(gmns_node)
        gmns_link_df = pd.read_csv(gmns_link)

        curr = self.project.conn.cursor()
        curr.execute("""select count(*) from nodes""")
        nd_ct = curr.fetchone()[0]

        if nd_ct != gmns_node_df.shape[0]:
            warn("Number of nodes created is different than expected.")
            return

        rand_lk = random.choice([x[0] for x in curr.execute("""select link_id from links""").fetchall()])
        from_node = gmns_link_df.loc[gmns_link_df.link_id == rand_lk, "from_node_id"].item()
        to_node = gmns_link_df.loc[gmns_link_df.link_id == rand_lk, "to_node_id"].item()
        a_node = curr.execute(f"""select a_node from links where link_id = {rand_lk}""").fetchone()[0]
        b_node = curr.execute(f"""select b_node from links where link_id = {rand_lk}""").fetchone()[0]

        if from_node != a_node or to_node != b_node:
            self.fail("At least one link is disconnected from its start/end nodes")
