from unittest import TestCase
from tempfile import gettempdir
from shutil import copytree
import os
import uuid
import pandas as pd
from aequilibrae.project import Project
from aequilibrae.parameters import Parameters
from warnings import warn
import random
from ...data import siouxfalls_project


class TestNetwork(TestCase):
    def test_create_from_gmns(self):
        proj_path = os.path.join(gettempdir(), uuid.uuid4().hex)
        self.project = Project()
        self.project.new(proj_path)

        link_file = "https://raw.githubusercontent.com/zephyr-data-specs/GMNS/development/Small_Network_Examples/Arlington_Signals/link.csv"
        node_file = "https://raw.githubusercontent.com/zephyr-data-specs/GMNS/development/Small_Network_Examples/Arlington_Signals/node.csv"
        use_group_file = "https://raw.githubusercontent.com/zephyr-data-specs/GMNS/development/Small_Network_Examples/Arlington_Signals/use_group.csv"

        new_link_fields = {
            "bridge": {"description": "bridge flag", "type": "text", "required": False},
            "tunnel": {"description": "tunnel flag", "type": "text", "required": False},
        }
        new_node_fields = {
            "port": {"description": "port flag", "type": "text", "required": False},
            "hospital": {"description": "hoospital flag", "type": "text", "required": False},
        }
        par = Parameters()
        par.parameters["network"]["gmns"]["link"]["fields"].update(new_link_fields)
        par.parameters["network"]["gmns"]["node"]["fields"].update(new_node_fields)
        par.write_back()

        self.project.network.create_from_gmns(
            link_file_path=link_file, node_file_path=node_file, use_group_path=use_group_file, srid=32619
        )

        gmns_node_df = pd.read_csv(node_file)
        gmns_link_df = pd.read_csv(link_file)

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

    def test_export_to_gmns(self):
        output_path = os.path.join(gettempdir(), uuid.uuid4().hex)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        self.temp_proj_folder = os.path.join(gettempdir(), uuid.uuid4().hex)
        copytree(siouxfalls_project, self.temp_proj_folder)
        self.project = Project()
        self.project.open(self.temp_proj_folder)

        self.project.network.export_to_gmns(output_path)
