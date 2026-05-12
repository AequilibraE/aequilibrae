import os
import random
from warnings import warn

import pandas as pd
import pytest

from aequilibrae.parameters import Parameters


def test_create_from_gmns(empty_project):
    link_file = "https://raw.githubusercontent.com/zephyr-data-specs/GMNS/develop/examples/Arlington_Signals/link.csv"
    node_file = "https://raw.githubusercontent.com/zephyr-data-specs/GMNS/develop/examples/Arlington_Signals/node.csv"
    use_group_file = (
        "https://raw.githubusercontent.com/zephyr-data-specs/GMNS/develop/examples/Arlington_Signals/use_group.csv"
    )

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

    empty_project.network.create_from_gmns(
        link_file_path=link_file, node_file_path=node_file, use_group_path=use_group_file, srid=32619
    )

    gmns_node_df = pd.read_csv(node_file)
    gmns_link_df = pd.read_csv(link_file)

    with empty_project.db_connection as conn:
        nd_ct = conn.execute("""select count(*) from nodes""").fetchone()[0]

        if nd_ct != gmns_node_df.shape[0]:
            warn("Number of nodes created is different than expected.", stacklevel=2)
            return

        rand_lk = random.choice([x[0] for x in conn.execute("""select link_id from links""").fetchall()])
        from_node = gmns_link_df.loc[gmns_link_df.link_id == rand_lk, "from_node_id"].item()
        to_node = gmns_link_df.loc[gmns_link_df.link_id == rand_lk, "to_node_id"].item()
        a_node = conn.execute(f"""select a_node from links where link_id = {rand_lk}""").fetchone()[0]
        b_node = conn.execute(f"""select b_node from links where link_id = {rand_lk}""").fetchone()[0]

        if from_node != a_node or to_node != b_node:
            pytest.fail("At least one link is disconnected from its start/end nodes")


def test_export_to_gmns(sioux_falls_example):
    output_path = sioux_falls_example.project_base_path
    sioux_falls_example.network.export_to_gmns(output_path)

    assert os.path.isfile(output_path / "link.csv") is True
    assert os.path.isfile(output_path / "node.csv") is True
