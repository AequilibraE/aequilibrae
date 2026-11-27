import os
import random
from warnings import warn

import pandas as pd
import pytest


def test_create_from_gmns(empty_project, test_data_path):
    arlington = test_data_path / "GMNS" / "examples" / "Arlington_Signals"
    link_file = arlington / "link.csv"
    node_file = arlington / "node.csv"
    use_group_file = arlington / "use_group.csv"

    new_link_fields = {
        "bridge": {"description": "bridge flag", "type": "text", "required": False},
        "tunnel": {"description": "tunnel flag", "type": "text", "required": False},
    }
    new_node_fields = {
        "port": {"description": "port flag", "type": "text", "required": False},
        "hospital": {"description": "hoospital flag", "type": "text", "required": False},
    }
    par = empty_project.project_parameters
    par.parameters["network"]["gmns"]["link"]["fields"].update(new_link_fields)
    par.parameters["network"]["gmns"]["node"]["fields"].update(new_node_fields)
    par.write_back()

    empty_project.tables.importer.gmns(
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


def test_gmns_capacity_becomes_a_link_field(empty_project, test_data_path):
    """Test that importing GMNS links adds a capacity field, which new projects lack, and fills it and the lanes."""
    arlington = test_data_path / "GMNS" / "examples" / "Arlington_Signals"
    empty_project.tables.importer.gmns(
        link_file_path=arlington / "link.csv",
        node_file_path=arlington / "node.csv",
        use_group_path=arlington / "use_group.csv",
        srid=32619,
    )

    # The GMNS links leave both empty for some links
    with empty_project.db_connection as conn:
        sql = "SELECT count(capacity_ab), count(lanes_ab), sum(capacity_ab) FROM links_full"
        with_capacity, with_lanes, total_capacity = conn.execute(sql).fetchone()
    assert with_capacity > 0 and with_lanes > 0
    assert total_capacity > 0
    assert empty_project.tables.links.fields.get_description("capacity_*") == "Capacity"


def test_export_to_gmns(sioux_falls_example):
    output_path = sioux_falls_example.project_base_path
    sioux_falls_example.tables.export_to_gmns(output_path)

    assert os.path.isfile(output_path / "link.csv") is True
    assert os.path.isfile(output_path / "node.csv") is True
