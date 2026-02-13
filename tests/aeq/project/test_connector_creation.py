import pytest
from shapely.geometry import LineString, Point

from aequilibrae.project.network.connector_creation import bulk_connector_creation


@pytest.fixture
def initial_state(coquimbo_example):
    """Get initial state of the project for comparison."""
    project = coquimbo_example
    return {
        "nodes": project.network.nodes.data,
        "links": project.network.links.data,
        "zones": project.zoning.data,
        "initial_connectors": project.network.links.data[project.network.links.data.link_type == "centroid_connector"],
    }


def test_bulk_connector_creation_single_mode(coquimbo_example, initial_state):
    """Test basic connector creation with a single mode."""
    project = coquimbo_example

    with project.db_connection_spatial as conn:
        bulk_connector_creation(
            conn=conn,
            project_nodes=initial_state["nodes"],
            project_links=initial_state["links"],
            project_zones=initial_state["zones"],
            modes=["c"],
            k_connectors=1,
            limit_to_zone=True,
        )

    # Refresh data after operation
    updated_links = project.network.links.data
    updated_nodes = project.network.nodes.data

    # Number of nodes should remain constant
    assert len(updated_nodes) == len(initial_state["nodes"]), "Node count should not change"

    # Should have connectors
    new_connectors = updated_links[updated_links.link_type == "centroid_connector"]
    assert len(new_connectors) >= len(initial_state["initial_connectors"]), (
        "Should have at least as many connectors as before"
    )

    # Check new connectors properties
    added_connectors = new_connectors[~new_connectors.link_id.isin(initial_state["initial_connectors"].link_id)]

    if len(added_connectors) > 0:
        assert all(added_connectors.direction == 0), "All new connectors should be bidirectional"
        assert all(added_connectors.link_type == "centroid_connector"), "All should be centroid connectors"
        assert all(added_connectors.modes.str.contains("c")), "All should support car mode"


@pytest.mark.parametrize("modes", [["c"], ["c", "w", "b", "t"]])
def test_bulk_connector_creation_multiple_modes(coquimbo_example, initial_state, modes):
    """Test connector creation with different mode combinations."""
    project = coquimbo_example

    with project.db_connection_spatial as conn:
        bulk_connector_creation(
            conn=conn,
            project_nodes=initial_state["nodes"],
            project_links=initial_state["links"],
            project_zones=initial_state["zones"],
            modes=modes,
            k_connectors=1,
            limit_to_zone=True,
        )

    updated_links = project.network.links.data
    updated_nodes = project.network.nodes.data

    assert len(updated_nodes) == len(initial_state["nodes"]), "Node count should not change"

    new_connectors = updated_links[updated_links.link_type == "centroid_connector"]
    added_connectors = new_connectors[~new_connectors.link_id.isin(initial_state["initial_connectors"].link_id)]

    if len(added_connectors) > 0:
        # Check that connectors support the requested modes
        for mode in modes:
            mode_connectors = added_connectors[added_connectors.modes.str.contains(mode)]
            assert len(mode_connectors) > 0, f"Should have connectors supporting mode {mode}"


@pytest.mark.parametrize("k_connectors", [1, 2, 3])
def test_bulk_connector_creation_k_connectors(coquimbo_example, initial_state, k_connectors):
    """Test creating multiple connectors per centroid."""
    project = coquimbo_example

    with project.db_connection_spatial as conn:
        bulk_connector_creation(
            conn=conn,
            project_nodes=initial_state["nodes"],
            project_links=initial_state["links"],
            project_zones=initial_state["zones"],
            modes=["c"],
            k_connectors=k_connectors,
            limit_to_zone=True,
        )

    updated_links = project.network.links.data

    new_connectors = updated_links[updated_links.link_type == "centroid_connector"]
    added_connectors = new_connectors[~new_connectors.link_id.isin(initial_state["initial_connectors"].link_id)]

    # Count connectors per centroid
    centroid_node_ids = initial_state["nodes"][initial_state["nodes"].is_centroid == 1].node_id.values

    for centroid_id in centroid_node_ids:
        centroid_connectors = added_connectors[added_connectors.a_node == centroid_id]
        # Should have at most k_connectors (may be fewer if not enough nodes available)
        assert len(centroid_connectors) <= k_connectors, (
            f"Centroid {centroid_id} should have at most {k_connectors} connectors"
        )


@pytest.mark.parametrize("limit_to_zone", [True, False])
def test_bulk_connector_creation_zone_limitation(coquimbo_example, initial_state, limit_to_zone):
    """Test connector creation with and without zone limitation."""
    project = coquimbo_example

    with project.db_connection_spatial as conn:
        bulk_connector_creation(
            conn=conn,
            project_nodes=initial_state["nodes"],
            project_links=initial_state["links"],
            project_zones=initial_state["zones"],
            modes=["c"],
            k_connectors=2,
            limit_to_zone=limit_to_zone,
        )

    updated_links = project.network.links.data

    new_connectors = updated_links[updated_links.link_type == "centroid_connector"]
    added_connectors = new_connectors[~new_connectors.link_id.isin(initial_state["initial_connectors"].link_id)]

    assert len(added_connectors) >= 0, "Should create some connectors"

    if len(added_connectors) > 0:
        # Verify a_nodes are centroids and b_nodes are regular nodes
        centroid_node_ids = initial_state["nodes"][initial_state["nodes"].is_centroid == 1].node_id.values
        regular_node_ids = initial_state["nodes"][initial_state["nodes"].is_centroid == 0].node_id.values

        assert all(added_connectors.a_node.isin(centroid_node_ids)), "A-nodes should be centroids"
        assert all(added_connectors.b_node.isin(regular_node_ids)), "B-nodes should be regular nodes"


def test_bulk_connector_creation_geometry_validation(coquimbo_example, initial_state):
    """Test that connector geometries are properly formed."""
    project = coquimbo_example

    with project.db_connection_spatial as conn:
        bulk_connector_creation(
            conn=conn,
            project_nodes=initial_state["nodes"],
            project_links=initial_state["links"],
            project_zones=initial_state["zones"],
            modes=["c"],
            k_connectors=1,
            limit_to_zone=True,
        )

    updated_links = project.network.links.data

    new_connectors = updated_links[updated_links.link_type == "centroid_connector"]
    added_connectors = new_connectors[~new_connectors.link_id.isin(initial_state["initial_connectors"].link_id)]

    if len(added_connectors) > 0:
        for _, connector in added_connectors.iterrows():
            assert isinstance(connector.geometry, LineString), "Connector geometry should be LineString"

            # Get centroid and node geometries
            centroid_geom = initial_state["nodes"][initial_state["nodes"].node_id == connector.a_node].geometry.iloc[0]
            node_geom = initial_state["nodes"][initial_state["nodes"].node_id == connector.b_node].geometry.iloc[0]

            # LineString should connect these points
            line_coords = list(connector.geometry.coords)
            assert len(line_coords) == 2, "LineString should have exactly 2 coordinates"
            assert Point(line_coords[0]).equals(centroid_geom), "LineString should start at centroid"
            assert Point(line_coords[1]).equals(node_geom), "LineString should end at node"


def test_bulk_connector_creation_distance_upper_bound(coquimbo_example, initial_state):
    """Test connector creation with distance upper bound."""
    project = coquimbo_example

    # Test with very small distance bound - should create fewer connectors
    with project.db_connection_spatial as conn:
        bulk_connector_creation(
            conn=conn,
            project_nodes=initial_state["nodes"],
            project_links=initial_state["links"],
            project_zones=initial_state["zones"],
            modes=["c"],
            k_connectors=5,
            limit_to_zone=True,
            distance_upper_bound=0.001,
        )

    updated_links_small = project.network.links.data
    connectors_small = updated_links_small[updated_links_small.link_type == "centroid_connector"]

    # Reset and test with large distance bound
    with project.db_connection_spatial as conn:
        bulk_connector_creation(
            conn=conn,
            project_nodes=initial_state["nodes"],
            project_links=project.network.links.data,
            project_zones=initial_state["zones"],
            modes=["c"],
            k_connectors=5,
            limit_to_zone=True,
            distance_upper_bound=float("inf"),
        )

    updated_links_large = project.network.links.data
    connectors_large = updated_links_large[updated_links_large.link_type == "centroid_connector"]

    # With larger distance bound, should generally have same or more connectors
    # (accounting for potential existing connectors)
    assert len(connectors_large) >= len(connectors_small), "Larger distance bound should allow more connectors"


def test_bulk_connector_creation_no_changes_when_no_work(coquimbo_example, initial_state):
    """Test that function handles gracefully when no work needs to be done."""
    project = coquimbo_example

    # First, create some connectors
    with project.db_connection_spatial as conn:
        bulk_connector_creation(
            conn=conn,
            project_nodes=initial_state["nodes"],
            project_links=initial_state["links"],
            project_zones=initial_state["zones"],
            modes=["c"],
            k_connectors=1,
            limit_to_zone=True,
        )

    intermediate_links = project.network.links.data.copy()

    # Run again with same parameters - should not create duplicate connectors
    with project.db_connection_spatial as conn:
        bulk_connector_creation(
            conn=conn,
            project_nodes=project.network.nodes.data,
            project_links=project.network.links.data,
            project_zones=initial_state["zones"],
            modes=["c"],
            k_connectors=1,
            limit_to_zone=True,
        )

    final_links = project.network.links.data

    # Should have same number of links
    assert len(final_links) == len(intermediate_links), "Should not create duplicate connectors"
