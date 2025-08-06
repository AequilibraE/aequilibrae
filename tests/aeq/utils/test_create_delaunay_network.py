import pytest

from aequilibrae.utils.create_delaunay_network import DelaunayAnalysis


def test_create_delaunay_network(sioux_falls_example):
    da = DelaunayAnalysis(sioux_falls_example)

    # Test invalid source
    with pytest.raises(ValueError, match="Source must be 'zones' or 'network'"):
        da.create_network("nodes")

    with sioux_falls_example.db_connection as conn:
        # Test creating network with default source
        da.create_network()
        assert conn.execute("select count(*) from delaunay_network").fetchone()[0] == 59

        # Test overwriting network with "network" source
        da.create_network("network", True)
        assert conn.execute("select count(*) from delaunay_network").fetchone()[0] == 62

    # Test attempting to create network without overwrite flag
    with pytest.raises(ValueError, match="Delaunay network already exist. Use the overwrite flag to re-run it"):
        da.create_network()


def test_assign_matrix(sioux_falls_example):
    demand = sioux_falls_example.matrices.get_matrix("demand_omx")
    demand.computational_view(["matrix"])
    da = DelaunayAnalysis(sioux_falls_example)

    # Create network and assign matrix
    da.create_network()
    da.assign_matrix(demand, "delaunay_test")


def test_create_network_with_invalid_overwrite_flag(sioux_falls_example):
    da = DelaunayAnalysis(sioux_falls_example)

    # Create network initially
    da.create_network()

    # Test creating network without overwrite flag
    with pytest.raises(ValueError, match="Delaunay network already exist. Use the overwrite flag to re-run it"):
        da.create_network()

    # Test creating network with overwrite flag
    da.create_network(overwrite=True)
    with sioux_falls_example.db_connection as conn:
        assert conn.execute("select count(*) from delaunay_network").fetchone()[0] > 0


def test_assign_matrix_with_invalid_matrix(sioux_falls_example):
    da = DelaunayAnalysis(sioux_falls_example)
    da.create_network()

    # Test assigning matrix with invalid input
    with pytest.raises(AttributeError):
        da.assign_matrix(None, "invalid_test")


def test_create_network_with_zones_source(sioux_falls_example):
    da = DelaunayAnalysis(sioux_falls_example)

    # Test creating network with "zones" source
    da.create_network(source="zones")
    with sioux_falls_example.db_connection as conn:
        assert conn.execute("select count(*) from delaunay_network").fetchone()[0] > 0


def test_create_network_with_empty_project(empty_project):
    da = DelaunayAnalysis(empty_project)

    # Test creating network when project has no zones or nodes
    with empty_project.db_connection as conn:
        conn.execute("DELETE FROM zones")
        conn.execute("DELETE FROM nodes")
        conn.commit()

    with pytest.raises(Exception):
        da.create_network()
