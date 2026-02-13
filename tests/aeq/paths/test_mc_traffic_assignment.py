import pytest

from aequilibrae import TrafficAssignment, TrafficClass


@pytest.fixture(scope="function")
def assignment_setup(sioux_falls_example):
    sioux_falls_example.network.build_graphs()
    car_graph = sioux_falls_example.network.graphs["c"]  # type: Graph
    truck_graph = sioux_falls_example.network.graphs["T"]  # type: Graph
    moto_graph = sioux_falls_example.network.graphs["M"]  # type: Graph

    for graph in [car_graph, truck_graph, moto_graph]:
        graph.set_skimming(["free_flow_time"])
        graph.set_graph("free_flow_time")
        graph.set_blocked_centroid_flows(False)

    car_matrix = sioux_falls_example.matrices.get_matrix("demand_mc")
    car_matrix.computational_view(["car"])

    truck_matrix = sioux_falls_example.matrices.get_matrix("demand_mc")
    truck_matrix.computational_view(["trucks"])

    moto_matrix = sioux_falls_example.matrices.get_matrix("demand_mc")
    moto_matrix.computational_view(["motorcycle"])

    assignment = TrafficAssignment()
    carclass = TrafficClass("car", car_graph, car_matrix)
    carclass.set_pce(1.0)
    motoclass = TrafficClass("motorcycle", moto_graph, moto_matrix)
    motoclass.set_pce(0.2)
    truckclass = TrafficClass("truck", truck_graph, truck_matrix)
    truckclass.set_pce(2.5)

    algorithms = ["msa", "cfw", "bfw", "frank-wolfe"]

    yield {
        "project": sioux_falls_example,
        "car_matrix": car_matrix,
        "truck_matrix": truck_matrix,
        "moto_matrix": moto_matrix,
        "assignment": assignment,
        "carclass": carclass,
        "motoclass": motoclass,
        "truckclass": truckclass,
        "algorithms": algorithms,
    }

    # Teardown
    for mat in [car_matrix, truck_matrix, moto_matrix]:
        mat.close()
    sioux_falls_example.close()


def test_set_classes(assignment_setup):
    assignment = assignment_setup["assignment"]
    carclass = assignment_setup["carclass"]
    truckclass = assignment_setup["truckclass"]
    motoclass = assignment_setup["motoclass"]

    with pytest.raises(AttributeError):
        assignment.set_classes([1, 2])

    with pytest.raises(Exception):
        assignment.set_classes(carclass)

    assignment.set_classes([carclass, truckclass, motoclass])


def test_execute_and_save_results(assignment_setup):
    assignment = assignment_setup["assignment"]
    carclass = assignment_setup["carclass"]
    truckclass = assignment_setup["truckclass"]
    motoclass = assignment_setup["motoclass"]

    assignment.set_classes([carclass, truckclass, motoclass])

    for cls in assignment.classes:
        cls.graph.set_skimming(["free_flow_time", "distance"])
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})

    assignment.set_capacity_field("capacity")
    assignment.set_time_field("free_flow_time")

    assignment.max_iter = 20
    assignment.set_algorithm("bfw")
    assignment.execute()

    assignment.save_results("save_to_database")
    assignment.save_skims(matrix_name="my_skims", which_ones="all")

    with pytest.raises(ValueError):
        assignment.save_results("save_to_database")

    with pytest.raises(FileExistsError):
        assignment.save_skims(matrix_name="my_skims", which_ones="all")
