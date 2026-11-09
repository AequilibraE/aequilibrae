import numpy as np
import pytest
from aequilibrae.paths.cython.route_choice_set import RouteChoiceSet
from aequilibrae.utils.cython.bridge import Bridge

from .test_route_choice_a_star import compressed_graph


@pytest.mark.parametrize("path_size_logit", [False, True])
@pytest.mark.parametrize("bfsle", [False, True])
@pytest.mark.parametrize("turn", [False, True])
def test_rebuild_results_table(turn, bfsle, path_size_logit):
    choice = RouteChoiceSet(compressed_graph(turn))
    with Bridge() as bridge:
        choice.batched(
            nodes=[(10, 60), (60, 10)],
            max_routes=2,
            max_depth=6,
            penalty=4.0,
            bfsle=bfsle,
            path_size_logit=path_size_logit,
            cores=1,
            bridge=bridge,
        )
    results = choice.get_results(rebuild=True)
    assert results["origin id"].dtype == np.uint32
    assert results["destination id"].dtype == np.uint32
    assert {tuple(route) for route in results["route set"]} == {(71, 72, 12), (55, 56, 24)}
    assert all(route.dtype == np.int64 for route in results["route set"])

    if path_size_logit:
        assert results["mask"].dtype == bool
        assert (results[["cost", "path overlap", "probability"]].dtypes == np.float64).all()
    else:
        assert list(results.columns) == ["origin id", "destination id", "route set"]


def test_rebuild_requires_results():
    choice = RouteChoiceSet(compressed_graph(False))
    with pytest.raises(RuntimeError, match="results not computed yet"):
        choice.get_results(rebuild=True)
