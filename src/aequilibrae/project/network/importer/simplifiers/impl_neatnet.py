import geopandas as gpd
import logging
import numpy as np
import shapely
from shapely.geometry import Point

from aequilibrae.project.network.importer.staged_network import StagedNetwork
from aequilibrae.utils.optional_dependency import require

logger = logging.getLogger(__name__)


def run_neatnet_simplify(net: StagedNetwork, **_) -> StagedNetwork:
    """Run ``neatnet.simplify_network`` on the staged network's edges and reproject back.

    The neatnet result loses per-edge provenance — use ``simplify="osmnx"`` if
    full provenance matters.
    """
    require("neatnet", feature="neatnet simplification")
    import neatnet  # type: ignore

    if len(net.links) == 0:
        return net

    edges = net.links.copy()
    utm = edges.geometry.estimate_utm_crs()
    simplified = neatnet.simplify_network(edges.to_crs(utm)).to_crs("EPSG:4326")
    return _gdf_to_staged(simplified, source_meta=net.source_meta)


def _gdf_to_staged(edges_gdf: gpd.GeoDataFrame, source_meta: dict) -> StagedNetwork:
    edges = edges_gdf.copy().reset_index(drop=True)
    if edges.crs is None:
        edges = edges.set_crs("EPSG:4326")

    # Vectorised endpoint extraction via shapely 2.x array API
    coords, indices = shapely.get_coordinates(edges.geometry, return_index=True)
    last_pos = np.searchsorted(indices, np.arange(len(edges)), side="right") - 1
    first_pos = np.searchsorted(indices, np.arange(len(edges)), side="left")
    starts = coords[first_pos]
    ends = coords[last_pos]

    endpoints = {}
    a_nodes = np.empty(len(edges), dtype=np.int64)
    b_nodes = np.empty(len(edges), dtype=np.int64)
    next_id = 10000
    for i, (s, e) in enumerate(zip(starts, ends)):
        for arr, target in ((s, a_nodes), (e, b_nodes)):
            key = (round(float(arr[0]), 7), round(float(arr[1]), 7))
            nid = endpoints.get(key)
            if nid is None:
                nid = endpoints[key] = next_id
                next_id += 1
            target[i] = nid

    edges["a_node"] = a_nodes
    edges["b_node"] = b_nodes
    edges["link_id"] = np.arange(1, len(edges) + 1, dtype=np.int64)
    for col, default in (("direction", 0), ("modes", "c"), ("link_type", "unknown")):
        if col not in edges.columns:
            edges[col] = default
    utm = edges.geometry.estimate_utm_crs()
    edges["distance"] = edges.geometry.to_crs(utm).length.astype(float)

    nodes = gpd.GeoDataFrame(
        {
            "node_id": list(endpoints.values()),
            "geometry": [Point(x, y) for x, y in endpoints],
            "modes": "c",
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    out = StagedNetwork(nodes=nodes, links=edges, source_meta=dict(source_meta))
    out.validate()
    return out
