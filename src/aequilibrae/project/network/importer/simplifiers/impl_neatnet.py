"""neatnet-based simplifier implementation (opt-in)."""

from __future__ import annotations

import logging

from aequilibrae.utils.optional_dependency import require

from ..ir import RoutableNetwork

logger = logging.getLogger(__name__)


def run_neatnet_simplify(net: RoutableNetwork, **_) -> RoutableNetwork:
    """Run ``neatnet.simplify_network`` on the IR's edges and reproject back.

    Implementation note: neatnet operates on a GeoDataFrame of LineStrings in
    a projected CRS. We project to auto-UTM, run, and reproject to EPSG:4326.

    The neatnet result loses per-edge provenance (the package does not
    propagate input attributes one-for-one). For now we drop the dict-of-dicts
    after neatnet simplification — users wanting full provenance should use
    the default ``simplify="osmnx"``.
    """
    require("neatnet", feature="neatnet simplification")
    import neatnet  # type: ignore

    if len(net.links) == 0:
        return net

    edges = net.links.copy()
    utm = str(edges.geometry.estimate_utm_crs())
    edges_proj = edges.to_crs(utm)

    try:
        simplified = neatnet.simplify_network(edges_proj)
    except AttributeError:
        # Older API
        simplified = neatnet.simplify(edges_proj)  # type: ignore[attr-defined]

    simplified_geo = simplified.to_crs("EPSG:4326")

    # neatnet returns only the simplified link geometries. We rebuild nodes
    # from endpoints and re-derive ids.
    return _gdf_to_ir(simplified_geo, source_meta=net.source_meta)


def _gdf_to_ir(edges_gdf, source_meta: dict) -> RoutableNetwork:
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import Point

    edges = edges_gdf.copy().reset_index(drop=True)
    if edges.crs is None:
        edges = edges.set_crs("EPSG:4326")

    # Build endpoint nodes
    endpoints: dict[tuple[float, float], int] = {}
    next_id = 10000

    def _node_id_for(x, y):
        nonlocal next_id
        key = (round(x, 7), round(y, 7))
        if key not in endpoints:
            endpoints[key] = next_id
            next_id += 1
        return endpoints[key]

    a_nodes = []
    b_nodes = []
    for geom in edges.geometry:
        coords = list(geom.coords)
        a_nodes.append(_node_id_for(*coords[0]))
        b_nodes.append(_node_id_for(*coords[-1]))
    edges["a_node"] = a_nodes
    edges["b_node"] = b_nodes
    edges["link_id"] = np.arange(1, len(edges) + 1, dtype=np.int64)
    if "direction" not in edges.columns:
        edges["direction"] = 0
    if "modes" not in edges.columns:
        edges["modes"] = "c"
    if "link_type" not in edges.columns:
        edges["link_type"] = "unknown"
    utm = edges.crs if str(edges.crs).upper() != "EPSG:4326" else edges.geometry.estimate_utm_crs()
    edges["distance"] = edges.geometry.to_crs(utm).length.astype(float)

    node_rows = []
    for (x, y), nid in endpoints.items():
        node_rows.append({
            "node_id": nid,
            "geometry": Point(x, y),
            "modes": "c",
        })
    nodes = gpd.GeoDataFrame(node_rows, geometry="geometry", crs="EPSG:4326")

    ir = RoutableNetwork(nodes=nodes, links=edges, source_meta=dict(source_meta))
    ir.validate()
    return ir
