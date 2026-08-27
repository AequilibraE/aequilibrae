# AequilibraE transport-modeling notebooks

A complete, hands-on transport modeling course built on AequilibraE's bundled example
models, with interactive [JupyterGIS](https://jupytergis.readthedocs.io) maps.

| # | Notebook | What you learn |
|---|----------|----------------|
| 01 | [Project & network](01_project_and_network.ipynb) | Project structure, links/nodes/zones as GeoDataFrames, first map |
| 02 | [Zones & connectors](02_zones_and_connectors.ipynb) | Hexagonal zoning, centroids, centroid connectors |
| 03 | [Paths & skimming](03_paths_and_skimming.ipynb) | Graphs, shortest paths, zone-to-zone skim matrices |
| 04 | [Trip distribution](04_trip_distribution.ipynb) | Gravity model calibration, deterrence functions, IPF |
| 05 | [Traffic assignment](05_traffic_assignment.ipynb) | BPR volume-delay, BFW equilibrium, congestion mapping |
| 06 | [Route choice](06_route_choice.ipynb) | BFSLE choice sets, path-size logit assignment |
| 07 | [Public transport](07_public_transport.ipynb) | GTFS import, transit database, route/stop mapping |
| 08 | [Full model workflow](08_full_model_workflow.ipynb) | Base/future years, select-link analysis, scenario comparison |

## Setup

```bash
pip install aequilibrae jupytergis jupyterlab matplotlib
jupyter lab
```

That is the entire setup: AequilibraE's spatial database engine is pure Python
(shapely + pyproj + SQLite's built-in R*Tree), so no native SpatiaLite package or
download is required on any platform.

The interactive maps render inside **JupyterLab** (the JupyterGIS extension installs
with the `jupytergis` wheel). Each map is a live document: use the layer tree to
toggle layers, edit symbology, or export to QGIS with `doc.export_to_qgis(...)`.

All notebooks are self-contained, run top-to-bottom in a few minutes each, and write
only to a throw-away temporary folder.

The notebooks mirror the worked examples in the
[AequilibraE documentation](https://www.aequilibrae.com) — refer there for deeper
background on each modeling stage.
