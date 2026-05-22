# AequilibraE

[![Downloads](https://img.shields.io/pypi/dm/aequilibrae.svg?maxAge=2592000)](https://pypi.python.org/pypi/aequilibrae)
[![Documentation](https://github.com/AequilibraE/aequilibrae/actions/workflows/documentation.yml/badge.svg)](https://github.com/AequilibraE/aequilibrae/actions/workflows/documentation.yml)
[![unit tests](https://github.com/AequilibraE/aequilibrae/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/AequilibraE/aequilibrae/actions/workflows/unit_tests.yml)
[![Code coverage](https://github.com/AequilibraE/aequilibrae/actions/workflows/test_linux_with_coverage.yml/badge.svg)](https://github.com/AequilibraE/aequilibrae/actions/workflows/test_linux_with_coverage.yml)
[![Packaging](https://github.com/AequilibraE/aequilibrae/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/AequilibraE/aequilibrae/actions/workflows/build_wheels.yml)

AequilibraE is an open-source transportation modeling package for Python 3.10+,
released under a permissive, business-friendly license.

It is designed as general-purpose modeling software and imposes very little
structure on models built with it. Many core algorithms can also be used without
an AequilibraE project by working directly with pandas DataFrames and NumPy
arrays, which makes the package useful when transportation modeling is one
component of a larger analytical pipeline.

## What it provides

- Project-based network modeling backed by SQLite/SpatiaLite databases.
- Network editing through the Python API, SQL, or GIS tools that support SpatiaLite.
- Data integrity support through spatial database triggers.
- Native AequilibraE matrices, OMX matrix interoperability, sparse matrices, and skim outputs.
- Multi-class traffic assignment with class-specific networks, value of time, generalized costs, MSA, Frank-Wolfe, conjugate Frank-Wolfe, and biconjugate Frank-Wolfe workflows.
- Public transport support including GTFS import, route map matching, transit graph construction, preloading, and Optimal Strategies transit assignment.
- Trip distribution models, including gravity models and cache-optimized IPF.
- OSM and GMNS network import/export support.
- Performance-critical Cython kernels built with C++17 and OpenMP.

AequilibraE project data is primarily stored in SQLite/SpatiaLite databases,
with matrix files stored alongside the project. This keeps project data in
widely supported formats while allowing AequilibraE to maintain consistency
between links, nodes, zones, modes, matrices, results, scenarios, and transit
data.

## Development status

AequilibraE is developed in the open and uses GitHub Actions for linting,
testing, coverage, documentation, source distributions, and wheels. The test
workflow covers Windows, Ubuntu Linux, and macOS across Python 3.10 through
3.14. The current wheel workflow builds on Ubuntu, Windows, and Ubuntu ARM.

For local development:

```bash
pip install -e ".[dev]"
pytest tests/ --durations=50 --dist=loadscope -n 4 --random-order --verbose
ruff check aequilibrae/
```

The Sphinx documentation lives under `docs/`, and doctests are part of the
documentation workflow.

This repository also contains OpenSpec/Codex/Copilot context for
spec-driven brownfield development. See `AGENTS.md` for agent operating rules
and `openspec/project.md` for the detailed project architecture snapshot.

## Comprehensive documentation

[AequilibraE documentation built with Sphinx ](http://www.aequilibrae.com)


### What is available only in QGIS

Some common resources for transportation modeling are inherently visual, and therefore they make more sense if
available within a GIS platform. For that reason, many resources are available only from AequilibraE's 
[QGIS plugin](http://plugins.qgis.org/plugins/qaequilibrae/),
which uses AequilibraE as its computational workhorse and also provides GUIs for most of AequilibraE's tools. Said tool
is developed independently and may lag behind the Python package. More details can be found in its
[GitHub repository](https://github.com/AequilibraE/qaequilibrae).
