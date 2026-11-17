"""Public namespace contracts and isolated import/dependency checks."""

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXPORTS = {
    "aequilibrae": ["Project", "__version__"],
    "aequilibrae.project": [
        "Tables",
        "Scenario",
        "Parameters",
        "Log",
        "Matrices",
        "Results",
        "MigrationManager",
        "NetworkSimplifier",
    ],
    "aequilibrae.builder": [
        "Importer",
        "GMNSBuilder",
        "GMNSExporter",
        "GTFSRouteSystemBuilder",
        "TransitGraphBuilder",
        "export_gtfs",
    ],
    "aequilibrae.routing": [
        "Network",
        "Graph",
        "PathResults",
        "SkimResults",
        "NetworkSkimming",
        "available_heaps",
        "estimate_heuristic_scale",
    ],
    "aequilibrae.traffic": ["TrafficClass", "TrafficAssignment", "AssignmentResults", "AssignmentPaths", "VDF"],
    "aequilibrae.traffic.vdf": ["bpr", "bpr2", "conical", "inrets", "akcelik", "builtin_vdfs"],
    "aequilibrae.transit": ["Transit", "TransitClass", "TransitAssignment", "TransitGraph", "TransitAssignmentResults"],
    "aequilibrae.route_choice": ["RouteChoice", "SubAreaAnalysis"],
    "aequilibrae.distribution": ["Ipf", "GravityCalibration", "GravityApplication", "SyntheticGravityModel"],
    "aequilibrae.matrix": [
        "AequilibraeMatrix",
        "AequilibraeSparseMatrix",
        "MatrixStore",
        "from_df",
        "from_dict",
        "from_file",
    ],
    "aequilibrae.examples": ["create_example", "list_examples"],
    "aequilibrae.utils": ["basic_config", "ensure_spatialite_binaries", "set_known_spatialite_folder"],
}
SOURCE = Path(__file__).resolve().parents[1] / "src" / "aequilibrae"


@pytest.mark.parametrize("namespace, names", EXPORTS.items())
def test_public_exports(namespace, names):
    module = importlib.import_module(namespace)
    assert set(module.__all__) == set(names)
    assert len(module.__all__) == len(names)
    for name in names:
        assert getattr(module, name) is not None


@pytest.mark.parametrize("order", [list(EXPORTS), list(reversed(EXPORTS)), sorted(EXPORTS)])
def test_import_orders_without_creation_dependencies(order):
    script = """
import importlib
import importlib.abc
import sys

class WithoutCreationDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'osmnx', 'pyrosm', 'neatnet', 'overturemaps'}:
            raise ModuleNotFoundError(f'creation dependency unavailable: {fullname}', name=fullname)

sys.meta_path.insert(0, WithoutCreationDependencies())
"""
    script += f"\nexports = {EXPORTS!r}\norder = {order!r}\n"
    script += """
for namespace in order:
    module = importlib.import_module(namespace)
    for name in exports[namespace]:
        getattr(module, name)

from aequilibrae.matrix import from_dict
from aequilibrae.routing import Network
from aequilibrae.traffic import TrafficAssignment, TrafficClass
from aequilibrae.traffic.vdf import bpr
import numpy as np
import pandas as pd

network = Network(pd.DataFrame({
    'link_id': [1], 'a_node': [1], 'b_node': [2], 'direction': [0], 'modes': ['c'],
    'time': [1.0], 'capacity': [100.0],
}), centroids=[1, 2])
demand = from_dict({'flow': np.array([[0., 3.], [2., 0.]])}, [1, 2])
configuration = TrafficClass(mode='c', cost='time', demand=demand)
assignment = TrafficAssignment(network=network, classes=[configuration], time_field='time',
    capacity_field='capacity', vdf=bpr, algorithm='all-or-nothing')
assignment.execute(cores=1, elementwise_cores=1)
assert assignment.results()['PCE_tot'].iloc[0] == 5.0
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env={**os.environ, "AEQ_SHOW_PROGRESS": "FALSE"}
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("package", ["routing", "matrix"])
def test_low_level_dependencies(package):
    higher_levels = (
        "aequilibrae.project",
        "aequilibrae.traffic",
        "aequilibrae.transit",
        "aequilibrae.route_choice",
        "aequilibrae.builder",
    )
    for path in (SOURCE / package).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            for module in modules:
                assert not module.startswith(higher_levels), (path, module)


def test_internal_imports_use_implementation_modules():
    # __init__ files define the public facades; consumers import concrete homes instead.
    facades = {name for name in EXPORTS if (SOURCE.parent / name.replace(".", "/") / "__init__.py").is_file()}
    for path in SOURCE.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            assert node.level == 0, path
            if node.module in facades:
                for name in node.names:
                    assert name.name not in EXPORTS[node.module], (path, node.module, name.name)
