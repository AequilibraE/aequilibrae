thisdir = 'aequilibrae/paths'

pyfiles = [
    '__init__.py',
    'assignment_context.py',
    'assignment_paths.py',
    'connectivity_analysis.py',
    'graph.py',
    'graph_building.py',
    'linear_approximation.py',
    'network.py',
    'network_skimming.py',
    'optimal_strategies.py',
    'path_heuristics.py',
    'route_choice.py',
    'routing_context.py',
    'sub_area.py',
    'traffic_assignment.py',
    'traffic_class.py',
    'transit_costing.py',
    'vdf.py',
]

pkgs = [
    'cython',
    'results',
]

py.install_sources(
    pyfiles,
    pure: false,
    subdir: thisdir,
)

foreach pkg : pkgs
    subdir(pkg)
endforeach
