thisdir = 'aequilibrae/routing'

pyfiles = [
    '__init__.py',
    'connectivity_analysis.py',
    'graph.py',
    'graph_building.py',
    'mapping.py',
    'network.py',
    'network_skimming.py',
    'path_heuristics.py',
    'routing_context.py',
    'transit_graph.py',
]

pkgs = ['cython', 'results']

py.install_sources(pyfiles, pure: false, subdir: thisdir)

foreach pkg : pkgs
    subdir(pkg)
endforeach
