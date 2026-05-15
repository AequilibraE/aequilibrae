thisdir = 'aequilibrae/paths'

pyfiles = [
  'AoN.py',
  '__init__.py',
  'all_or_nothing.py',
  'assignment_paths.py',
  'connectivity_analysis.py',
  'graph.py',
  'graph_building.py',
  'linear_approximation.py',
  'multi_threaded_aon.py',
  'multi_threaded_paths.py',
  'network_skimming.py',
  'optimal_strategies.py',
  'public_transport.py',
  'route_choice.py',
  'sub_area.py',
  'traffic_assignment.py',
  'traffic_class.py',
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
