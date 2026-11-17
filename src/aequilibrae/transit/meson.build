thisdir = 'aequilibrae/transit'

pyfiles = [
  '__init__.py',
  'column_order.py',
  'constants.py',
  'date_tools.py',
  'gtfs_exporter.py',
  'gtfs_loader.py',
  'lib_gtfs.py',
  'parse_csv.py',
  'optimal_strategies.py',
  'route_map_matcher.py',
  'route_types.py',
  'transit.py',
  'transit_assignment.py',
  'transit_assignment_results.py',
  'transit_class.py',
  'transit_graph_builder.py',
]

pkgs = [
  'functions',
  'transit_elements',
]

py.install_sources(
  pyfiles,
  pure: false,
  subdir: thisdir,
)

foreach pkg : pkgs
  subdir(pkg)
endforeach
