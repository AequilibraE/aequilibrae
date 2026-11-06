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
  'route_map_matcher.py',
  'route_types.py',
  'transit.py',
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
