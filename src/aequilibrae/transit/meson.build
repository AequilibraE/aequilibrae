thisdir = 'aequilibrae/transit'

pyfiles = [
  '__init__.py',
  'column_order.py',
  'constants.py',
  'date_tools.py',
  'gtfs_loader.py',
  'lib_gtfs.py',
  'route_map_matcher.py',
  'transit.py',
  'transit_graph_builder.py',
]

pkgs = [
  'functions',
  'gtfs_writer',
  'route_system_reader',
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
