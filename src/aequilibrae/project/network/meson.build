thisdir = 'aequilibrae/project/network'

pyfiles = [
  '__init__.py',
  'connector_creation.py',
  'gmns_builder.py',
  'gmns_exporter.py',
  'haversine.py',
  'link_types.py',
  'links.py',
  'modes.py',
  'network.py',
  'network.yml',
  'nodes.py',
  'periods.py',
]

pkgs = [
  'osm',
]

py.install_sources(
  pyfiles,
  pure: false,
  subdir: thisdir,
)

foreach pkg : pkgs
  subdir(pkg)
endforeach
