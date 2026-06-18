thisdir = 'aequilibrae/project/network'

pyfiles = [
  '__init__.py',
  'connector_creation.py',
  'gmns_builder.py',
  'gmns_exporter.py',
  'haversine.py',
  'link.py',
  'link_type.py',
  'link_types.py',
  'links.py',
  'mode.py',
  'modes.py',
  'network.py',
  'network.yml',
  'node.py',
  'nodes.py',
  'period.py',
  'periods.py',
  'safe_class.py',
]

pkgs = [
  'importer',
]

py.install_sources(
  pyfiles,
  pure: false,
  subdir: thisdir,
)

foreach pkg : pkgs
  subdir(pkg)
endforeach
