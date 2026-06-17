thisdir = 'aequilibrae/project/network/importer/sources'

pyfiles = [
  '__init__.py',
  'base.py',
]

pkgs = [
  'generic',
  'osm',
  'overture',
]

py.install_sources(
  pyfiles,
  pure: false,
  subdir: thisdir,
)

foreach pkg : pkgs
  subdir(pkg)
endforeach
