thisdir = 'aequilibrae/project/network/importer'

pyfiles = [
  '__init__.py',
  'about_writer.py',
  'db_writer.py',
  'download_cache.py',
  'exceptions.py',
  'importer.py',
  'staged_network.py',
]

pkgs = [
  'schema',
  'simplifiers',
  'sources',
]

py.install_sources(
  pyfiles,
  pure: false,
  subdir: thisdir,
)

foreach pkg : pkgs
  subdir(pkg)
endforeach
