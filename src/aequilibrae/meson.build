thisdir = 'aequilibrae'

pyfiles = [
  '__init__.py',
  'log.py',
  'parameters.py',
  'parameters.yml',
]

pkgs = [
  'distribution',
  'matrix',
  'paths',
  'project',
  'reference_files',
  'transit',
  'utils',
]

py.install_sources(
  pyfiles,
  pure: false,
  subdir: thisdir,
)

foreach pkg : pkgs
  subdir(pkg)
endforeach
