thisdir = 'aequilibrae'

pyfiles = [
  '__init__.py',
  '_assignment.py',
  'context.py',
  'log.py',
  'parameters.py',
  'parameters.yml',
]

pkgs = [
  'builder',
  'distribution',
  'examples',
  'matrix',
  'routing',
  'route_choice',
  'traffic',
  'project',
  'reference_files',
  'transit',
  'utils',
]

version_file = configure_file(
  input: '_version.py.in',
  output: '_version.py',
  configuration: {'PROJECT_VERSION': meson.project_version()},
)

py.install_sources(
  pyfiles + [version_file],
  pure: false,
  subdir: thisdir,
)

foreach pkg : pkgs
  subdir(pkg)
endforeach
