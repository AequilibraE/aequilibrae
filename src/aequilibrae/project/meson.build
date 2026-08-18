thisdir = 'aequilibrae/project'

pyfiles = [
  'project_table.py',
  '__init__.py',
  'about.py',
  'field_editor.py',
  'path_lock.py',
  'project.py',
  'project_cleaning.py',
  'project_creation.py',
  'scenario.py',
  'zoning.py',
]

pkgs = [
  'data',
  'database_specification',
  'network',
  'tools',
]

py.install_sources(
  pyfiles,
  pure: false,
  subdir: thisdir,
)

foreach pkg : pkgs
  subdir(pkg)
endforeach
