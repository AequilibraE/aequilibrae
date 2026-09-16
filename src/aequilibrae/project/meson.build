thisdir = 'aequilibrae/project'

pyfiles = [
  'project_table.py',
  '__init__.py',
  '__main__.py',
  'about.py',
  'database_connection.py',
  'field_editor.py',
  'project.py',
  'project_cleaning.py',
  'project_creation.py',
  'scenario.py',
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
