thisdir = 'aequilibrae/project'

pyfiles = [
  '__init__.py',
  'about.py',
  'basic_table.py',
  'data_loader.py',
  'database_connection.py',
  'field_editor.py',
  'project.py',
  'project_cleaning.py',
  'project_creation.py',
  'scenario.py',
  'table_loader.py',
  'zone.py',
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
