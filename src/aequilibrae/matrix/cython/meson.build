thisdir = 'aequilibrae/matrix/cython'

py.install_sources('__init__.py', pure: false, subdir: thisdir)

py.extension_module(
  'dense_matrix',
  'dense_matrix.pyx',
  dependencies: deps,
  install: true,
  subdir: thisdir,
  include_directories: include_directories('../cpp'),
)
