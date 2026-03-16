thisdir = 'aequilibrae/matrix'

pyfiles = [
  '__init__.py',
  'aequilibrae_matrix.py',
]

exts = [
  'coo_demand',
  'sparse_matrix',
]

py.install_sources(
  pyfiles,
  pure: false,
  subdir: thisdir,
)

foreach ext : exts
  py.extension_module(
    ext,
    ext + '.pyx',
    dependencies: deps,
    install: true,
    subdir: thisdir,
  )
endforeach
