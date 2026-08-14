thisdir = 'aequilibrae/utils/cython'

pyfiles = [
  '__init__.py',
]

exts = [
  'bar',
  'bridge',
  'openmp_helper',
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
    include_directories: incdir,
  )
endforeach
