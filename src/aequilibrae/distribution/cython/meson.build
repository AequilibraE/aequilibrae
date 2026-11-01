thisdir = 'aequilibrae/distribution/cython'

exts = [
  'gravity_core',
  'ipf_core',
]

foreach ext : exts
  py.extension_module(
    ext,
    ext + '.pyx',
    dependencies: deps,
    install: true,
    subdir: thisdir,
  )
endforeach
