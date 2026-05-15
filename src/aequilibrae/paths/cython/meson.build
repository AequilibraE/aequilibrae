thisdir = 'aequilibrae/paths/cython'

pyfiles = [
  '__init__.py',
]

exts = [
  'AoN',
  'graph_building',
  'public_transport',
  'route_choice_link_loading_results',
  'route_choice_set',
  'route_choice_set_results',
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
