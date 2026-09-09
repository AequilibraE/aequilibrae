thisdir = 'aequilibrae/paths/cython'

pyfiles = [
    '__init__.py',
]

exts = [
    'AoN',
    'basic_path_finding',
    'graph_building',
    'parallel_numpy',
    'path_file_saving',
    'public_transport',
    'route_choice_link_loading_results',
    'route_choice_set',
    'route_choice_set_results',
    'skimming_core',
    'vdf_core',
    'new_path_finding',
    'graph_context',
    'search_results',
    'dijkstra',
]

py.install_sources(
    pyfiles,
    pure: false,
    subdir: thisdir,
)

foreach ext : exts
    sources = [ext + '.pyx']
    py.extension_module(
        ext,
        sources,
        dependencies: deps,
        install: true,
        subdir: thisdir,
        include_directories: incdir,
    )
endforeach
