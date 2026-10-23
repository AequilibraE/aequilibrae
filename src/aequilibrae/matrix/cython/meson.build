thisdir = 'aequilibrae/matrix/cython'

pyfiles = [
    '__init__.py',
]

exts = [
    'dense_matrix',
    'matrix_base',
    'sparse_matrix',
    'sparse_matrix_builder',
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
