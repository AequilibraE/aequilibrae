py.install_sources(
    '__init__.py',
    'aequilibrae_matrix.py',
    subdir: 'aequilibrae.matrix'
)

py.install_sources(
    'coo_demand.pxd',
    'coo_demand.pyx',
    'sparse_matrix.pxd',
    'sparse_matrix.pyx',
    subdir: 'aequilibrae.matrix'
)

py.extension_module(
    'coo_demand',
    'coo_demand.pyx',
    cpp_args: cpp_args,
    link_args: link_args,
    install: true,
    subdir: 'aequilibrae.matrix'
)

py.extension_module(
    'sparse_matrix',
    'sparse_matrix.pyx',
    cpp_args: cpp_args,
    link_args: link_args,
    install: true,
    subdir: 'aequilibrae.matrix'
)

subdir('cython')