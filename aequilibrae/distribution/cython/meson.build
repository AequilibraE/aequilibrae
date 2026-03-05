py.install_sources(
    'ipf_core.pyx',
    subdir: 'aequilibrae.distribution.cython'
)

py.extension_module(
    'ipf_core',
    'ipf_core.pyx',
    cpp_args: cpp_args,
    link_args: link_args,
    install: true,
    subdir: 'aequilibrae.distribution.cython'
)
