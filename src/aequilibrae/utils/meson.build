thisdir = 'aequilibrae/utils'

pyfiles = [
    '__init__.py',
    'aeq_signal.py',
    'core_setter.py',
    'create_delaunay_network.py',
    'create_example.py',
    'db_utils.py',
    'geo_index.py',
    'geo_utils.py',
    'get_table.py',
    'list_tables_in_db.py',
    'logging_utils.py',
    'model_run_utils.py',
    'python_signal.py',
    'qgis_utils.py',
    'resolve_cysignals.py',
    'spatialite_utils.py',
]

pkgs = [
    'cython',
    'interface',
    'simwrapper',
]

py.install_sources(
    pyfiles,
    pure: false,
    subdir: thisdir,
)

foreach pkg : pkgs
    subdir(pkg)
endforeach
