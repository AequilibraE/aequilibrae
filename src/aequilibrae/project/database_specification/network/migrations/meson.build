thisdir = 'aequilibrae/project/database_specification/network/migrations'

pyfiles = [
    '000_initial_migration.sql',
    '001_add_cols_to_results.sql',
    '002_add_scenario_table.py',
    '003_protect_link_end_nodes.py',
    '004_fix_default_period_delete_trigger.sql',
    'migrations.py',
]

py.install_sources(
    pyfiles,
    pure: false,
    subdir: thisdir,
)
