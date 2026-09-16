thisdir = 'aequilibrae/project/database_specification/transit/migrations'

pyfiles = [
  '000_initial_migration.py',
  '001_allow_duplicate_nodes.py',
  '002_support_saving_multiple_transit_periods.py',
  '003_align_taz_id_and_node_id.py',
  '004_move_results_to_project.py',
  '005_repair_foreign_keys.py',
  '006_normalise_route_link_stop_ids.py',
  'migrations.py',
]

py.install_sources(
  pyfiles,
  pure: false,
  subdir: thisdir,
)
