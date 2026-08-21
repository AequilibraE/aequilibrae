import pathlib

path = pathlib.Path(__file__).parent

migrations: tuple[int, pathlib.Path] = [
    path / "000_initial_migration.py",
    path / "001_allow_duplicate_nodes.py",
    path / "002_support_saving_multiple_transit_periods.py",
    path / "003_align_taz_id_and_node_id.py",
    path / "004_move_results_to_project.py",
    path / "005_repair_foreign_keys.py",
]
