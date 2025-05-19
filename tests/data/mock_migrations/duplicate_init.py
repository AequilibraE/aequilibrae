import pathlib

path = pathlib.Path(__file__).parent

migrations = [
    path / "000_initial_migration.sql",
    path / "000_initial_migration.sql",
]
