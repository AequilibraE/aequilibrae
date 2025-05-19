import pathlib

path = pathlib.Path(__file__).parent

migrations: tuple[int, pathlib.Path] = [path / "000_initial_migration.sql", path / "001_allow_duplicate_nodes.sql"]
