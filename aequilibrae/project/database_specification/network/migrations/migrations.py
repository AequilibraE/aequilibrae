import pathlib

path = pathlib.Path(__file__).parent

migrations: tuple[int, pathlib.Path] = [path / "000_initial_migration.sql", path / "001_add_cols_to_results.sql"]
