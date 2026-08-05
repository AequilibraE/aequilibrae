import pathlib

path = pathlib.Path(__file__).parent

migrations = [path / "000_initial_migration.py", path / "-001_negative_id.py"]  # Negative ID
