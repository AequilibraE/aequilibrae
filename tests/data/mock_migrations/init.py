import pathlib

path = pathlib.Path(__file__).parent

migrations: tuple[int, pathlib.Path] = [
    path / "000_initial_migration.sql",
    path / "001_add_users.sql",
    path / "002_add_posts.sql",
    path / "003_add_comments.py",
    path / "004_invalid_migration.py",
    path / "005_non_callable_migrate.py",
]
