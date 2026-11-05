"""Backup coverage, including SQLite-only sources that cannot be opened as Projects."""

import importlib
import sqlite3
from contextlib import closing

import pytest

from aequilibrae.project.tools import backup
from aequilibrae.project.tools.cli import cli

backup_module = importlib.import_module("aequilibrae.project.tools.backup")
DATABASES = ("project_database.sqlite", "public_transport.sqlite", "results_database.sqlite")


def make_database(path, value=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE payload (id INTEGER PRIMARY KEY, value BLOB)")
        conn.execute("INSERT INTO payload VALUES (?, ?)", (value, b"\x00\xff"))
        conn.execute("CREATE INDEX payload_value ON payload(value)")
        conn.execute("CREATE VIEW payload_view AS SELECT * FROM payload")


def query(path, sql):
    with closing(sqlite3.connect(path)) as conn:
        return conn.execute(sql).fetchall()


@pytest.fixture
def plain_project(tmp_path):
    source = tmp_path / "source"
    for folder in (source, source / "scenarios" / "future"):
        for name in DATABASES:
            make_database(folder / name)
        (folder / "parameters.yml").write_text("run: {}\n")
        (folder / "aequilibrae.log").write_text("A project log\n")
        (folder / "matrices").mkdir()
        (folder / "matrices" / "demand.omx").write_bytes(b"matrix contents")
        (folder / "run" / "nested").mkdir(parents=True)
        (folder / "run" / "nested" / "helper.py").write_text("VALUE = 1\n")
        (folder / "run" / "__pycache__").mkdir()
        (folder / "run" / "__pycache__" / "helper.pyc").write_bytes(b"cache")
        (folder / "unrelated.txt").write_text("Not an AequilibraE file")
        make_database(folder / "unrelated.sqlite")
    return source


def test_backup_preserves_databases_and_only_project_files(plain_project, tmp_path):
    target = backup(plain_project, tmp_path / "backup")
    for relative in (".", "scenarios/future"):
        source = plain_project / relative
        copied = target / relative
        for name in DATABASES:
            assert query(source / name, "SELECT * FROM sqlite_master") == query(
                copied / name, "SELECT * FROM sqlite_master"
            )
            assert query(copied / name, "SELECT * FROM payload") == [(1, b"\x00\xff")]
        assert (copied / "parameters.yml").read_bytes() == (source / "parameters.yml").read_bytes()
        assert (copied / "aequilibrae.log").read_bytes() == (source / "aequilibrae.log").read_bytes()
        assert (copied / "matrices" / "demand.omx").read_bytes() == b"matrix contents"
        assert (copied / "run" / "nested" / "helper.py").is_file()
        assert not (copied / "run" / "__pycache__").exists()
        assert not (copied / "unrelated.txt").exists()
        assert not (copied / "unrelated.sqlite").exists()


def test_cli_backup(plain_project, tmp_path, capsys):
    target = tmp_path / "cli-backup"
    cli(["-p", str(plain_project), "backup", str(target)])
    assert query(target / "project_database.sqlite", "SELECT id FROM payload") == [(1,)]
    assert str(target) in capsys.readouterr().out


def test_backup_includes_committed_wal(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    database = source / "project_database.sqlite"
    with closing(sqlite3.connect(database)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("CREATE TABLE data (value INTEGER)")
        conn.execute("INSERT INTO data VALUES (42)")
        conn.commit()
        conn.execute("INSERT INTO data VALUES (99)")
        target = backup(source, tmp_path / "backup")
        assert query(target / database.name, "SELECT * FROM data") == [(42,)]
        assert not (target / "project_database.sqlite-wal").exists()


def test_path_safety(plain_project, tmp_path):
    with pytest.raises(FileExistsError):
        backup(plain_project, plain_project)
    with pytest.raises(ValueError, match="outside"):
        backup(plain_project, plain_project / "nested")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        backup(plain_project, existing)
    with pytest.raises(FileNotFoundError):
        backup(tmp_path / "missing", tmp_path / "backup")
    assert not (tmp_path / "backup").exists()


def test_failure_leaves_output_and_does_not_copy_files(plain_project, tmp_path, monkeypatch):
    original = backup_module._backup_database

    def fail_transit(source, destination):
        if source.name == "public_transport.sqlite":
            raise sqlite3.DatabaseError("deliberate failure")
        original(source, destination)

    monkeypatch.setattr(backup_module, "_backup_database", fail_transit)
    target = tmp_path / "partial"
    with pytest.raises(sqlite3.DatabaseError, match="deliberate failure"):
        backup(plain_project, target)
    assert (target / "project_database.sqlite").is_file()
    assert not (target / "parameters.yml").exists()
    assert not (target / "matrices").exists()


def test_cli_dispatch_without_opening_project(tmp_path, monkeypatch, capsys):
    calls = []

    def operation(source, destination):
        calls.append((source, destination))
        return destination

    monkeypatch.setattr(backup_module, "backup", operation)
    source, target = str(tmp_path / "source"), str(tmp_path / "target")
    cli(["-p", source, "backup", target])
    assert calls == [(source, target)]
    assert target in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli(["-p", source, "backup", target, "--unknown"])
    assert len(calls) == 1
