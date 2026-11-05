import re
import subprocess
import sys
from pathlib import Path

import aequilibrae


def test_version_matches_meson_project():
    meson_file = Path(__file__).resolve().parents[1] / "meson.build"
    match = re.search(r"^\s*version:\s*'([^']+)'", meson_file.read_text(), re.MULTILINE)
    assert match is not None
    assert aequilibrae.version == match.group(1)


def test_import_does_not_require_distribution_metadata():
    script = """
import importlib.metadata

original_version = importlib.metadata.version

def version(name):
    if name == "aequilibrae":
        raise importlib.metadata.PackageNotFoundError(name)
    return original_version(name)

importlib.metadata.version = version

import aequilibrae
from aequilibrae._version import version as generated_version

assert aequilibrae.version == generated_version
print(aequilibrae.version)
"""
    result = subprocess.run([sys.executable, "-P", "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == aequilibrae.version
