import os
import subprocess
import sys

import pytest

CYCLES = 3000

# Without the DLL pin in spatialite_utils, SQLite LoadLibrary/FreeLibrary's mod_spatialite on
# every connection open/close, and each cycle leaks a Windows TLS index. The process aborts once
# the ~1088-slot limit is reached, at roughly 1000 cycles.
SCRIPT = f"""
import sqlite3
from aequilibrae.utils import spatialite_utils

spatialite_utils.ensure_spatialite_binaries()
for _ in range({CYCLES}):
    conn = sqlite3.connect(":memory:")
    spatialite_utils.load_spatialite_extension(conn)
    conn.close()
print("SURVIVED")
"""


@pytest.mark.skipif(os.name != "nt", reason="TLS-index exhaustion on extension load/unload is Windows-specific")
def test_spatialite_survives_repeated_connection_cycles():
    # A regression aborts the interpreter (exit code 3), so the loop runs in a subprocess
    # to fail this test rather than kill the pytest process.
    result = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"Process died after repeated spatialite loads: {result.stderr}"
    assert "SURVIVED" in result.stdout
