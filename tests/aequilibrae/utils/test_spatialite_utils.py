import os
import subprocess
import sys

import pytest

CYCLES = 3000

# Without the pin in spatialite_utils, SQLite unloads mod_spatialite after every connection closes.
# Repeated cycles exhaust Windows TLS indexes or macOS pthread_atfork slots. macOS can continue
# loading Spatialite after its at-fork table is full, so the script probes pthread_atfork explicitly.
SCRIPT = f"""
import ctypes
import os
import sqlite3
import sys
from aequilibrae.utils import spatialite_utils

spatialite_utils.ensure_spatialite_binaries()
for _ in range({CYCLES}):
    conn = sqlite3.connect(":memory:")
    spatialite_utils.load_spatialite_extension(conn)
    conn.close()

if sys.platform == "darwin":
    pthread_atfork = ctypes.CDLL(None).pthread_atfork
    pthread_atfork.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    pthread_atfork.restype = ctypes.c_int
    status = pthread_atfork(None, None, None)
    if status != 0:
        raise RuntimeError(f"pthread_atfork failed with {{status}}: {{os.strerror(status)}}")

print("SURVIVED")
"""


@pytest.mark.skipif(
    os.name != "nt" and sys.platform != "darwin",
    reason="Extension load/unload exhaustion is specific to Windows and macOS",
)
def test_spatialite_survives_repeated_connection_cycles():
    # The Windows regression aborts the interpreter, so isolate both platform checks in a subprocess.
    result = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"Process died after repeated spatialite loads: {result.stderr}"
    assert "SURVIVED" in result.stdout
