import os
import sys
from importlib.metadata import version
from pathlib import Path

npth = Path(__file__).parent.parent.parent
if npth not in sys.path:
    sys.path.append(npth)
    print(npth)

release_version = version("aequilibrae")

with open(os.path.join(npth, "docs/source/useful_links/version_history.rst"), mode="r") as f:
    txt = f.read()

print(f"python/v{release_version}")
assert f"python/v{release_version}" in txt
