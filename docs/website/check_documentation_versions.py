import os
import sys
from pathlib import Path

npth = Path(__file__).parent.parent.parent
if npth not in sys.path:
    sys.path.append(npth)
    print(npth)

with open(npth / "__version__.py") as f:
    exec(f.read())

# We check if the reference to all existing versions were added by checking
# that the current version is referenced
with open(os.path.join(npth, "source/version_history.rst"), mode="r") as f:
    txt = f.read()

assert f"{release_version}" in txt
assert f"V.{release_version}" in txt
