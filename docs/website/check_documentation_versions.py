import os
import sys
from pathlib import Path
import pkg_resources

npth = Path(__file__).parent.parent.parent
if npth not in sys.path:
    sys.path.append(npth)
    print(npth)

release_version = pkg_resources.get_distribution("aequilibrae").version

# We check if the reference to all existing versions were added by checking
# that the current version is referenced
with open(os.path.join(npth, "docs/source/_static/switcher.json"), mode="r") as f:
    txt = f.read()

print(f"python/V.{release_version}")
assert f"python/V.{release_version}" in txt
