import os
import sys

npth = os.path.abspath(".")
if npth not in sys.path:
    sys.path.append(npth)
    print(npth)

with open("../../__version__.py") as f:
    exec(f.read())

# We check if the reference to all existing versions were added by checking
# that the current version is referenced
with open(os.path.join(npth, "docs/source/version_history.rst"), mode="r") as f:
    txt = f.read()

assert f"`{release_version}" in txt
assert f"V.{release_version}" in txt
