import os
import sys
import shutil

npth = os.path.abspath(".")
try:
    from aequilibrae.paths.__version__ import release_version
except ImportError as e:
    sys.path.insert(0, npth)
    from aequilibrae.paths.__version__ import release_version

# We check if the reference to all existing versions were added by checking
# that the current version is referenced
with open(os.path.join(npth, "docs/source/index.rst"), mode="r") as f:
    txt = f.read()

assert(release_version in txt)

