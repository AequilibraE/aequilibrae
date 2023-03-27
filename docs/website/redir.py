import os
import sys
import shutil

npth = os.path.abspath(".")
if npth not in sys.path:
    sys.path.append(npth)

from __version__ import release_version


version = f"V.{release_version}"

docs = npth + "/docs/build/html"
docs_dest = npth + f"/docs/build/htmlv/{version}"
shutil.copytree(docs, docs_dest)
