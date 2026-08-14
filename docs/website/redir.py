import os
import sys
import shutil
from importlib.metadata import version as distribution_version

npth = os.path.abspath(".")
if npth not in sys.path:
    sys.path.append(npth)

release_version = distribution_version("aequilibrae")

version = f"V.{release_version}"

docs = npth + "/docs/build/html"
docs_dest = npth + f"/docs/build/htmlv/{version}"
shutil.copytree(docs, docs_dest)
