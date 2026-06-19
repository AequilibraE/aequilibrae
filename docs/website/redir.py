import os
import shutil
import sys
from importlib.metadata import version

npth = os.path.abspath(".")
if npth not in sys.path:
    sys.path.append(npth)

release_version = version("aequilibrae")
version_folder = f"V.{release_version}"

docs = npth + "/docs/build/html"
docs_dest = npth + f"/docs/build/htmlv/{version_folder}"
shutil.copytree(docs, docs_dest)
