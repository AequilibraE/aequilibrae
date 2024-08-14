import os
import sys
import shutil
import pkg_resources

npth = os.path.abspath(".")
if npth not in sys.path:
    sys.path.append(npth)

release_version = pkg_resources.get_distribution("aequilibrae").version

version = f"V.{release_version}"

docs = npth + "/docs/build/html"
docs_dest = npth + f"/docs/build/htmlv/{version}"
shutil.copytree(docs, docs_dest)
