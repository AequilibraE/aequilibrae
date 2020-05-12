import os
import sys
import shutil

npth = os.path.abspath(".")
try:
    from aequilibrae.paths.__version__ import release_version
except ImportError as e:
    sys.path.insert(0, npth)
    from aequilibrae.paths.__version__ import release_version
    import warnings

    warnings.warn(f"It is really annoying to deal with Flake8 sometimes. {e.args}")

version = f"V.{release_version}"

docs = npth + "/docs/build/html"
docs_dest = npth + f"/docs/build/htmlv/{version}"
shutil.copytree(docs, docs_dest)
