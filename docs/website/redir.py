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

with open(os.path.join(npth, "docs/website/index.html"), mode="r") as f:
    txt = f.read()

version = f"V.{release_version}"
txt = txt.replace("VERSION", version)
with open(os.path.join(npth, "docs/website/index.html"), mode="w") as f:
    f.write(txt)

docs = npth + "/docs/build/html"
docs_dest = npth + f"/docs/build/htmlv/{version}"
shutil.copytree(docs, docs_dest)


# We check if the reference to all existing versions were added by checking
# that the current version is referenced
with open(os.path.join(npth, "docs/source/index.rst"), mode="r") as f:
    txt = f.read()

assert(release_version in txt)

