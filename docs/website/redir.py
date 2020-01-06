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

    warnings.warn("It is really annoying to deal with Flake8 sometimes. {}".format(e.args))

f = open(os.path.join(npth, "docs/website/index.html"), mode="r")
txt = f.read()
f.close()

version = "V.{}".format(release_version)
txt = txt.replace("VERSION", version)
f = open(os.path.join(npth, "docs/website/index.html"), mode="w")
f.write(txt)
f.close()

docs = npth + "/docs/build/html"
docs_dest = npth + "/docs/build/htmlv/{}".format(version)
shutil.copytree(docs, docs_dest)
