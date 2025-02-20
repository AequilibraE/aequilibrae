import os
import subprocess
import sys
from pathlib import Path

project_dir = Path(__file__).parent.parent
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))

def zip_html():
    """
    Compress HTML documentation into a zip file.

    Borrowed from Pandas (see pandas/doc/make.py)
    """
    print("oi")
    zip_fname = os.path.join("docs", "build", "html", "AequilibraE.zip")
    if os.path.exists(zip_fname):
        os.remove(zip_fname)  # noqa: TID251
    dirname = os.path.join("docs", "build", "html")
    print("dirname: ", dirname)
    fnames = os.listdir(dirname)
    print("fnames: ", fnames)
    os.chdir(dirname)
    # subprocess.check_call(["zip", zip_fname, "-r", "-q", *fnames], stdout=sys.stdout, stderr=sys.stderr)
