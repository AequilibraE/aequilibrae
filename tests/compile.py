import os
import subprocess
import sys
from os.path import join, dirname

pth = join(dirname(dirname(__file__)), "aequilibrae", "paths")

os.chdir(pth)
subprocess.Popen(
    f"{sys.executable} setup_assignment.py build_ext --inplace", shell=True, stdout=subprocess.PIPE
).stdout.read()
