from os.path import join, dirname
import os
import subprocess


pth = join(dirname(dirname(__file__)), 'aequilibrae', 'paths')

os.chdir(pth)
subprocess.Popen("python setup_Assignment.py build_ext --inplace", shell=True, stdout=subprocess.PIPE).stdout.read()

