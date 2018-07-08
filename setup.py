import sys
import os
import numpy as np
from setuptools import setup
from setuptools import Extension
from distutils.core import setup as cython_setup

sys.dont_write_bytecode = True

here = os.path.dirname(os.path.realpath(__file__))
whole_path = os.path.join(here, 'aequilibrae/paths','setup_Assignment.py')

print('python ' + whole_path + ' build_ext --inplace')
os.system('python ' + whole_path + ' build_ext --inplace')


if __name__ == "__main__":
    with open("README.md", "r") as fh:
        long_description = fh.read()

    setup(install_requires=['numpy', 'cython'],
          packages=['aequilibrae'],
          zip_safe=False,
          name='aequilibrae',
          long_description=long_description,
          long_description_content_type="text/markdown",
          version='0.5.0',
          description="A package for transportation modeling",
          author="Pedro Camargo",
          author_email="pedro@xl-optim.com",
          url="https://github.com/AequilibraE/aequilibrae",
          license='See license.txt',
          classifiers=[
              'Programming Language :: Python',
              'Programming Language :: Python :: 3.5',
              'Programming Language :: Python :: 3.6',
              'Programming Language :: Python :: 3.7',
          ],
          )
