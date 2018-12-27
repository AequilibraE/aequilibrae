import sys
import os
import numpy as np
from setuptools import setup, find_packages
from setuptools import Extension
from distutils.core import setup as cython_setup
from Cython.Distutils import build_ext

sys.dont_write_bytecode = True

here = os.path.dirname(os.path.realpath(__file__))
whole_path = os.path.join(here, 'aequilibrae/paths', 'AoN.pyx')
ext_module = Extension('AoN',
                       [whole_path],
                       include_dirs=[np.get_include()])
if __name__ == "_  _main__":
    with open("README.md", "r") as fh:
        long_description = fh.read()

    setup(install_requires=['numpy', 'cython'],
          packages=[pkg for pkg in find_packages('aequilibrae')],
          package_dir={'': 'aequilibrae'},
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
          cmdclass={"build_ext": build_ext},
          ext_modules=[ext_module]
          )