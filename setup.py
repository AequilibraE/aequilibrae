from __future__ import division, print_function, absolute_import
import setuptools
import sys

sys.dont_write_bytecode = True

import numpy as np

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Distutils import build_ext

import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True

src_dir = 'aequilibrae/paths'
ext_module = Extension(src_dir + 'AoN',
                       [src_dir + "AoN.pyx"],
                       include_dirs=[np.get_include()])

setup(cmdclass={'build_ext': build_ext},
      ext_modules=[ext_module])

with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == "__main__":
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
          url="https://github.com/AequilibraE/pyquilibrae",
          license='See license.txt',
          cmdclass={"build_ext": build_ext},
          ext_modules=[ext_module]
          )
