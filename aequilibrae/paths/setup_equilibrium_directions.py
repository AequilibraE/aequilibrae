import os
import sys
import platform
import numpy as np
import Cython.Compiler.Options
from Cython.Distutils import build_ext
from Cython.Build import cythonize

# Cython.Compiler.Options.annotate = True

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

sys.dont_write_bytecode = True

if "WINDOWS" in platform.platform().upper():
    ext_modules = [
        Extension(
            "equilibrium_directions",
            ["equilibrium_directions.pyx"],
            extra_compile_args=['/openmp'],
            extra_link_args=['/openmp'],
            include_dirs=[np.get_include()],
        )
    ]
else:
    ext_modules = [
        Extension(
            "equilibrium_directions",
            ["equilibrium_directions.pyx"],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'],
            include_dirs=[np.get_include()],
        )
    ]

setup(
    name='equilibrium_directions',
    ext_modules=cythonize(ext_modules),
)