import sys
import platform
import numpy as np
import pyarrow as pa
import Cython.Compiler.Options
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import shutil

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
            "AoN",
            ["AoN.pyx"],
            extra_compile_args=["/openmp", "/O2"],
            extra_link_args=["/openmp", "/parquet"],
            include_dirs=[np.get_include(), pa.get_include()],
            language="c++",
        )
    ]
else:
    # NOTE: on linux and mac, create appropriately named symlinks after pip install pyarrow with
    # python -c "import pyarrow; pyarrow.create_library_symlinks()"
    # Only needs to be done once
    pa.create_library_symlinks()

    ext_modules = [
        Extension(
            "AoN",
            ["AoN.pyx"],
            extra_compile_args=["-fopenmp", "-std=c++11", "-O3"],  # do we want -Ofast?
            extra_link_args=["-fopenmp", "-lparquet"],
            include_dirs=[np.get_include(), pa.get_include()],
            language="c++",
            # I got inexplicable segfaults without the following line, see
            # https://arrow.apache.org/docs/python/extending.html# (see end of doc)
            define_macros=[("_GLIBCXX_USE_CXX11_ABI", "0")],
            # rpath only for *nix, we hack it in __init__ for win
            runtime_library_dirs=pa.get_library_dirs(),
        )
    ]

for ext in ext_modules:
    ext.libraries.extend(pa.get_libraries())
    ext.libraries.extend(["parquet"])
    ext.library_dirs.extend(pa.get_library_dirs())

setup(name="AoN", ext_modules=cythonize(ext_modules))
