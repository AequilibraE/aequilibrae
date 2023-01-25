import numpy as np
import platform
from Cython.Build import cythonize

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

if "WINDOWS" in platform.platform().upper():
    ext_modules = [
        Extension(
            "ipf_core",
            ["ipf_core.pyx"],
            extra_compile_args=["/openmp"],
            extra_link_args=["/openmp"],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            include_dirs=[np.get_include()],
        )
    ]
else:
    ext_modules = [
        Extension(
            "ipf_core",
            ["ipf_core.pyx"],
            extra_compile_args=["-fopenmp"],  # do we want -Ofast?
            extra_link_args=["-fopenmp"],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            include_dirs=[np.get_include()],
        )
    ]

setup(name="ipf_core", ext_modules=cythonize(ext_modules))
