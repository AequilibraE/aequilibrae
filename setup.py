import importlib.util as iutil
import logging
import os
import platform
from os.path import dirname, join, isfile

import numpy as np
from Cython.Distutils import build_ext
from setuptools import Extension
from setuptools import setup, find_packages

with open("__version__.py") as f:
    exec(f.read())

include_dirs = [np.get_include()]
if iutil.find_spec("pyarrow") is not None:
    import pyarrow as pa

    include_dirs.append(pa.get_include())

is_win = "WINDOWS" in platform.platform().upper()
is_mac = "MACOS" in platform.platform().upper()
logging.error(platform.platform())
print(platform.platform())
prefix = "/" if is_win else "-f"
cpp_std = "/std:c++17" if is_win else "-std=c++17"
compile_args = [cpp_std, "-Wno-unreachable-code"] if is_mac else [f"{prefix}openmp", cpp_std]
link_args = [] if is_mac else [f"{prefix}openmp"]

ext_mod_aon = Extension(
    "aequilibrae.paths.AoN",
    [join(dirname(os.path.realpath(__file__)), "aequilibrae/paths", "AoN.pyx")],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    include_dirs=include_dirs,
    language="c++",
)

ext_mod_ipf = Extension(
    "aequilibrae.distribution.ipf_core",
    [join(dirname(os.path.realpath(__file__)), "aequilibrae/distribution", "ipf_core.pyx")],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    include_dirs=include_dirs,
    language="c++",
)

with open("requirements.txt", "r") as fl:
    install_requirements = [x.strip() for x in fl.readlines()]

pkgs = [pkg for pkg in find_packages()]

pkg_data = {
    "aequilibrae.reference_files": ["spatialite.sqlite", "nauru.zip", "sioux_falls.zip"],
    "aequilibrae.paths": ["parameters.pxi", "*.pyx"],
    "aequilibrae.distribution": ["*.pyx"],
    "aequilibrae": ["./parameters.yml"],
    "aequilibrae.project": ["database_specification/tables/*.*", "database_specification/triggers/*.*"],
}
loose_modules = ["__version__", "parameters"]

if __name__ == "__main__":
    setup(
        name="aequilibrae",
        version=release_version,  # noqa: F821
        # TODO: Fix the requirements and optional requirements to bring directly from the requirements file
        install_requires=install_requirements,
        packages=pkgs,
        package_dir={"": "."},
        py_modules=loose_modules,
        package_data=pkg_data,
        zip_safe=False,
        description="A package for transportation modeling",
        author="Pedro Camargo",
        author_email="c@margo.co",
        url="https://github.com/AequilibraE/aequilibrae",
        license="See license.txt",
        classifiers=[
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
        ],
        cmdclass={"build_ext": build_ext},
        ext_modules=[ext_mod_aon, ext_mod_ipf],
    )
