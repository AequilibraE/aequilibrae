import importlib.util as iutil
import os
import platform
from os.path import dirname, join, isfile

import numpy as np
from Cython.Distutils import build_ext
from setuptools import Extension
from setuptools import setup, find_packages

from aequilibrae.paths.__version__ import release_version

spec = iutil.find_spec("pyarrow")

include_dirs = [np.get_include()]
if spec is not None:
    import pyarrow as pa

    include_dirs.append(pa.get_include())

whole_path = join(dirname(os.path.realpath(__file__)), "aequilibrae/paths", "AoN.pyx")
ext_module = Extension("aequilibrae.paths.AoN", [whole_path], include_dirs=include_dirs, language="c++")

# this is for building pyarrow on platforms w/o wheel, like our one of our macos/python combos
if "WINDOWS" not in platform.platform().upper():
    ext_module.extra_compile_args.append("-std=c++17")

reqs = ["numpy>=1.18.0,<1.22", "scipy", "pyaml", "cython", "pyshp", "requests", "shapely >= 1.7.0", "pandas", "pyproj"]

if isfile("requirements.txt"):
    # We just make sure to keep the requirements sync'ed with the setup file
    with open("requirements.txt", "r") as fl:
        install_requirements = [x.strip() for x in fl.readlines()]
    assert sorted(install_requirements) == sorted(reqs)

pkgs = [pkg for pkg in find_packages()]

pkg_data = {
    "aequilibrae.reference_files": ["spatialite.sqlite", "nauru.zip", "sioux_falls.zip"],
    "aequilibrae.paths": ["parameters.pxi"],
    "aequilibrae": ["parameters.yml"],
    "aequilibrae.project": ["database_specification/tables/*.*", "database_specification/triggers/*.*",
                            "database_specification/transit/*.*"],
}
loose_modules = ["__version__", "parameters"]

if __name__ == "__main__":
    setup(
        name="aequilibrae",
        version=release_version,
        # TODO: Fix the requirements and optional requirements to bring directly from the requirements file
        install_requires=reqs,
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
        ],
        cmdclass={"build_ext": build_ext},
        ext_modules=[ext_module],
    )
