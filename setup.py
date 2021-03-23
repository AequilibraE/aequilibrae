import sys
import os
import platform
import numpy as np
import pyarrow as pa
from setuptools import setup, find_packages
from setuptools import Extension
from Cython.Distutils import build_ext
from aequilibrae.paths.__version__ import release_version

sys.dont_write_bytecode = True

here = os.path.dirname(os.path.realpath(__file__))
whole_path = os.path.join(here, "aequilibrae/paths", "AoN.pyx")
ext_module = Extension(
    "aequilibrae.paths.AoN", [whole_path], include_dirs=[np.get_include(), pa.get_include()], language="c++"
)

# this is for building pyarrow on platforms w/o wheel, like our one of our macos/python combos
if "WINDOWS" not in platform.platform().upper():
    ext_module.extra_compile_args.append("-std=c++11")

pkgs = [pkg for pkg in find_packages()]

pkg_data = {
    "aequilibrae.reference_files": ["spatialite.sqlite", "nauru.zip", "sioux_falls.zip"],
    "aequilibrae.paths": ["parameters.pxi"],
    "aequilibrae": ["parameters.yml"],
    "aequilibrae.project": ["database_specification/tables/*.*", "database_specification/triggers/*.*"],
}
loose_modules = ["__version__", "parameters"]

if __name__ == "__main__":
    setup(
        name="aequilibrae",
        version=release_version,
        # TODO: Fix the requirements and optional requirements to bring directly from the requirements file
        install_requires=["numpy", "PyQt5", "pyaml", "pandas", "requests", "shapely", "scipy", "pyarrow"],
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
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
        ],
        cmdclass={"build_ext": build_ext},
        ext_modules=[ext_module],
    )
