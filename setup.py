import sys
import os
import numpy as np
from setuptools import setup, find_packages
from setuptools import Extension
from Cython.Distutils import build_ext
from aequilibrae.paths.__version__ import release_version

sys.dont_write_bytecode = True

here = os.path.dirname(os.path.realpath(__file__))
whole_path = os.path.join(here, "aequilibrae/paths", "AoN.pyx")
ext_module = Extension("aequilibrae.paths.AoN", [whole_path], include_dirs=[np.get_include()])

whole_path2 = os.path.join(here, "aequilibrae/paths", "TrafficAssignmentCy.pyx")
whole_path3 = os.path.join(here, "aequilibrae/paths", "TrafficAssignment.cpp")
ext_module2 = Extension(
    "aequilibrae.paths.TrafficAssignmentCy",
    [whole_path2, whole_path3],
    language="c++",
    extra_compile_args=["-ffast-math"],
    include_dirs=[np.get_include()],
)

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
        install_requires=["numpy", "PyQt5", "pyaml", "pandas", "requests", "shapely", "scipy", "cvxopt"],
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
        ext_modules=[ext_module, ext_module2],
    )
