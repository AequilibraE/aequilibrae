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

if "WINDOWS" in platform.platform().upper():
    ext_module.extra_compile_args.extend(["/openmp", "/O2"])
    ext_module.extra_link_args.extend(["/openmp"])
else:
    # NOTE: on linux and mac, create appropriately named symlinks after pip install pyarrow with
    # python -c "import pyarrow; pyarrow.create_library_symlinks()"
    # Only needs to be done once
    pa.create_library_symlinks()
    ext_module.extra_compile_args.extend(["-fopenmp", "-std=c++11", "-O3"])  # do we want -Ofast?
    ext_module.extra_link_args = ["-fopenmp", "-lparquet"]
    # I got inexplicable segfaults without the following line, see
    # https://arrow.apache.org/docs/python/extending.html# (see end of doc)
    ext_module.define_macros.extend([("_GLIBCXX_USE_CXX11_ABI", "0")])
    # rpath only for *nix, we hack it in __init__ for win
    ext_module.runtime_library_dirs.extend(pa.get_library_dirs())
ext_module.libraries.extend(pa.get_libraries())
ext_module.libraries.extend(["parquet"])
ext_module.library_dirs.extend(pa.get_library_dirs())

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
