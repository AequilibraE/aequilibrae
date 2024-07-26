import importlib.util as iutil
import platform
from os.path import join

import numpy as np
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup, find_packages
from setuptools.discovery import FlatLayoutPackageFinder

with open("__version__.py") as f:
    exec(f.read())

include_dirs = [np.get_include()]
libraries = []
library_dirs = []
if iutil.find_spec("pyarrow") is not None:
    import pyarrow as pa

    pa.create_library_symlinks()
    include_dirs.append(pa.get_include())
    libraries.extend(pa.get_libraries())
    library_dirs.extend(pa.get_library_dirs())

is_win = "WINDOWS" in platform.platform().upper()
is_mac = any(e in platform.platform().upper() for e in ["MACOS", "DARWIN"])
prefix = "/" if is_win else "-f"
cpp_std = "/std:c++17" if is_win else "-std=c++17"
compile_args = [cpp_std, f"{prefix}openmp{':llvm' if is_win else ''}"]
compile_args += ["-Wno-unreachable-code"] if is_mac else []
link_args = [f"{prefix}openmp"]

extension_args = {
    "extra_compile_args": compile_args,
    "extra_link_args": link_args,
    "define_macros": [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    "include_dirs": include_dirs,
    "libraries": libraries,
    "library_dirs": library_dirs,
    "language": "c++",
}

ext_mod_aon = Extension("aequilibrae.paths.AoN", [join("aequilibrae", "paths", "cython", "AoN.pyx")], **extension_args)

ext_mod_ipf = Extension(
    "aequilibrae.distribution.ipf_core",
    [join("aequilibrae", "distribution", "ipf_core.pyx")],
    **extension_args,
)

ext_mod_put = Extension(
    "aequilibrae.paths.public_transport",
    [join("aequilibrae", "paths", "cython", "public_transport.pyx")],
    **extension_args,
)

ext_mod_rc = Extension(
    "aequilibrae.paths.cython.route_choice_set",
    [join("aequilibrae", "paths", "cython", "route_choice_set.pyx")],
    **extension_args,
)

ext_mod_coo_demand = Extension(
    "aequilibrae.paths.cython.coo_demand",
    [join("aequilibrae", "paths", "cython", "coo_demand.pyx")],
    **extension_args,
)

ext_mod_rc_ll_results = Extension(
    "aequilibrae.paths.cython.route_choice_link_loading_results",
    [join("aequilibrae", "paths", "cython", "route_choice_link_loading_results.pyx")],
    **extension_args,
)

ext_mod_rc_set_results = Extension(
    "aequilibrae.paths.cython.route_choice_set_results",
    [join("aequilibrae", "paths", "cython", "route_choice_set_results.pyx")],
    **extension_args,
)

ext_mod_graph_building = Extension(
    "aequilibrae.paths.graph_building",
    [join("aequilibrae", "paths", "cython", "graph_building.pyx")],
    **extension_args,
)

ext_mod_sparse_matrix = Extension(
    "aequilibrae.matrix.sparse_matrix",
    [join("aequilibrae", "matrix", "sparse_matrix.pyx")],
    **extension_args,
)

with open("requirements.txt", "r") as fl:
    install_requirements = [x.strip() for x in fl.readlines()]

pkgs = find_packages(exclude=FlatLayoutPackageFinder.DEFAULT_EXCLUDE)

pkg_data = {
    "aequilibrae.reference_files": ["spatialite.sqlite", "nauru.zip", "sioux_falls.zip", "coquimbo.zip"],
    "aequilibrae.paths": ["parameters.pxi", "*.pyx"],
    "aequilibrae.distribution": ["*.pyx"],
    "aequilibrae": ["./parameters.yml"],
    "aequilibrae.project": [
        "database_specification/network/tables/*.*",
        "database_specification/network/triggers/*.*",
        "database_specification/transit/tables/*.*",
        "database_specification/transit/triggers/*.*",
    ],
}

with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == "__main__":
    setup(
        name="aequilibrae",
        version=release_version,  # noqa: F821
        install_requires=install_requirements,
        packages=pkgs,
        package_dir={"": "."},
        package_data=pkg_data,
        zip_safe=False,
        description="A package for transportation modeling",
        long_description=long_description,
        author="Pedro Camargo",
        author_email="c@margo.co",
        url="https://github.com/AequilibraE/aequilibrae",
        license="See LICENSE.TXT",
        license_files=("LICENSE.TXT",),
        classifiers=[
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ],
        cmdclass={"build_ext": build_ext},
        ext_modules=cythonize(
            [
                ext_mod_aon,
                ext_mod_ipf,
                ext_mod_put,
                ext_mod_rc,
                ext_mod_coo_demand,
                ext_mod_rc_ll_results,
                ext_mod_rc_set_results,
                ext_mod_graph_building,
                ext_mod_sparse_matrix,
            ],
            compiler_directives={"language_level": "3str"},
        ),
    )
