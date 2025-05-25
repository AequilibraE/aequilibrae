import importlib.util as iutil
import platform
from os.path import join
import os

import numpy as np
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup, find_packages
from setuptools.discovery import FlatLayoutPackageFinder
from multiprocessing import cpu_count

include_dirs = [np.get_include()]
libraries = []
library_dirs = []

is_win = "WINDOWS" in platform.platform().upper()
is_mac = any(e in platform.platform().upper() for e in ["MACOS", "DARWIN"])
prefix = "/" if is_win else "-f"
cpp_std = "/std:c++17" if is_win else "-std=c++17"
compile_args = [cpp_std, f"{prefix}openmp"]
compile_args += ["-Wno-unreachable-code"] if is_mac else []
link_args = [f"{prefix}openmp"]

if os.getenv("AEQ_DEBUG"):
    compile_args.extend(["-O0", "-g"])

if os.getenv("AEQ_ASAN"):
    compile_args.append(f"{prefix}sanitize=address")
    link_args.append(f"{prefix}sanitize=address")

    if not is_win:
        compile_args.append(f"{prefix}sanitize=undefined")
        link_args.append(f"{prefix}sanitize=undefined")


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
    [join("aequilibrae", "distribution", "cython", "ipf_core.pyx")],
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

ext_mod_coo_demand = Extension(
    "aequilibrae.matrix.coo_demand",
    [join("aequilibrae", "matrix", "coo_demand.pyx")],
    **extension_args,
)


if __name__ == "__main__":
    setup(
        packages=find_packages(exclude=FlatLayoutPackageFinder.DEFAULT_EXCLUDE),
        package_dir={"": "."},
        zip_safe=False,
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
            nthreads=min(60, cpu_count()),  # Windows does not do well with more than 60 threads
        ),
    )
