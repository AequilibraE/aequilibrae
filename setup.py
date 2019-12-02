import sys
import os
import numpy as np
from setuptools import setup, find_packages
from setuptools import Extension
from Cython.Distutils import build_ext

sys.dont_write_bytecode = True

here = os.path.dirname(os.path.realpath(__file__))
whole_path = os.path.join(here, "aequilibrae/paths", "AoN.pyx")
ext_module = Extension("paths.AoN", [whole_path], include_dirs=[np.get_include()])

pkgs = [pkg for pkg in find_packages()]

pkg_data = {
    "aequilibrae.reference_files": ["spatialite.sqlite"],
    "aequilibrae.paths": ["parameters.pxi"],
    "aequilibrae": ["parameter_default.yml", "parameters.yml"],
}
loose_modules = ["__version__", "parameters", "reserved_fields"]


if __name__ == "__main__":
    with open("README.rst", "r") as fh:
        long_description = fh.read()
    setup(
        name="aequilibrae",
        version="0.5.1",
        install_requires=["numpy", "cython"],
        packages=pkgs,
        package_dir={"": "."},
        py_modules=loose_modules,
        package_data=pkg_data,
        zip_safe=False,
        long_description=long_description,
        long_description_content_type="text/markdown",
        description="A package for transportation modeling",
        author="Pedro Camargo",
        author_email="pedro@xl-optim.com",
        url="https://github.com/AequilibraE/aequilibrae",
        license="See license.txt",
        classifiers=[
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
        ],
        cmdclass={"build_ext": build_ext},
        ext_modules=[ext_module],
    )
