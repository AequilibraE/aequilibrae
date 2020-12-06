# from distutils.core import setup
from Cython.Build import cythonize
from setuptools import setup
from distutils.extension import Extension

# ext_1 = Extension(
#     "quadraticassignmentcyt",
#     # our Cython source
#     sources=[
#         "quadraticassignmentcyt.pyx",
#         "ShortestPathComputation.cpp",
#         "TrafficAssignment.cpp",
#     ],  # additional source file(s)
#     language="c++",  # generate C++ code,
#     extra_compile_args=["-ffast-math"],
# )

# ext_2 = Extension(
#     "spPath",
#     # our Cython source
#     sources=["spPath.pyx", "ShortestPathComputation.cpp", "TrafficAssignment.cpp"],  # additional source file(s)
#     language="c++",  # generate C++ code,
#     extra_compile_args=["-ffast-math"],
# )

# ext_3 = Extension(
#     "cyPath",
#     # our Cython source
#     sources=["cyPath.pyx", "ShortestPathComputation.cpp", "TrafficAssignment.cpp"],  # additional source file(s)
#     language="c++",  # generate C++ code,
#     extra_compile_args=["-ffast-math"],
# )

ext_4 = Extension(
    "TrafficAssignmentCy",
    # our Cython source
    sources=[
        "TrafficAssignmentCy.pyx",
        "ShortestPathComputation.cpp",
        "TrafficAssignment.cpp",
    ],  # additional source file(s)
    language="c++",  # generate C++ code,
    extra_compile_args=["-ffast-math"],
)

module_list = [ext_4]  # ext_1, ext_2, ext_3,

setup(ext_modules=cythonize(module_list))

# setup(ext_modules = cythonize(Extension("quadraticassignmentcyt",
#                          # our Cython source
#            sources=["quadraticassignmentcyt.pyx", "ShortestPathComputation.cpp", "TrafficAssignment.cpp"],  # additional source file(s)
#            language="c++"          # generate C++ code,
#       )))
