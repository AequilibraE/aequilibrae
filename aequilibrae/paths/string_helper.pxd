from libcpp.string cimport string

cdef extern from "<string>" namespace "std" nogil:
    string to_string(long) except +
    string to_string(long long) except +
    string to_string(int) except +
