from libc.stddef cimport size_t


cdef extern from "array_hash_map.hpp" namespace "aequilibrae::utils::cpp" nogil:
    cdef cppclass CppArrayHashMap "aequilibrae::utils::cpp::ArrayHashMap"[Key, Value]:
        CppArrayHashMap(const Key *, const Value *, size_t) except +
        size_t lookup(const Key *, Value *, size_t) except +
