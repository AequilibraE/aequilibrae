from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdint cimport int64_t

cdef extern from "ParquetWriter.h":
    cdef cppclass ParquetWriter:
        ParquetWriter() except +
        int write_parquet(vector[int64_t] vec, string filename)