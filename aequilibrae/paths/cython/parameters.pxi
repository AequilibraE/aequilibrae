import numpy as np
cimport numpy as cnp

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

ITYPE = np.int64
ctypedef cnp.int64_t ITYPE_t

# EPS is the precision of DTYPE
cdef DTYPE_t DTYPE_EPS = 1E-15

# NULL_IDX is the index used in predecessor matrices to store a non-path
cdef ITYPE_t NULL_IDX = 18446744073709551615
