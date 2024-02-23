import cython

@cython.embedsignature(True)
cdef class SparseMatrix:
    """
    A class to implement sparse matrix operations such as reading, writing, and indexing
    """

    def __cinit__(self):
        """C level init. For C memory allocation and initialisation. Called exactly once per object."""
        pass

    def __init__(self):
        """Python level init, may be called multiple times, for things that can't be done in __cinit__."""
        pass

    def __dealloc__(self):
        """
        C level deallocation. For freeing memory allocated by this object. *Must* have GIL, `self` may be in a
        partially deallocated state already.
        """
        pass

    cpdef void helloworld(SparseMatrix self):
        print("Hello from cpdef")
