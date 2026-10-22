cimport cython
import numpy as np
import pandas as pd
from scipy.sparse import coo_array
from libcpp.memory cimport make_unique
from aequilibrae.matrix.aequilibrae_matrix import _zone_ids


cdef void _insert_many(
    COOStorage[cython.floating] *storage,
    const size_t[::1] rows,
    const size_t[::1] columns,
    const cython.floating[:, ::1] values,
    size_t field_count
) except *:
    cdef size_t count = rows.shape[0]
    if count == 0:
        return

    storage.rows.insert(storage.rows.end(), &rows[0], &rows[0] + count)
    storage.columns.insert(storage.columns.end(), &columns[0], &columns[0] + count)
    if field_count:
        storage.values.insert(
            storage.values.end(), &values[0, 0], &values[0, 0] + count * field_count
        )


@cython.final
cdef class SparseMatrixBuilder(_MatrixBase):
    """Own COO coordinates and contiguous [entries, fields] values."""

    def __init__(self, zone_count, field_names, dtype=np.float64, *, origin_count=None):
        if self.field_names is not None:
            raise RuntimeError("SparseMatrixBuilder cannot be reinitialised")
        super().__init__(zone_count, field_names, dtype, origin_count=origin_count)

        # Can't use a fused type on an extension type attribute, it's fine if
        # we just use two. Otherwise it would be a void*
        self._float32 = self.dtype == np.dtype(np.float32)
        if self._float32:
            self.storage32 = make_unique[COOStorage[float]]()
        else:
            self.storage64 = make_unique[COOStorage[double]]()

    @classmethod
    def from_df(cls, df, row, col, subset=None, *, index=None, index_name="main_index" , dtype=np.float64):
        if subset is None:
            subset = [name for name in df.columns if name not in (row, col)]

        rows, columns = _zone_ids(df[row].to_numpy()), _zone_ids(df[col].to_numpy())
        ids = np.union1d(rows, columns) if index is None else _zone_ids(index)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("index must contain unique zone IDs")

        mapping = pd.Index(ids)
        r, c = mapping.get_indexer(rows), mapping.get_indexer(columns)
        if np.any(r < 0) or np.any(c < 0):
            raise ValueError("found row or column values that are not in the index")

        cdef SparseMatrixBuilder result = cls(len(ids), subset, dtype=dtype)
        result.append_many(r, c, df.loc[:, list(result.field_names)].to_numpy(dtype=result.dtype))
        return result

    def append(self, origin, destination, values):
        """Append one complete field row using zero-based coordinates."""
        self.append_many([origin], [destination], np.asarray(values).reshape(1, -1))

    def append_many(self, rows, columns, values):
        """Copy COO records"""
        rows, columns = np.asarray(rows), np.asarray(columns)
        if rows.ndim != 1 or columns.shape != rows.shape:
            raise ValueError("COO coordinates must be equally sized one-dimensional arrays")

        for coordinates, count in ((rows, self.origin_count), (columns, self.destination_count)):
            if coordinates.size and (
                    coordinates.dtype.kind not in "iu"
                    or np.any(coordinates < 0)
                    or np.any(coordinates >= count)
            ):
                raise ValueError("COO coordinates must be integer indices within the matrix shape")

        data = np.ascontiguousarray(values, dtype=self.dtype)
        if data.shape != (rows.size, self.field_count):
            raise ValueError("values must have shape (entries, fields)")

        cdef const size_t[::1] r = np.ascontiguousarray(rows, dtype=np.uintp)
        cdef const size_t[::1] c = np.ascontiguousarray(columns, dtype=np.uintp)
        cdef const float[:, ::1] v32
        cdef const double[:, ::1] v64
        if self._float32:
            v32 = data
            _insert_many[float](self.storage32.get(), r, c, v32, self.field_count)
        else:
            v64 = data
            _insert_many[double](self.storage64.get(), r, c, v64, self.field_count)

    def reset(self):
        """Clear entries, retaining capacity."""
        if self._float32:
            self.view[float]().reset()
        else:
            self.view[double]().reset()

    def extend(self, SparseMatrixBuilder source not None):
        """Append another builder with matching dimensions, dtype and field order."""
        if self.dtype != source.dtype or self.field_names != source.field_names:
            raise ValueError("builder dtype and ordered field names must match")

        if self._float32:
            self.view[float]().extend(source.view[float]())
        else:
            self.view[double]().extend(source.view[double]())

    cdef object _arrays(self, cython.floating _type = 0):
        # Return snapshots, not memoryviews into reallocatable vector buffers.
        cdef size_t n, count
        cdef COOStorage[cython.floating] *storage
        if (cython.floating is float and not self._float32) or (cython.floating is double and self._float32):
            raise TypeError("view type must match the SparseMatrixBuilder dtype")

        if cython.floating is float:
            storage = self.storage32.get()
        else:
            storage = self.storage64.get()

        n = storage.rows.size()
        count = storage.values.size()
        if n:
            rows = np.asarray(<size_t[:n]>storage.rows.data()).copy()
            cols = np.asarray(<size_t[:n]>storage.columns.data()).copy()
        else:
            rows = np.empty(0, dtype=np.uintp)
            cols = np.empty(0, dtype=np.uintp)

        if count:
            values = np.asarray(<cython.floating[:count]>storage.values.data()).copy()
        else:
            values = np.empty(0, dtype=self.dtype)

        return rows, cols, values.reshape(n, self.field_count)

    @property
    def entry_count(self):
        if self._float32:
            return self.storage32.get().rows.size()
        return self.storage64.get().rows.size()

    def to_df(self, row="origin", col="destination"):
        """Return independent OD records using the original zone IDs."""
        if self.origin_count != self.destination_count:
            raise ValueError("zone-labelled DataFrames require a square builder")
        if row == col or row in self.field_names or col in self.field_names:
            raise ValueError("coordinate column names must be distinct from the demand fields")

        rows, cols, values = self.records()

        result = pd.DataFrame(values, columns=self.field_names)
        result.insert(0, col, cols)
        result.insert(0, row, rows)
        return result

    def records(self):
        """Return independent row, column and [entries, fields] arrays."""
        if self._float32:
            return self._arrays[float]()
        else:
            return self._arrays[double]()


    def to_scipy(self):
        """Return independent named CSR arrays"""
        rows, cols, values = self.records()

        result = {}
        for field, name in enumerate(self.field_names):
            nonzero = values[:, field] != 0

            result[name] = coo_array(
                (values[nonzero, field], (rows[nonzero], cols[nonzero])),
                shape=(self.origin_count, self.destination_count),
            ).tocsr()

        return result

    cdef CppSparseMatrixBuilderView[cython.floating] view(self, cython.floating _type=0) except * nogil:
        if (cython.floating is float and not self._float32) or (cython.floating is double and self._float32):
            with gil:
                raise TypeError("view type must match the SparseMatrixBuilder dtype")

        cdef CppSparseMatrixBuilderView[cython.floating] result
        result.origin_count = self.origin_count
        result.destination_count = self.destination_count
        result.field_count = self.field_count

        if cython.floating is float:
            result.storage = self.storage32.get()
        else:
            result.storage = self.storage64.get()
        return result
