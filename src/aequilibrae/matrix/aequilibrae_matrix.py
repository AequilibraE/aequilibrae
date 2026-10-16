import os
from abc import abstractmethod
from collections.abc import Iterator, MutableMapping
from copy import deepcopy
from typing import Any, Callable, Literal, Mapping, Self, Sequence, TypeAlias, overload
from uuid import uuid4

import h5py
import numpy as np
import openmatrix as omx
import pandas as pd
import scipy.sparse
from numpy.typing import ArrayLike, DTypeLike

OMX_SPECIAL_ATTRS = {"OMX_VERSION", "OMX_CREATED_WITH", "SHAPE", "TITLE"}


def _copy_file_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy({key: value for key, value in metadata.items() if key not in OMX_SPECIAL_ATTRS})


def _cast_index(value: np.ndarray, dataset: h5py.Dataset) -> np.ndarray:
    if dataset.shape != value.shape:
        raise ValueError(f"index shape may not change, expected {dataset.shape}, got {value.shape}")
    return value.astype(dataset.dtype, casting="same_value")


def _write_index(file: omx.File, name: str, value: np.ndarray) -> None:
    dataset = file.lookup.get(name)
    if dataset is None:
        file.lookup.create_dataset(name, data=value)
    else:
        # Keep the existing dataset and reject casts that would change zone IDs.
        value = _cast_index(value, dataset)
        dataset[...] = value


def _metadata_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(_metadata_equal(left[key], right[key]) for key in left)
        )

    left, right = np.asarray(left), np.asarray(right)
    equal_nan = left.dtype.kind in "fc" and right.dtype.kind in "fc"
    return np.array_equal(left, right, equal_nan=equal_nan)


def _lookup_to_key_error(mapping, key):
    try:
        return mapping[key]
    except LookupError as e:
        # OMX raises the parent class LookupError instead of the more
        # appropriate KeyError.
        raise KeyError(key) from e


def _query_matrices(
    matrices: Mapping[str, Any],
    subset: Sequence[str | dict[str, Any]],
    matrix_metadata: Callable[[str], Mapping[str, Any]],
) -> dict[str, Any]:
    mats = {}
    for key in subset:
        if isinstance(key, str):
            _validate_matrix_key(key)
            mats[key] = _lookup_to_key_error(matrices, key)

        elif isinstance(key, dict):
            for name in matrices.keys():
                attrs = matrix_metadata(name)

                if all(attr in attrs and _metadata_equal(attrs[attr], value) for attr, value in key.items()):
                    _validate_matrix_key(name)
                    mats[name] = _lookup_to_key_error(matrices, name)
        else:
            raise TypeError("subset entries must be matrix names or attribute queries")

    return mats


_IndexKey: TypeAlias = str | tuple[str] | tuple[str, Any] | tuple[str, Any, Any]


def _validate_name(name: object, kind: str) -> None:
    if not isinstance(name, str):
        raise TypeError(f"{kind} name must be a string")
    if not name or name in (".", "..") or any(char in name for char in ("/", "\\", "\0")):
        raise ValueError(f"{kind} name must be a non-empty name, not a path")


def _validate_matrix_key(key: object) -> None:
    _validate_name(key, "matrix")


def _zone_ids(value: ArrayLike) -> np.ndarray:
    # Keep integer IDs exact and booleans visible when reading Python sequences.
    value = np.asarray(value, dtype=object if isinstance(value, (list, tuple)) else None)
    if value.ndim != 1:
        raise ValueError(f"zone IDs must have one dimension, got {value.ndim}")

    if value.size:
        if value.dtype.kind == "O":
            if any(isinstance(zone, (bool, np.bool_)) or not isinstance(zone, (int, np.integer)) for zone in value):
                raise TypeError("zone IDs must be integers, not booleans")
        elif value.dtype.kind not in "iu":
            raise TypeError("zone IDs must be integers, not booleans")

        if int(value.min()) < 0 or int(value.max()) > np.iinfo(np.uint64).max:
            raise ValueError("zone IDs must be between 0 and 2**64 - 1")

    return np.array(value, dtype=np.uint64, copy=True, order="C")


def _validate_matrix_dtype(dtype: np.dtype) -> None:
    if dtype.kind not in "biufc":
        raise TypeError(f"matrix values must have an integer, float, bool or complex dtype, got {dtype}")


def _matrix_values(value: ArrayLike, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(value, order="C")
    _validate_matrix_dtype(value.dtype)

    if value.shape != shape:
        raise ValueError(f"got bad shape, expected {shape}, got {value.shape}")
    return value


def _write_matrix(file: omx.File, name: str, value: np.ndarray, attributes: Mapping[str, Any]) -> None:
    dataset = file.data.get(name)
    if dataset is not None and dataset.dtype == value.dtype:
        # Keep the dataset, including its dtype and compression, when replacing values.
        dataset.attrs.update(attributes)
        dataset[...] = value
        return

    # Keep destination attributes and apply source attributes before replacing the dataset.
    attrs = dict(dataset.attrs) if dataset is not None else {}
    attrs.update(attributes)
    temporary_name = f"__aequilibrae_{uuid4().hex}"

    try:
        file.create_matrix(temporary_name, obj=value, attrs=attrs)
        if dataset is not None:
            del file.data[name]

        file.data.move(temporary_name, name)
    finally:
        if temporary_name in file.data:
            del file.data[temporary_name]


class _MatrixIndexer:
    def __init__(self, matrix: "_Matrix", *, zones: bool) -> None:
        self._matrix = matrix
        self._zones = zones

    def _translate_zones(self, selection: Any) -> Any:
        if selection is None or selection is Ellipsis:
            return selection

        positions = self._matrix._zone_positions
        if isinstance(selection, slice):
            start = None if selection.start is None else positions[selection.start]
            stop = None if selection.stop is None else positions[selection.stop]

            if stop is not None:
                if selection.step is not None and selection.step < 0:
                    # A negative stop would count from the end, not include the first zone.
                    stop = stop - 1 if stop else None
                else:
                    stop += 1
            return slice(start, stop, selection.step)

        labels = np.asarray(selection)
        if labels.dtype.kind == "b":  # boolean selection
            return selection

        if isinstance(selection, (list, tuple)):
            # Keep integer labels exact when a list mixes small and large zone IDs.
            labels = np.asarray(selection, dtype=object)

        if labels.ndim == 0:
            return positions[labels.item()]

        return np.fromiter((positions[zone] for zone in labels.flat), dtype=np.intp, count=labels.size).reshape(
            labels.shape
        )

    def _parse_key(self, key: _IndexKey) -> tuple[str, tuple[Any, ...]]:
        args = key if isinstance(key, tuple) else (key,)
        if not 1 <= len(args) <= 3:
            raise TypeError("expected a matrix name, a name and row, or a name, row and column")

        name = args[0]
        _validate_matrix_key(name)

        selection = args[1:]
        if self._zones:
            selection = tuple(self._translate_zones(value) for value in selection)

        return name, selection

    # Normal lookup
    @overload
    def __getitem__(self, key: str | tuple[str]) -> np.ndarray: ...

    # Dataset slice (partial read)
    @overload
    def __getitem__(self, key: tuple[str, Any] | tuple[str, Any, Any]) -> np.ndarray | np.generic: ...

    def __getitem__(self, key: _IndexKey) -> np.ndarray | np.generic:
        name, selection = self._parse_key(key)
        if not selection:
            return self._matrix[name]

        dataset = _lookup_to_key_error(self._matrix._backend, name)
        return dataset[selection]

    # Dataset slice (partial write), allows in-place modification without writing the whole matrix
    def __setitem__(self, key: _IndexKey, value: ArrayLike) -> None:
        name, selection = self._parse_key(key)
        if not selection:
            self._matrix[name] = value
        else:
            dataset = _lookup_to_key_error(self._matrix._backend, name)
            dataset[selection] = value


class _Matrix(MutableMapping[str, Any]):
    def __init__(self, index: ArrayLike, index_name: str) -> None:
        self._backend: MutableMapping[str, Any]
        self._metadata: MutableMapping[str, Any]
        self._index: np.ndarray
        assert hasattr(self, "_backend") and hasattr(self, "_metadata")

        _validate_name(index_name, "index")
        self.__index_name = index_name
        self._index = self._validate_index(index)  # Validate without writing a store's index
        self._zone_positions = {int(zone): position for position, zone in enumerate(self._index)}

    def save_as_omx(
        self,
        path: str | os.PathLike,
        mode: Literal["w", "a", "r+", "w-", "x"] = "x",
        *,
        subset: Sequence[str | dict[str, Any]] | None = None,
        **kwargs,
    ) -> None:
        if mode not in ("w", "a", "r+", "w-", "x"):
            raise ValueError("save_as_omx only supports modes 'w', 'a', 'r+', 'w-' and 'x'")

        if subset is not None:
            # Query backend arrays or dataset handles without loading store values.
            keys = set(_query_matrices(self._backend, subset, self.matrix_metadata))
        else:
            keys = set(self)

        if (shape := kwargs.get("shape")) is not None and tuple(shape) != self.shape:
            raise ValueError(f"destination matrix shape does not match, expected {self.shape}, got {shape}")

        # OMX will try to save the OMX_SPECIAL_ATTRS when opened with write permissions, so we first open with read only
        # to validate an existing file before updating it with 'a' or 'r+'.
        if mode in ("a", "r+") and os.path.exists(path):
            with omx.open_file(path, mode="r") as file:
                if (shape := file.shape()) is not None and shape != self.shape:
                    raise ValueError(f"destination matrix shape does not match, expected {self.shape}, got {shape}")

                dataset = file.lookup.get(self.index_name)
                if dataset is not None:
                    _cast_index(self.index, dataset)

        with omx.open_file(path, mode=mode, **kwargs) as file:
            # Save the index without replacing its existing dtype or attributes.
            _write_index(file, self.index_name, self.index)

            for name in keys:
                # Keep destination attributes and apply source attributes when writing the matrix.
                _write_matrix(file, name, self[name], self.matrix_metadata(name))

            for key, value in self.metadata.items():
                # These attributes are managed by OMX so we shouldn't try to write them
                if key in OMX_SPECIAL_ATTRS:
                    continue

                file.attrs[key] = value

    @abstractmethod
    def subset(
        self,
        subset: Sequence[str | dict[str, Any]],
    ) -> "_Matrix":
        pass

    def _set_index(self, value: ArrayLike) -> None:
        self._index = self._validate_index(value)

    def _validate_index(self, value: ArrayLike) -> np.ndarray:
        value = _zone_ids(value)

        # hasattr because we might not have set it yet as this is used in the __init__ function
        if hasattr(self, "_index") and len(self._index) != len(value):
            raise ValueError(f"matrix and index shape may not change, expected {len(self._index)}, got {len(value)}")

        if len(np.unique(value)) != len(value):
            raise ValueError("found duplicate values in the index, the index must be unique")

        return value

    @property
    def index(self) -> np.ndarray:
        view = self._index.view()
        view.setflags(write=False)
        return view

    @index.setter
    def index(self, value: ArrayLike) -> None:
        self._set_index(value)
        self._zone_positions = {int(zone): position for position, zone in enumerate(self._index)}

    @property
    def mat(self) -> _MatrixIndexer:
        """Select by zone ID, with the matrix name first. Label slices include the stop."""
        return _MatrixIndexer(self, zones=True)

    @property
    def imat(self) -> _MatrixIndexer:
        """Select by zero-based position, with the matrix name first."""
        return _MatrixIndexer(self, zones=False)

    @property
    def index_name(self) -> str:
        return self.__index_name

    @property
    def metadata(self) -> MutableMapping[str, Any]:
        return self._metadata

    @abstractmethod
    def matrix_metadata(self, name: str) -> MutableMapping[str, Any]:
        """Return mutable attributes for an existing named matrix."""
        pass

    @property
    def shape(self) -> tuple[int, int]:
        length = len(self._index)
        return (length, length)

    def __delitem__(self, key: str) -> None:
        _validate_matrix_key(key)
        del self._backend[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._backend.keys())

    def __len__(self) -> int:
        return len(self._backend)

    def __contains__(self, key: object) -> bool:
        _validate_matrix_key(key)
        return key in self._backend.keys()

    def __eq__(self, other: object) -> bool:
        """Compare matrices, their ordered zone index and metadata. Matching NaNs are equal."""
        if not isinstance(other, _Matrix):
            return NotImplemented
        if self is other:
            return True
        if (
            self.index_name != other.index_name
            or not np.array_equal(self.index, other.index)
            or self.keys() != other.keys()
        ):
            return False

        # Ignore file attributes managed by OMX, as loading and copying do.
        left_metadata = {key: value for key, value in self.metadata.items() if key not in OMX_SPECIAL_ATTRS}
        right_metadata = {key: value for key, value in other.metadata.items() if key not in OMX_SPECIAL_ATTRS}
        if not _metadata_equal(left_metadata, right_metadata):
            return False

        return all(
            _metadata_equal(self.matrix_metadata(name), other.matrix_metadata(name))
            and np.array_equal(self[name], other[name], equal_nan=True)
            for name in self
        )

    def clear(self) -> None:
        """Remove all named matrices without reading their values."""
        for key in list(self):
            del self[key]

    def setdefault(self, key: str, default: ArrayLike | None = None) -> np.ndarray:
        """Insert a missing matrix, then return it using the usual read rules."""
        if key not in self:
            self[key] = default
        return self[key]


class AequilibraeMatrix(_Matrix):
    def __init__(
        self,
        index: ArrayLike,
        index_name: str = "main_index",
    ) -> None:
        self._backend: dict[str, np.ndarray] = {}
        self._metadata: dict[str, Any] = {}
        self._matrix_metadata: dict[str, dict[str, Any]] = {}
        super().__init__(index=index, index_name=index_name)

    @classmethod
    def _from_shared_arrays(
        cls,
        data: Mapping[str, np.ndarray],
        index: ArrayLike,
        index_name: str = "main_index",
    ) -> Self:
        """Keep supplied arrays without copying, including strided views.

        This private path lets DenseMatrix hand off its buffer. Normal matrix
        assignment still copies values. Each array keeps its own buffer alive.
        """
        self = cls(index=index, index_name=index_name)
        for name, values in data.items():
            _validate_matrix_key(name)
            if not isinstance(values, np.ndarray):
                raise TypeError("shared matrix values must be NumPy arrays")
            _validate_matrix_dtype(values.dtype)
            if values.shape != self.shape:
                raise ValueError(f"got bad shape, expected {self.shape}, got {values.shape}")
            self._backend[name] = values
            self._matrix_metadata[name] = {}
        return self

    @classmethod
    def from_omx(
        cls,
        path: str | os.PathLike,
        index_name: str = "main_index",
        subset: Sequence[str | dict[str, Any]] | None = None,
        **kwargs,
    ) -> Self:
        _validate_name(index_name, "index")
        with omx.open_file(path, mode="r") as file:
            index = file.lookup.get(index_name)
            if index is None:
                raise ValueError(f"index not found, expected group at '/lookup/{index_name}'")

            if subset is None:
                subset = file.list_matrices()

            datasets = _query_matrices(file.data, subset, lambda name: file.data[name].attrs)

            self = cls(index=index, index_name=index_name, **kwargs)

            # We should ignore the OMX special attributes
            attrs = _copy_file_metadata(file.attrs)

            self.update({key: np.asarray(dataset) for key, dataset in datasets.items()})
            for key, dataset in datasets.items():
                self.matrix_metadata(key).update(deepcopy(dict(dataset.attrs)))
            self.metadata.update(attrs)
            return self

    def matrix_metadata(self, name: str) -> MutableMapping[str, Any]:
        _validate_matrix_key(name)
        return self._matrix_metadata[name]

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        del self._matrix_metadata[key]

    # Normal write
    def __setitem__(self, key: str, value: ArrayLike) -> None:
        _validate_matrix_key(key)
        value = _matrix_values(value, self.shape)
        self._backend[key] = np.asarray(value, copy=True, order="C")
        self._matrix_metadata.setdefault(key, {})

    # Normal lookup
    def __getitem__(self, key: str) -> np.ndarray:
        _validate_matrix_key(key)
        return self._backend[key].view()

    def subset(
        self,
        subset: Sequence[str | dict[str, Any]],
    ) -> Self:
        mats = _query_matrices(self, subset, self.matrix_metadata)

        other = self.__class__(index=self.index, index_name=self.index_name)
        other.update(mats)
        for key in mats:
            other.matrix_metadata(key).update(deepcopy(dict(self.matrix_metadata(key))))
        other.metadata.update(_copy_file_metadata(self.metadata))

        return other


class MatrixStore(_Matrix):
    def __init__(
        self,
        path: str | os.PathLike,
        index: ArrayLike | None = None,
        index_name: str = "main_index",
        mode: Literal["r", "w", "a", "r+", "w-", "x"] = "r",
        **kwargs,
    ) -> None:
        _validate_name(index_name, "index")
        if mode == "r" and index is not None:
            raise ValueError("cannot supply an index when opening a read-only matrix store")

        file = omx.open_file(
            path,
            mode=mode,
            **kwargs,
        )
        try:
            supplied_index = index is not None
            if not supplied_index:
                index = file.lookup.get(index_name)
                if index is None:
                    raise ValueError(
                        f"index not found, expected group at '/lookup/{index_name}'. "
                        "Either provide an index directly when opening a writable store, "
                        "or an index name of one that exists in the OMX file"
                    )

            self._backend: omx.File = file
            self._metadata: h5py.AttributeManager = file.attrs
            super().__init__(index=index, index_name=index_name)
            if (shape := file.shape()) is not None and shape != self.shape:
                raise ValueError(f"matrix shape does not match index, expected {self.shape}, got {shape}")

            for dataset in file.data.values():
                _validate_matrix_dtype(dataset.dtype)

            if supplied_index:
                self.index = index
        except:
            file.close()
            raise

    def matrix_metadata(self, name: str) -> h5py.AttributeManager:
        _validate_matrix_key(name)
        return self._backend.data[name].attrs

    @property
    def omx(self) -> omx.File:
        return self._backend

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: Any, **_kwargs: Any) -> None:
        self.close()

    def close(self) -> None:
        if hasattr(self, "_backend"):
            self.omx.close()

    def __del__(self) -> None:
        self.close()

    def _set_index(self, value: ArrayLike) -> None:
        if self._backend.mode == "r":
            raise PermissionError("cannot change the index of a read-only matrix store")

        value = self._validate_index(value)
        _write_index(self._backend, self.index_name, value)
        self._index = value

    def __setitem__(self, key: str, value: ArrayLike) -> None:
        _validate_matrix_key(key)
        value = _matrix_values(value, self.shape)

        # Keep the dataset's existing attributes, even when its dtype changes.
        _write_matrix(self._backend, key, value, {})

    # Normal lookup
    def __getitem__(self, key: str) -> np.ndarray:
        _validate_matrix_key(key)
        value = _lookup_to_key_error(self._backend, key)
        return np.asarray(value)

    def subset(
        self,
        subset: Sequence[str | dict[str, Any]],
    ) -> AequilibraeMatrix:
        datasets = _query_matrices(self._backend.data, subset, self.matrix_metadata)

        other = AequilibraeMatrix(index=self.index, index_name=self.index_name)
        other.update({key: np.asarray(dataset) for key, dataset in datasets.items()})

        for key, dataset in datasets.items():
            other.matrix_metadata(key).update(deepcopy(dict(dataset.attrs)))
        other.metadata.update(_copy_file_metadata(self.metadata))

        return other


@overload
def from_file(
    path: str | os.PathLike,
    mode: None = None,
    *,
    index_name: str | None = None,
    subset: Sequence[str | dict[str, Any]] | None = None,
    store: Literal[False] = False,
) -> AequilibraeMatrix: ...


@overload
def from_file(
    path: str | os.PathLike,
    mode: Literal["r", "r+"] | None = None,
    *,
    index_name: str | None = None,
    subset: None = None,
    store: Literal[True],
) -> MatrixStore: ...


def from_file(
    path: str | os.PathLike,
    mode: Literal["r", "r+"] | None = None,
    *,
    index_name: str | None = None,
    subset: Sequence[str | dict[str, Any]] | None = None,
    store: bool = False,
) -> AequilibraeMatrix | MatrixStore:
    """
    Load an OMX file as an in-memory matrix or an on-disk store.

    If ``index_name`` is omitted, and there is only one zone index in the file,
    it is selected automatically. If the file contains multiple indices,
    ``main_index`` is selected when present, otherwise, pass ``index_name``
    explicitly.

    Supply ``subset`` to load selected matrices into an in-memory
    :class:`AequilibraeMatrix`. Supply ``store=True`` instead to open a
    :class:`MatrixStore`; stores do not accept ``subset``.
    """
    if store:
        if subset is not None:
            raise ValueError("store=True and subset may not be provided at once")
        if mode is not None and mode not in ("r", "r+"):
            raise ValueError("from_file stores only support modes 'r' and 'r+' (file must exist)")
    elif mode is not None:
        raise ValueError("an in-memory matrix cannot have a mode")

    if index_name is None:
        with omx.open_file(path, mode="r") as file:
            index_names = list(file.lookup.keys())

        if len(index_names) == 1:
            index_name = index_names[0]
        elif "main_index" in index_names:
            index_name = "main_index"
        elif not index_names:
            raise ValueError(f"No zone index found in OMX matrix file {str(path)!r}")
        else:
            raise ValueError(f"Multiple zone indices found in OMX matrix file {str(path)!r}; please specify index_name")

    if store:
        return MatrixStore(path=path, index_name=index_name, mode=mode if mode is not None else "r")
    return AequilibraeMatrix.from_omx(path=path, index_name=index_name, subset=subset)


@overload
def from_dict(
    data: Mapping[str, ArrayLike],
    index: ArrayLike,
    *,
    path: None = None,
    mode: None = None,
    index_name: str = "main_index",
    metadata: Mapping[str, Any] | None = None,
    store: Literal[False] = False,
) -> AequilibraeMatrix: ...


@overload
def from_dict(
    data: Mapping[str, ArrayLike],
    index: ArrayLike,
    *,
    path: str | os.PathLike,
    mode: Literal["w", "a", "r+", "w-", "x"] | None = None,
    index_name: str = "main_index",
    metadata: Mapping[str, Any] | None = None,
    store: Literal[True],
) -> MatrixStore: ...


def from_dict(
    data: Mapping[str, ArrayLike],
    index: ArrayLike,
    *,
    path: str | os.PathLike | None = None,
    mode: Literal["w", "a", "r+", "w-", "x"] | None = None,
    index_name: str = "main_index",
    metadata: Mapping[str, Any] | None = None,
    store: bool = False,
) -> AequilibraeMatrix | MatrixStore:
    mat: AequilibraeMatrix | MatrixStore
    if store:
        if path is None:
            raise ValueError("a matrix store must have a path")

        if mode is not None and mode not in ("w", "a", "r+", "w-", "x"):
            raise ValueError("from_dict stores only support modes 'w', 'a', 'r+', 'w-' and 'x'")

        # Check names, shapes and dtypes before opening a file or changing its index.
        validated_index = AequilibraeMatrix(index=index, index_name=index_name).index
        prepared = {}
        for name, value in data.items():
            _validate_matrix_key(name)
            prepared[name] = _matrix_values(value, (len(validated_index), len(validated_index)))

        data = prepared

        mat = MatrixStore(
            path=path,
            index=validated_index,
            index_name=index_name,
            mode=mode if mode is not None else "x",
        )
    else:
        if path is not None:
            raise ValueError("an in-memory matrix cannot have a path")

        if mode is not None:
            raise ValueError("an in-memory matrix cannot have a mode")
        mat = AequilibraeMatrix(index=index, index_name=index_name)

    try:
        mat.update(data)
        if metadata is not None:
            mat.metadata.update(_copy_file_metadata(metadata))

    except Exception:
        if isinstance(mat, MatrixStore):
            mat.close()
        raise

    return mat


def from_df(
    df: pd.DataFrame,
    row: str,
    col: str,
    subset: Sequence[str] | None = None,
    *,
    index: ArrayLike | None = None,
    index_name: str = "main_index",
    fill_value: float = 0,
    dtype: DTypeLike = float,
) -> AequilibraeMatrix:
    dtype = np.dtype(dtype)
    if subset is None:
        subset = [name for name in df.columns if name not in (row, col)]

    row_ids = _zone_ids(df[row].to_numpy())
    col_ids = _zone_ids(df[col].to_numpy())
    if index is None:
        index = np.union1d(row_ids, col_ids)

    mat = AequilibraeMatrix(index=index, index_name=index_name)
    matrix_index = pd.Index(mat.index)
    rows = matrix_index.get_indexer(row_ids)
    cols = matrix_index.get_indexer(col_ids)
    if np.any(rows == -1) or np.any(cols == -1):
        raise ValueError("found row or column values that are not in the index")

    for name in subset:
        values = df[name].to_numpy(dtype=dtype, copy=False)
        values_matrix: np.ndarray = np.full(mat.shape, fill_value, dtype=dtype)
        values_matrix[rows, cols] = 0
        np.add.at(values_matrix, (rows, cols), values)
        mat[name] = values_matrix

    return mat


def from_scipy(
    data: Mapping[str, scipy.sparse.sparray],
    index: ArrayLike,
    *,
    index_name: str = "main_index",
) -> AequilibraeMatrix:
    for value in data.values():
        if not isinstance(value, scipy.sparse.sparray):
            raise TypeError("data values must be SciPy sparse arrays")
        if value.ndim != 2 or value.shape[0] != value.shape[1]:  # type: ignore
            raise ValueError("sparse arrays must be square")

    mat = AequilibraeMatrix(index=index, index_name=index_name)
    mat.update({name: value.toarray() for name, value in data.items()})  # type: ignore

    return mat
