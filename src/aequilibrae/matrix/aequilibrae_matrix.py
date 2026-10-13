import os
from abc import abstractmethod
from collections.abc import Iterator, MutableMapping
from copy import deepcopy
from typing import Any, Literal, Mapping, Self, Sequence, TypeAlias, overload

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
        if not np.array_equal(dataset[:], value):
            dataset[...] = value


def _query_omx(file: omx.File, subset: Sequence[str | dict[str, Any]]) -> list[tuple[str, h5py.Dataset]]:
    mats = []
    for key in subset:
        if isinstance(key, str):
            mats.append((key, file[key]))
        else:
            for dataset in file[key]:
                name = dataset.name.rsplit("/", 1)[-1]
                mats.append((name, dataset))

    return mats


_IndexKey: TypeAlias = str | tuple[str] | tuple[Any, str] | tuple[Any, Any, str]


def _validate_matrix_key(key: str) -> None:
    if not isinstance(key, str):
        raise TypeError("matrix name must be a string")


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
            raise TypeError("expected a matrix name, a row and name, or a row, column and name")

        name = args[-1]
        _validate_matrix_key(name)

        selection = args[:-1]
        if self._zones:
            selection = tuple(self._translate_zones(value) for value in selection)

        return name, selection

    # Normal lookup
    @overload
    def __getitem__(self, key: str | tuple[str]) -> np.ndarray: ...

    # Dataset slice (partial read)
    @overload
    def __getitem__(self, key: tuple[Any, str] | tuple[Any, Any, str]) -> np.ndarray | np.generic: ...

    def __getitem__(self, key: _IndexKey) -> np.ndarray | np.generic:
        name, selection = self._parse_key(key)
        if not selection:
            return self._matrix[name]

        try:
            dataset = self._matrix._backend[name]
        except LookupError as exc:
            raise KeyError(name) from exc

        return dataset[selection]

    # Dataset slice (partial write), allows in-place modification without writing the whole matrix
    def __setitem__(self, key: _IndexKey, value: ArrayLike) -> None:
        name, selection = self._parse_key(key)
        if not selection:
            self._matrix[name] = value
        else:
            try:
                dataset = self._matrix._backend[name]
            except LookupError as exc:
                raise KeyError(name) from exc

            dataset[selection] = value


class _Matrix(MutableMapping[str, Any]):
    def __init__(self, index: ArrayLike, index_name: str) -> None:
        self._backend: MutableMapping[str, Any]
        self._metadata: MutableMapping[str, Any]
        self._index: np.ndarray
        assert hasattr(self, "_backend") and hasattr(self, "_metadata")

        self.__index_name = index_name
        self._index = self._validate_index(index)  # Validate without writing a store's index
        self._zone_positions = {int(zone): position for position, zone in enumerate(self._index)}

    @classmethod
    def from_omx(
        cls,
        path: str | os.PathLike,
        index_name: str = "main_index",
        subset: Sequence[str | dict[str, Any]] | None = None,
        **kwargs,
    ) -> Self:
        with omx.open_file(path, mode="r") as file:
            index = file.lookup.get(index_name)
            if index is None:
                raise ValueError(f"index not found, expected group at '/lookup/{index_name}'")

            if subset is None:
                subset = file.list_matrices()

            datasets = dict(_query_omx(file, subset))

            self = cls(index=index, index_name=index_name, **kwargs)

            # We should ignore the OMX special attributes
            attrs = _copy_file_metadata(file.attrs)

            self.update({key: np.asarray(dataset) for key, dataset in datasets.items()})
            for key, dataset in datasets.items():
                self.matrix_metadata(key).update(deepcopy(dict(dataset.attrs)))
            self.metadata.update(attrs)
            return self

    def save_as_omx(
        self,
        path: str | os.PathLike,
        subset: list[str] | None = None,
        overwrite: bool = False,
        **kwargs,
    ) -> None:

        keys = set(self) if subset is None else set(subset)
        if missing_keys := keys - self.keys():
            raise ValueError(f"found non-existent keys in subset: {missing_keys}")

        if (shape := kwargs.get("shape")) is not None and tuple(shape) != self.shape:
            raise ValueError(f"destination matrix shape does not match, expected {self.shape}, got {shape}")

        # OMX will try to save the OMX_SPECIAL_ATTRS when opened with write permissions, so we first open with read only
        # to validate the file, then re-open with write permissions only once we know it's ok to save.
        if os.path.exists(path):
            with omx.open_file(path, mode="r") as file:
                if (shape := file.shape()) is not None and shape != self.shape:
                    raise ValueError(f"destination matrix shape does not match, expected {self.shape}, got {shape}")

                dataset = file.lookup.get(self.index_name)
                if dataset is not None:
                    index = _cast_index(self.index, dataset)
                    if not overwrite and not np.array_equal(dataset[:], index):
                        raise ValueError(
                            f"index '{self.index_name}' already exists with different values in file: {path}"
                        )

                if not overwrite:
                    if existing_keys := keys & set(file.list_matrices()):
                        raise ValueError(f"matrices {existing_keys} already exist in file: {path}")

                    # Ignoring OMX special attributes, we don't write them anyway
                    if existing_keys := file.attrs.keys() & (self.metadata.keys() - OMX_SPECIAL_ATTRS):
                        raise ValueError(f"attributes {existing_keys} already exist in file: {path}")

        with omx.open_file(path, mode="a", **kwargs) as file:
            # Save the index without replacing its existing dtype or attributes.
            _write_index(file, self.index_name, self.index)

            for name in keys:
                # OMX replaces the dataset, so keep destination attributes and apply source attributes.
                dataset = file.data.get(name)
                attributes = dict(dataset.attrs) if dataset is not None else {}
                attributes.update(self.matrix_metadata(name))

                file[name] = self[name]
                file[name].attrs.update(attributes)

            for key, value in self.metadata.items():
                # These attributes are managed by OMX so we shouldn't try to write them
                if key in OMX_SPECIAL_ATTRS:
                    continue

                file.attrs[key] = value

    @abstractmethod
    def subset(
        self,
        subset: Sequence[str],
    ) -> "_Matrix":
        pass

    def _set_index(self, value: ArrayLike) -> None:
        self._index = self._validate_index(value)

    def _validate_index(self, value: ArrayLike) -> np.ndarray:
        value = np.asarray(value)
        if value.ndim != 1:
            raise ValueError(f"got too many dimensions for the index, expected 1, got {value.ndim}")

        # hasattr because we might not have set it yet as this is used in the __init__ function
        if hasattr(self, "_index") and len(self._index) != len(value):
            raise ValueError(f"matrix and index shape may not change, expected {len(self._index)}, got {len(value)}")

        if len(np.unique(value)) != len(value):
            raise ValueError("found duplicate values in the index, the index must be unique")

        if value.size and not np.can_cast(value.dtype, np.uintp, casting="safe"):
            if np.isdtype(value.dtype, "signed integer"):
                if value.min() < 0:
                    raise TypeError(f"cannot cast {value.dtype} to {np.uintp} safely (found negative values)")
            else:
                raise TypeError(f"cannot cast {value.dtype} to {np.uintp} safely")

        return np.asarray(value, copy=True, dtype=np.uintp, order="C")

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
    def loc(self) -> _MatrixIndexer:
        """Select by zone ID, with the matrix name last. Label slices include the stop."""
        return _MatrixIndexer(self, zones=True)

    @property
    def iloc(self) -> _MatrixIndexer:
        """Select by zero-based position, with the matrix name last."""
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
        del self._backend[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._backend.keys())

    def __len__(self) -> int:
        return len(self._backend)

    def __contains__(self, key: object) -> bool:
        return key in self._backend.keys()


class AequilibraEMatrix(_Matrix):
    def __init__(
        self,
        index: ArrayLike,
        index_name: str = "main_index",
    ) -> None:
        self._backend: dict[str, np.ndarray] = {}
        self._metadata: dict[str, Any] = {}
        self._matrix_metadata: dict[str, dict[str, Any]] = {}
        super().__init__(index=index, index_name=index_name)

    def matrix_metadata(self, name: str) -> MutableMapping[str, Any]:
        return self._matrix_metadata[name]

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        del self._matrix_metadata[key]

    # Normal write
    def __setitem__(self, key: str, value: ArrayLike) -> None:
        _validate_matrix_key(key)
        value = np.asarray(value, order="C")
        shape = self.shape

        if value.shape != shape:
            raise ValueError(f"got bad shape, expected {shape}, got {value.shape}")

        self._backend[key] = np.asarray(value, copy=True, order="C")
        self._matrix_metadata.setdefault(key, {})

    # Normal lookup
    def __getitem__(self, key: str) -> np.ndarray:
        _validate_matrix_key(key)
        return self._backend[key].view()

    def subset(
        self,
        subset: Sequence[str],
    ) -> Self:
        mats = {key: self[key] for key in subset}

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

            if supplied_index:
                self.index = index
        except:
            file.close()
            raise

    def matrix_metadata(self, name: str) -> h5py.AttributeManager:
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
        value = np.asarray(value, order="C")

        shape = self.shape
        if value.shape != shape:
            raise ValueError(f"got bad shape, expected {shape}, got {value.shape}")

        # OMX replaces the dataset, so keep its existing attributes.
        attributes = dict(self._backend[key].attrs) if key in self else {}
        self._backend[key] = value
        self._backend[key].attrs.update(attributes)

    # Normal lookup
    def __getitem__(self, key: str) -> np.ndarray:
        _validate_matrix_key(key)
        try:
            value = self._backend[key]
        except LookupError as e:
            raise KeyError(key) from e

        return np.asarray(value)

    def subset(
        self,
        subset: Sequence[str | dict[str, Any]],
    ) -> AequilibraEMatrix:
        datasets = dict(_query_omx(self._backend, subset))

        other = AequilibraEMatrix(index=self.index, index_name=self.index_name)
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
    index_name: str = "main_index",
    subset: Sequence[str | dict[str, Any]] | None = None,
    store: Literal[False] = False,
) -> AequilibraEMatrix: ...


@overload
def from_file(
    path: str | os.PathLike,
    mode: Literal["r", "r+"] | None = None,
    *,
    index_name: str = "main_index",
    subset: None = None,
    store: Literal[True],
) -> MatrixStore: ...


def from_file(
    path: str | os.PathLike,
    mode: Literal["r", "r+"] | None = None,
    *,
    index_name: str = "main_index",
    subset: Sequence[str | dict[str, Any]] | None = None,
    store: bool = False,
) -> AequilibraEMatrix | MatrixStore:
    if store:
        if subset is not None:
            raise ValueError("store=True and subset may not be provided at once")

        if mode is not None and mode not in ("r", "r+"):
            raise ValueError("from_file stores only support modes 'r' and 'r+' (file must exist)")

        return MatrixStore(path=path, index_name=index_name, mode=mode if mode is not None else "r")
    else:
        if mode is not None:
            raise ValueError("an in-memory matrix cannot have a mode")
        return AequilibraEMatrix.from_omx(path=path, index_name=index_name, subset=subset)


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
) -> AequilibraEMatrix: ...


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
) -> AequilibraEMatrix | MatrixStore:
    mat: AequilibraEMatrix | MatrixStore
    if store:
        if path is None:
            raise ValueError("a matrix store must have a path")

        if mode is not None and mode not in ("w", "a", "r+", "w-", "x"):
            raise ValueError("from_dict stores only support modes 'w', 'a', 'r+', 'w-' and 'x'")

        mat = MatrixStore(
            path=path,
            index=index,
            index_name=index_name,
            mode=mode if mode is not None else "x",
        )
    else:
        if path is not None:
            raise ValueError("an in-memory matrix cannot have a path")

        if mode is not None:
            raise ValueError("an in-memory matrix cannot have a mode")
        mat = AequilibraEMatrix(index=index, index_name=index_name)

    mat.update(data)
    if metadata is not None:
        mat.metadata.update(_copy_file_metadata(metadata))

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
) -> AequilibraEMatrix:
    dtype = np.dtype(dtype)
    if subset is None:
        subset = [name for name in df.columns if name not in (row, col)]

    if index is None:
        index = np.union1d(df[row].to_numpy(), df[col].to_numpy())

    mat = AequilibraEMatrix(index=index, index_name=index_name)
    matrix_index = pd.Index(mat.index)
    rows = matrix_index.get_indexer(df[row])
    cols = matrix_index.get_indexer(df[col])
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
) -> AequilibraEMatrix:
    for value in data.values():
        if not isinstance(value, scipy.sparse.sparray):
            raise TypeError("data values must be SciPy sparse arrays")
        if value.ndim != 2 or value.shape[0] != value.shape[1]:  # type: ignore
            raise ValueError("sparse arrays must be square")

    mat = AequilibraEMatrix(index=index, index_name=index_name)
    mat.update({name: value.toarray() for name, value in data.items()})  # type: ignore

    return mat
