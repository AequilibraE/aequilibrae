import os
from abc import abstractmethod
from collections.abc import Iterator, MutableMapping
from typing import Any, Literal, Mapping, Self, Sequence, overload

import h5py
import numpy as np
import openmatrix as omx
import pandas as pd
import scipy.sparse
from numpy.typing import ArrayLike

OMX_SPECIAL_ATTRS = {"OMX_VERSION", "OMX_CREATED_WITH", "SHAPE", "TITLE"}


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


@overload
def _key_to_key_and_rest(key: str | tuple[str, *tuple[Any, ...]]) -> tuple[str, tuple[Any, ...]]: ...


@overload
def _key_to_key_and_rest(
    key: dict[str, Any] | tuple[dict[str, Any], *tuple[Any, ...]],
) -> tuple[dict[str, Any], tuple[Any, ...]]: ...


def _key_to_key_and_rest(
    key: str | dict[str, Any] | tuple[str | dict[str, Any], *tuple[Any, ...]],
) -> tuple[str | dict[str, Any], tuple[Any, ...]]:
    if isinstance(key, tuple):
        return key[0], key[1:]
    else:
        return key, ()


class _Matrix(MutableMapping[str, Any]):
    def __init__(self, index: ArrayLike, index_name: str) -> None:
        self._backend: MutableMapping[str, Any]
        self._metadata: MutableMapping[str, Any]
        self._index: np.ndarray
        assert hasattr(self, "_backend") and hasattr(self, "_metadata")

        self.__index_name = index_name
        self.index = index  # Assign using property setter for validation

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

            mats = {key: np.asarray(dataset) for key, dataset in _query_omx(file, subset)}

            self = cls(index=index, index_name=index_name, **kwargs)

            # We should ignore the OMX special attributes
            attrs = {key: value for key, value in file.attrs.items() if key not in OMX_SPECIAL_ATTRS}

            self.update(mats)
            self.metadata.update(attrs)
            return self

    def save_as_omx(
        self,
        path: str | os.PathLike,
        subset: list[str] | None = None,
        matrix_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        overwrite: bool = False,
        **kwargs,
    ) -> None:

        keys = set(self) if subset is None else set(subset)
        if missing_keys := keys - self.keys():
            raise ValueError(f"found non-existent keys in subset: {missing_keys}")
        if matrix_metadata is not None and (missing_keys := matrix_metadata.keys() - keys):
            raise ValueError(f"matrix_metadata contains names outside the save subset: {missing_keys}")

        if (shape := kwargs.get("shape")) is not None and tuple(shape) != self.shape:
            raise ValueError(f"destination matrix shape does not match, expected {self.shape}, got {shape}")

        # OMX will try to save the OMX_SPECIAL_ATTRS when opened with write permissions, so we first open with read only
        # to validate the file, then re-open with write permissions only once we know it's ok to save.
        if os.path.exists(path):
            with omx.open_file(path, mode="r") as file:
                if (shape := file.shape()) is not None and shape != self.shape:
                    raise ValueError(f"destination matrix shape does not match, expected {self.shape}, got {shape}")

                if not overwrite:
                    if self.index_name in file.list_mappings():
                        raise ValueError(f"index '{self.index_name}' already exists in file: {path}")

                    if existing_keys := keys & set(file.list_matrices()):
                        raise ValueError(f"matrices {existing_keys} already exist in file: {path}")

                    # Ignoring OMX special attributes, we don't write them anyway
                    if existing_keys := file.attrs.keys() & (self.metadata.keys() - OMX_SPECIAL_ATTRS):
                        raise ValueError(f"attributes {existing_keys} already exist in file: {path}")

        with omx.open_file(path, mode="a", **kwargs) as file:
            # Save index, overwrite=True because we've already checked
            file.create_mapping(title=self.index_name, entries=self.index, overwrite=True)

            for name in keys:
                # OMX replaces the dataset, so keep its existing attributes.
                dataset = file.data.get(name)
                attributes = dict(dataset.attrs) if dataset is not None else {}

                file[name] = self[name]
                file[name].attrs.update(attributes)

            if matrix_metadata is not None:
                for name, attributes in matrix_metadata.items():
                    attrs = file[name].attrs
                    for key, value in attributes.items():
                        attrs[key] = value

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

        self._index = np.asarray(value, copy=True, dtype=np.uintp, order="C")

    @property
    def index(self) -> np.ndarray:
        view = self._index.view()
        view.setflags(write=False)
        return view

    @index.setter
    def index(self, value: ArrayLike) -> None:
        self._set_index(value)

    @property
    def index_name(self) -> str:
        return self.__index_name

    @property
    def metadata(self) -> MutableMapping[str, Any]:
        return self._metadata

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
        super().__init__(index=index, index_name=index_name)

    # Normal write
    @overload
    def __setitem__(self, key: str | tuple[str], value: ArrayLike) -> None: ...

    # Allows in-place modification without writing the whole matrix
    @overload
    def __setitem__(self, key: tuple[str, Any, *tuple[Any, ...]], value: ArrayLike) -> None: ...

    def __setitem__(self, key: str | tuple[str, *tuple[Any, ...]], value: ArrayLike) -> None:
        key, rest = _key_to_key_and_rest(key)

        if not rest:
            value = np.asarray(value, dtype="float64", order="C")
            shape = self.shape

            if value.shape != shape:
                raise ValueError(f"got bad shape, expected {shape}, got {value.shape}")

            self._backend[key] = np.asarray(value, copy=True, dtype="float64", order="C")
        else:
            self._backend[key][rest] = value

    # Normal lookup
    @overload
    def __getitem__(self, key: str | tuple[str]) -> np.ndarray: ...

    # dataset slice (partial read)
    @overload
    def __getitem__(self, key: tuple[str, Any, *tuple[Any, ...]]) -> np.ndarray | np.generic: ...

    def __getitem__(self, key: str | tuple[str, *tuple[Any, ...]]) -> np.ndarray | np.generic:
        key, rest = _key_to_key_and_rest(key)

        if rest:
            return self._backend[key][rest]
        else:
            return self._backend[key].view()

    def subset(
        self,
        subset: Sequence[str],
    ) -> Self:
        mats = {key: self[key] for key in subset}

        other = self.__class__(index=self.index, index_name=self.index_name)
        other.update(mats)
        other.metadata.update(self.metadata)

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
        file = omx.open_file(
            path,
            mode=mode,
            **kwargs,
        )
        try:
            if index is None:
                index = file.lookup.get(index_name)
                if index is None:
                    raise ValueError(
                        f"index not found, expected group at '/lookup/{index_name}'. "
                        "Either provide an index directly, or an index name of one that exists with the OMX file"
                    )

            self._backend: omx.File = file
            self._metadata: h5py.AttributeManager = file.attrs
            super().__init__(index=index, index_name=index_name)
        except:
            file.close()
            raise

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
        if self._backend.mode == "r" and hasattr(self, "_index"):
            raise PermissionError("cannot change the index of a read-only matrix store")

        super()._set_index(value)
        if self._backend.mode != "r":
            self._backend.create_mapping(self.index_name, self.index, overwrite=True)

    @overload
    def __setitem__(self, key: str | tuple[str], value: ArrayLike) -> None: ...

    # dataset slice (partial write), allows in-place modification without writing the whole matrix
    @overload
    def __setitem__(self, key: tuple[str, Any, *tuple[Any, ...]], value: ArrayLike) -> None: ...

    def __setitem__(self, key: str | tuple[str, *tuple[Any, ...]], value: ArrayLike) -> None:
        key, rest = _key_to_key_and_rest(key)

        if not rest:
            value = np.asarray(value, dtype="float64", order="C")

            shape = self.shape
            if value.shape != shape:
                raise ValueError(f"got bad shape, expected {shape}, got {value.shape}")

            # OMX replaces the dataset, so keep its existing attributes.
            attributes = dict(self._backend[key].attrs) if key in self else {}
            self._backend[key] = value
            self._backend[key].attrs.update(attributes)
        else:
            self._backend[key][rest] = value

    # Normal lookup
    @overload
    def __getitem__(self, key: str | tuple[str]) -> np.ndarray: ...

    # OMX attribute query, optionally with a dataset slice (partial read)
    @overload
    def __getitem__(self, key: dict[str, Any] | tuple[dict[str, Any], *tuple[Any, ...]]) -> list[np.ndarray]: ...

    # Named dataset slice (partial read)
    @overload
    def __getitem__(self, key: tuple[str, Any, *tuple[Any, ...]]) -> np.ndarray | np.generic: ...

    def __getitem__(
        self,
        key: str | dict[str, Any] | tuple[str, *tuple[Any, ...]] | tuple[dict[str, Any], *tuple[Any, ...]],
    ) -> np.ndarray | np.generic | list[np.ndarray]:
        key, rest = _key_to_key_and_rest(key)

        try:
            value = self._backend[key]
        except LookupError as e:
            raise KeyError(key) from e

        if isinstance(key, str):
            return value[rest] if rest else np.asarray(value)
        else:
            return [np.asarray(x[rest] if rest else x) for x in value]

    def subset(
        self,
        subset: Sequence[str | dict[str, Any]],
    ) -> AequilibraEMatrix:
        mats = {key: np.asarray(dataset) for key, dataset in _query_omx(self._backend, subset)}

        other = AequilibraEMatrix(index=self.index, index_name=self.index_name)
        other.update(mats)
        other.metadata.update(self.metadata)

        return other


@overload
def from_file(
    path: str | os.PathLike,
    mode: Literal["r", "w", "a", "r+", "w-", "x"] = "r",
    *,
    index_name: str = "main_index",
    subset: Sequence[str | dict[str, Any]] | None = None,
    store: Literal[False] = False,
) -> AequilibraEMatrix: ...


@overload
def from_file(
    path: str | os.PathLike,
    mode: Literal["r", "w", "a", "r+", "w-", "x"] = "r",
    *,
    index_name: str = "main_index",
    subset: None = None,
    store: Literal[True],
) -> MatrixStore: ...


@overload
def from_file(
    path: str | os.PathLike,
    mode: Literal["r", "w", "a", "r+", "w-", "x"] = "r",
    *,
    index_name: str = "main_index",
    subset: Sequence[str | dict[str, Any]] | None = None,
    store: bool,
) -> AequilibraEMatrix | MatrixStore: ...


def from_file(
    path: str | os.PathLike,
    mode: Literal["r", "w", "a", "r+", "w-", "x"] = "r",
    *,
    index_name: str = "main_index",
    subset: Sequence[str | dict[str, Any]] | None = None,
    store: bool = False,
) -> AequilibraEMatrix | MatrixStore:
    if store:
        if subset is not None:
            raise ValueError("store=True and subset may not be provided at once")
        return MatrixStore(path=path, index_name=index_name, mode=mode)
    else:
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


@overload
def from_dict(
    data: Mapping[str, ArrayLike],
    index: ArrayLike,
    *,
    path: str | os.PathLike | None = None,
    mode: Literal["w", "a", "r+", "w-", "x"] | None = None,
    index_name: str = "main_index",
    metadata: Mapping[str, Any] | None = None,
    store: bool,
) -> AequilibraEMatrix | MatrixStore: ...


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
        mat = MatrixStore(
            path=path,
            index=index,
            index_name=index_name,
            mode=mode if mode is not None else "a",
        )
    else:
        if path is not None:
            raise ValueError("an in-memory matrix cannot have a path")
        mat = AequilibraEMatrix(index=index, index_name=index_name)

    mat.update(data)
    if metadata is not None:
        mat.metadata.update(metadata)

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
) -> AequilibraEMatrix:
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
        values = df[name].to_numpy(dtype="float64", copy=False)
        values_matrix: np.ndarray = np.full(mat.shape, fill_value, dtype="float64")
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
