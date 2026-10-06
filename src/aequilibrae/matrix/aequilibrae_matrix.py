import os
from abc import abstractmethod
from collections.abc import MutableMapping
from typing import Any, Literal, Mapping, Self, Sequence, overload

import h5py
import numpy as np
import openmatrix as omx
import pandas as pd
import scipy.sparse


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


class Matrix(MutableMapping):
    def __init__(self, index: np.ndarray, index_name: str):
        self._backend: MutableMapping[str, np.ndarray]
        self._metadata: MutableMapping[str, Any]
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

            self.update(mats)
            self.metadata.update(dict(file.attrs))
            return self

    def save_as_omx(
        self,
        path: str | os.PathLike,
        subset: list[str] | None = None,
        matrix_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs,
    ) -> None:
        with omx.open_file(path, mode="a", **kwargs) as file:
            for name in self.keys() if subset is None else subset:
                file[name] = self[name]

            if matrix_metadata is not None:
                for name, attributes in matrix_metadata.items():
                    attrs = file[name].attrs
                    for key, value in attributes.items():
                        attrs[key] = value

            for key, value in self.metadata.items():
                file.attrs[key] = value

    @abstractmethod
    def subset(
        self,
        subset: Sequence[str],
    ) -> Self:
        pass

    def _get_index(self) -> np.ndarray:
        view = self.__index.view()
        view.setflags(write=False)
        return view

    def _set_index(self, value: np.ndarray) -> None:
        if value.ndim != 1:
            raise ValueError(f"got too many dimensions for the index, expected 1, got {value.ndim}")

        if len(np.unique(value)) != len(value):
            raise ValueError("found duplicate values in the index, the index must be unique")

        self.__index = np.asarray(value, copy=True, dtype=np.uintp, order="C")

    index = property(_get_index, _set_index)

    @property
    def index_name(self) -> str:
        return self.__index_name

    @property
    def metadata(self) -> MutableMapping[str, Any]:
        return self._metadata

    @property
    def shape(self) -> tuple[int, int]:
        length = len(self.__index)
        return (length, length)

    def __delitem__(self, key: str) -> None:
        del self._backend[key]

    def __iter__(self):
        return iter(self._backend.keys())

    def __len__(self) -> int:
        return len(self._backend)


class AequilibraEMatrix(Matrix):
    def __init__(
        self,
        index: np.ndarray,
        index_name: str = "main_index",
    ):
        self._backend: dict[str, np.ndarray] = {}
        self._metadata: dict[str, Any] = {}
        super().__init__(index=index, index_name=index_name)

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        shape = self.shape

        if value.shape != shape:
            raise ValueError(f"got bad shape, expected {shape}, got {value.shape}")

        self._backend[key] = np.asarray(value, copy=True, dtype="float64", order="C")

    def __getitem__(self, key: str) -> np.ndarray:
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


class MatrixStore(Matrix):
    def __init__(
        self,
        path: str | os.PathLike,
        index: np.ndarray | None = None,
        index_name: str = "main_index",
        mode: Literal["r", "w", "a", "r+", "w-", "x"] = "r",
        **kwargs,
    ):
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
            super().__init__(index=np.asarray(index), index_name=index_name)
        except:
            file.close()
            raise

    @property
    def omx(self) -> omx.File:
        return self._backend

    def __enter__(self):
        return self

    def __exit__(self, *_args, **_kwargs):
        self.close()

    def close(self):
        self.omx.close()

    def __del__(self):
        self.close()

    def _set_index(self, value: np.ndarray) -> None:
        super()._set_index(value)
        if self._backend.mode != "r":
            self._backend.create_mapping(self.index_name, self.index, overwrite=True)

    index = property(Matrix._get_index, _set_index)

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        shape = self.shape

        if value.shape != shape:
            raise ValueError(f"got bad shape, expected {shape}, got {value.shape}")

        self._backend[key] = value

    @overload
    def __getitem__(self, key: str) -> np.ndarray: ...

    @overload
    def __getitem__(self, key: dict[str, Any]) -> list[np.ndarray]: ...

    def __getitem__(self, key: str | dict) -> np.ndarray | list[np.ndarray]:
        value = self._backend[key]

        if isinstance(key, str):
            # OMX escape hatch to access HDF5 objects
            if key.startswith("/"):
                return value
            return np.asarray(value)
        else:
            return [np.asarray(x) for x in value]

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
    lazy: Literal[False] = False,
) -> AequilibraEMatrix: ...


@overload
def from_file(
    path: str | os.PathLike,
    mode: Literal["r", "w", "a", "r+", "w-", "x"] = "r",
    *,
    index_name: str = "main_index",
    subset: None = None,
    lazy: Literal[True] = True,
) -> MatrixStore: ...


def from_file(
    path: str | os.PathLike,
    mode: Literal["r", "w", "a", "r+", "w-", "x"] = "r",
    *,
    index_name: str = "main_index",
    subset: Sequence[str | dict[str, Any]] | None = None,
    lazy: bool = False,
) -> AequilibraEMatrix | MatrixStore:
    if lazy:
        if subset is not None:
            raise ValueError("both lazy and subset may not provided at once")
        return MatrixStore(path=path, index_name=index_name, mode=mode)
    else:
        return AequilibraEMatrix.from_omx(path=path, index_name=index_name, subset=subset)


@overload
def from_dict(
    data: Mapping[str, np.ndarray],
    index: np.ndarray,
    *,
    path: None = None,
    mode: None = None,
    index_name: str = "main_index",
    metadata: Mapping[str, Any] | None = None,
    lazy: Literal[False] = False,
) -> AequilibraEMatrix: ...


@overload
def from_dict(
    data: Mapping[str, np.ndarray],
    index: np.ndarray,
    *,
    path: str | os.PathLike,
    mode: Literal["w", "a", "r+", "w-", "x"] = "a",
    index_name: str = "main_index",
    metadata: Mapping[str, Any] | None = None,
    lazy: Literal[True] = True,
) -> MatrixStore: ...


def from_dict(
    data: Mapping[str, np.ndarray],
    index: np.ndarray,
    *,
    path: str | os.PathLike | None = None,
    mode: Literal["w", "a", "r+", "w-", "x"] | None = None,
    index_name: str = "main_index",
    metadata: Mapping[str, Any] | None = None,
    lazy: bool = False,
) -> AequilibraEMatrix | MatrixStore:
    if lazy:
        if path is None:
            raise ValueError("a lazy matrix must have a path and a mode")
        mat = MatrixStore(
            path=path,
            index=index,
            index_name=index_name,
            mode=mode if mode is not None else "a",
        )
    else:
        if path is not None:
            raise ValueError("a non-lazy matrix cannot have a path")
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
    index: np.ndarray | None = None,
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
        values_matrix = np.full(mat.shape, fill_value, dtype="float64")
        values_matrix[rows, cols] = 0
        np.add.at(values_matrix, (rows, cols), values)
        mat[name] = values_matrix

    return mat


def from_scipy(
    data: Mapping[str, scipy.sparse.sparray],
    index: np.ndarray,
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
