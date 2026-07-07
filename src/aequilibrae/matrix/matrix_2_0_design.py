"""AequilibraE 2.0 matrix interface — design proposal.

This file is a self-contained interface specification, not shipping code:
signatures and semantics are normative, bodies raise NotImplementedError.

Mental model
------------
A :class:`Matrix` is an open OMX file that behaves like a ``dict`` of named
2D NumPy arrays ("cores") sharing one or more zone indices (OMX "mappings").

OMX is the only storage format. Ways into and out of other representations
(arrays, DataFrames, legacy AEM files) exist only as functions/classmethods;
the class itself performs no non-OMX io.

Key semantic decisions
----------------------
1. Explicit lifecycle. ``Matrix.open(path, mode)`` / ``Matrix.create(path,
   index=...)`` replace the 1.x two-phase ``AequilibraeMatrix()`` +
   ``create_empty()/load()/create_from_omx()`` dance. Both are context
   managers. ``Matrix.create(None, ...)`` gives a temp-file-backed scratch
   matrix, replacing ``memory_only=True`` and ``random_name()``.

2. Buffered reads, explicit flush. ``mat["car"]`` reads the core once and
   returns the cached in-memory array; subsequent accesses return the same
   buffer, so in-place idioms like ``np.fill_diagonal(mat["time"], 0)`` work
   naturally. ``mat["car"] = arr`` replaces the buffer. Nothing touches disk
   until ``save()``, ``close()`` or context exit. ``mode="r"`` forbids writes
   at assignment time, not at save time.

3. Index is required at creation and its length is immutable (it fixes the
   matrix shape, as in OMX itself). Additional mappings are managed through
   ``mat.indices``. There is no stateful ``set_index()``/``current_index``;
   anything needing a specific mapping takes it by name.

4. Matrix objects never cross into Cython. Kernels receive plain buffers
   produced by ``to_array()``: always 3D ``(zones, zones, k)`` — even for
   k=1 — C-contiguous, requested dtype, any core subset in any order. Typed
   kernel signatures (``double[:, :, ::1] demand, long long[::1] index``)
   replace the 1.x ``computational_view``/``matrix_view``/``view_names``
   machinery, the core-adjacency restriction, and the in-place reshape hacks
   in ``all_or_nothing.py`` and ``AssignmentResults``. Consumers such as
   ``TrafficClass`` extract buffers at construction and may close the file
   immediately — no HDF5 handle is held during computation. Results flow
   back out through ``Matrix.from_arrays``.

Dropped without replacement (the NumPy/pandas idiom is shorter than the
method was)::

    1.x                                  2.0 idiom
    -----------------------------------  -----------------------------------
    mat.rows() / mat.columns()           mat["car"].sum(axis=1) / (axis=0)
    mat.nan_to_num()                     np.nan_to_num(mat["car"], copy=False)
    mat.export("x.csv")                  mat.to_dataframe().to_csv("x.csv")
    mat.setName(x) / setDescription(x)   mat.attrs["name"] = x
    mat.get_matrix(core, copy=True)      mat[core].copy()
    mat.create_from_omx(...)             Matrix.open(...)  (OMX is the format)
    mat.random_name() + memory_only      Matrix.create(None, ...)
    __getattr__ core access (mat.car)    removed — mat["car"] only
    computational_view / matrix_view     to_array() + typed kernel signatures
    set_index() / current_index          mat.index, mat.indices[name]

Usage
-----
Create a demand matrix from scratch::

    with Matrix.create(None, index=graph.centroids, cores=["car", "truck"]) as mat:
        mat["car"][:] = 10.0
        mat["truck"][:] = 50.0

Add a derived core to an existing skim file::

    with Matrix.open(skims_path, "r+") as imped:
        time = imped["free_flow_time_final"].copy()
        np.fill_diagonal(time, 0)
        imped["final_time_with_intrazonals"] = time + intrazonals

Trip list (long format) to matrix::

    mat = Matrix.from_dataframe("demand.omx", pd.read_csv("trips.csv"),
                                origin="from_zone", destination="to_zone")

Hand demand to computation::

    demand = mat.to_array(["car_am"])   # (zones, zones, 1) float64 C-contiguous
    index = mat.index                   # (zones,) int64
    mat.close()                         # nothing else needs the file

Write skims produced by computation::

    Matrix.from_arrays(path, dict(zip(skim_names, arrays)), index=graph.centroids)
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Iterator, MutableMapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, DTypeLike


class Indices(MutableMapping[str, np.ndarray]):
    """dict-like view over the OMX mappings (zone indices) of a :class:`Matrix`.

    Values are 1D integer arrays of length ``zones``. The first mapping ever
    created is the default index, exposed as ``Matrix.index``. Mappings can be
    added (``mat.indices["census"] = arr``), replaced, or deleted at any time,
    except that the default index cannot be deleted and no mapping may change
    length. Reads are buffered and writes flushed exactly as for cores.
    """

    def __getitem__(self, name: str) -> np.ndarray:
        raise NotImplementedError

    def __setitem__(self, name: str, values: ArrayLike) -> None:
        """Add or replace a mapping. ``len(values)`` must equal ``zones`` and
        values must be unique integers. Raises OSError on a read-only file."""
        raise NotImplementedError

    def __delitem__(self, name: str) -> None:
        """Remove a mapping. Deleting the default index raises ValueError."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class Matrix:
    """An OMX-backed matrix: a mapping of core names to (zones, zones) arrays.

    Instances are created through :meth:`open` or :meth:`create` (or the
    conversion constructors :meth:`from_arrays` / :meth:`from_dataframe`),
    never by calling the class directly. All constructors return an object
    holding an open OMX file handle; use as a context manager or call
    :meth:`close` when done.
    """

    # ── lifecycle ────────────────────────────────────────────────────────

    @classmethod
    def open(cls, path: str | PathLike, mode: str = "r") -> "Matrix":
        """Open an existing OMX file.

        :param path: Path to the OMX file.
        :param mode: ``"r"`` (read-only, default) or ``"r+"`` (editable).
        :raises FileNotFoundError: if the file does not exist.
        :raises ValueError: if the file has no mapping (an un-indexed matrix
            cannot be related to a model) or is not square.
        """
        raise NotImplementedError

    @classmethod
    def create(
        cls,
        path: str | PathLike | None,
        index: ArrayLike,
        *,
        index_name: str = "index",
        cores: Sequence[str] = (),
        dtype: DTypeLike = np.float64,
        fill: float = 0.0,
        attrs: dict | None = None,
    ) -> "Matrix":
        """Create a new OMX file and open it in ``"r+"`` mode.

        :param path: Path for the new file. ``None`` creates a scratch matrix
            backed by a temp file, deleted on :meth:`close` (the 2.0
            replacement for ``memory_only=True`` and ``random_name()``).
        :param index: 1D array of unique integer zone IDs. Required — it fixes
            the matrix shape for the lifetime of the file, and becomes the
            default index under the name ``index_name``.
        :param cores: Optional core names to pre-declare, filled with ``fill``
            at ``dtype``. Cores can equally be added later by assignment.
        :param attrs: Optional initial file-level attributes
            (e.g. ``{"description": ...}``).
        :raises FileExistsError: if ``path`` already exists.
        """
        raise NotImplementedError

    def save(self) -> None:
        """Flush all in-memory edits (dirty core buffers, mappings, attrs) to
        disk now. No-op if nothing is dirty. Raises OSError in ``"r"`` mode
        only if there is something to flush (which cannot happen, as writes
        are rejected at assignment time)."""
        raise NotImplementedError

    def close(self) -> None:
        """Flush (if writable) and close the file. Scratch matrices delete
        their backing temp file. Any array previously returned by
        ``__getitem__`` remains valid — it is an ordinary in-memory ndarray —
        but is no longer connected to anything. Idempotent."""
        raise NotImplementedError

    def __enter__(self) -> "Matrix":
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Equivalent to :meth:`close`. On an exception in ``"r+"`` mode the
        file is still flushed — partially updated files beat silently
        discarded work; callers wanting transactional behaviour write to a
        scratch matrix and :meth:`copy` on success."""
        raise NotImplementedError

    # ── cores: MutableMapping protocol ───────────────────────────────────

    def __getitem__(self, name: str) -> np.ndarray:
        """Return the core as a (zones, zones) array.

        The array is read from disk once and cached; subsequent accesses
        return the same buffer, so in-place mutation is visible to later reads
        and is persisted by :meth:`save`/:meth:`close` (in ``"r+"`` mode; in
        ``"r"`` mode in-place edits are simply never written back).

        :raises KeyError: if the core does not exist.
        """
        raise NotImplementedError

    def __setitem__(self, name: str, values: ArrayLike) -> None:
        """Add a new core or replace an existing one.

        ``values`` must broadcast to ``(zones, zones)``; scalars are allowed
        (``mat["ones"] = 1.0``). The write lands on disk at the next
        :meth:`save`/:meth:`close`.

        :raises OSError: if the file is open read-only.
        """
        raise NotImplementedError

    def __delitem__(self, name: str) -> None:
        raise NotImplementedError

    def __contains__(self, name: object) -> bool:
        raise NotImplementedError

    def __iter__(self) -> Iterator[str]:
        """Iterate over core names."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Number of cores."""
        raise NotImplementedError

    def keys(self) -> Sequence[str]:
        raise NotImplementedError

    def items(self) -> Iterator[tuple[str, np.ndarray]]:
        raise NotImplementedError

    def rename(self, old: str, new: str) -> None:
        """Rename a core, preserving its data and attributes."""
        raise NotImplementedError

    def __repr__(self) -> str:
        # e.g. <Matrix 'demand.omx' (r+): 3317 zones, cores=['car', 'truck']>
        raise NotImplementedError

    # ── indices (OMX mappings) ───────────────────────────────────────────

    @property
    def index(self) -> np.ndarray:
        """The default (first-created) zone index — 1D int array of length
        ``zones``. Shorthand for ``mat.indices[<default name>]``."""
        raise NotImplementedError

    @property
    def indices(self) -> Indices:
        """All zone indices (OMX mappings) as a mutable, dict-like view."""
        raise NotImplementedError

    # ── shape & metadata ─────────────────────────────────────────────────

    @property
    def zones(self) -> int:
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, int]:
        """``(zones, zones)``."""
        raise NotImplementedError

    @property
    def cores(self) -> list[str]:
        """Core names, equal to ``list(mat)``."""
        raise NotImplementedError

    @property
    def path(self) -> Path:
        raise NotImplementedError

    @property
    def mode(self) -> str:
        """``"r"`` or ``"r+"``."""
        raise NotImplementedError

    @property
    def attrs(self) -> MutableMapping:
        """File-level OMX attributes (name, description, anything
        JSON-serialisable). Mutable in ``"r+"`` mode; flushed with
        :meth:`save`. Per-core attributes, should they prove needed, would be
        exposed as ``mat.core_attrs(name)`` without complicating this path."""
        raise NotImplementedError

    # ── conversion out (no file io) ──────────────────────────────────────

    def to_array(
        self,
        cores: Sequence[str] | None = None,
        *,
        dtype: DTypeLike = np.float64,
    ) -> np.ndarray:
        """Stack cores into a computation-ready dense block.

        This is the stateless successor to 1.x ``computational_view`` and the
        one gateway from Matrix to the Cython kernels, which type it as
        ``double[:, :, ::1]`` and never see the Matrix object itself.

        Guarantees: always 3D ``(zones, zones, len(cores))`` — even for a
        single core — C-contiguous, in ``dtype``. Cores may be any subset in
        any order; there is no adjacency requirement. The result is a copy:
        for very large models an ``out=`` parameter accepting a
        caller-provided (possibly memmapped) array is the anticipated
        extension, deliberately left out of the initial interface.

        :param cores: Core names to stack, defaulting to all cores in
            :attr:`cores` order.
        """
        raise NotImplementedError

    def to_dataframe(
        self,
        cores: Sequence[str] | None = None,
        *,
        index: str | None = None,
        sparse: bool = True,
    ) -> pd.DataFrame:
        """Long-format DataFrame: one row per OD pair, one column per core,
        indexed by (origin, destination) zone IDs.

        :param index: Name of the mapping used to label zones. Defaults to
            the default index.
        :param sparse: If True (default) rows where every requested core is
            zero/NaN are omitted. CSV export is then simply
            ``mat.to_dataframe().to_csv(path)`` — pandas does the io.
        """
        raise NotImplementedError

    def to_dict(self, cores: Sequence[str] | None = None) -> dict[str, np.ndarray]:
        """``{name: (zones, zones) array}``. Arrays are copies, detached from
        the Matrix — useful for handing data past :meth:`close`."""
        raise NotImplementedError

    # ── conversion in (classmethod constructors, in-memory sources) ─────

    @classmethod
    def from_arrays(
        cls,
        path: str | PathLike | None,
        arrays: dict[str, ArrayLike],
        index: ArrayLike | None = None,
        *,
        attrs: dict | None = None,
    ) -> "Matrix":
        """Create a new OMX file from named 2D arrays and open it in ``"r+"``.

        The write-out path for computation results::

            Matrix.from_arrays(path, dict(zip(skim_names, arrays)),
                               index=graph.centroids)

        :param index: Zone IDs; defaults to ``np.arange(zones)`` (with a
            warning, as an unlabelled matrix is rarely what a model wants).
        :raises ValueError: if arrays are not square and identically shaped.
        """
        raise NotImplementedError

    @classmethod
    def from_dataframe(
        cls,
        path: str | PathLike | None,
        df: pd.DataFrame,
        *,
        origin: str,
        destination: str,
        cores: Sequence[str] | None = None,
        index: ArrayLike | None = None,
    ) -> "Matrix":
        """Create a new OMX file from a long-format (trip list) DataFrame.

        Successor to 1.x ``create_from_trip_list`` — takes a DataFrame, not a
        CSV path; reading the file is pandas' job. Duplicate (origin,
        destination) rows are summed.

        :param cores: Value columns to convert; defaults to every column
            other than ``origin`` and ``destination``.
        :param index: Zone IDs for the output. Defaults to the sorted union
            of IDs appearing in ``origin``/``destination``; pass explicitly to
            include empty zones so the shape matches the model's graph.
        """
        raise NotImplementedError

    # ── convenience ──────────────────────────────────────────────────────

    def copy(
        self,
        path: str | PathLike | None,
        cores: Sequence[str] | None = None,
        *,
        rename: dict[str, str] | None = None,
    ) -> "Matrix":
        """Copy this matrix (or a subset of cores) to a new OMX file and
        return it open in ``"r+"``. All mappings and file attributes are
        copied. ``path=None`` yields a scratch copy.

        :param rename: Old-name to new-name overrides for the copied cores.
        """
        raise NotImplementedError


# ── legacy bridge (module function — the class performs no non-OMX io) ────


def aem_to_omx(aem_path: str | PathLike, omx_path: str | PathLike) -> Path:
    """Convert a 1.x AEM file to OMX, carrying over all cores, all indices,
    and the name/description attributes. Returns ``omx_path``.

    The single supported touchpoint for legacy AEM files; nothing else in the
    2.0 API reads or writes them.
    """
    raise NotImplementedError
