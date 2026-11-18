from dataclasses import dataclass, field

import numpy as np

from aequilibrae.matrix.aequilibrae_matrix import AequilibraeMatrix, AequilibraeSparseMatrix
from aequilibrae.transit.transit_assignment_results import TransitAssignmentResults


@dataclass(frozen=True, kw_only=True)
class TransitClass:
    """Transit demand bound to a transit graph for hyperpath assignment."""

    name: str
    matrix: AequilibraeMatrix | AequilibraeSparseMatrix
    graph: object = field(default=None, repr=False, compare=False)
    results: TransitAssignmentResults = field(default_factory=TransitAssignmentResults, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a nonempty string")
        if not isinstance(self.matrix, (AequilibraeMatrix, AequilibraeSparseMatrix)):
            raise TypeError("Matrix must be an AequilibraeMatrix or AequilibraeSparseMatrix")
        if any(values.dtype != np.float64 for values in self.matrix.values()):
            raise TypeError("Matrix values need to be of type np.float64")

    @property
    def _id(self):
        return self.name

    @property
    def info(self):
        return {
            "name": self.name,
            "matrix_totals": {name: float(values.sum()) for name, values in self.matrix.items()},
        }
