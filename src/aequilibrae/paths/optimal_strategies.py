import logging

from aequilibrae.paths.cython.demand import assignment_demand

logger = logging.getLogger(__name__)


class OptimalStrategies:
    def __init__(self, assig_spec):
        self.__assig_spec = assig_spec  # type: TransitAssignment

    def execute(self):
        self.__classes = {}
        self.__results = {}
        self.__demand = {}

        for cls in self.__assig_spec.classes:
            if len(cls.matrix) != 1:
                raise ValueError("Transit assignment requires exactly one demand matrix per class")

            self.__demand[cls._id] = assignment_demand(cls.matrix, cls.graph.centroids, transpose=True)
            cls.results.prepare(cls.graph, cls.matrix)

            self.__results[cls._id] = cls.results
            self.__classes[cls._id] = cls.graph.context

        for cls in self.__assig_spec.classes:
            hyperpath = self.__classes[cls._id]

            logger.info(f"Executing S&F assignment  for {cls._id}")

            hyperpath.assign(self.__demand[cls._id], threads=self.__assig_spec.cores)
            self.__results[cls._id].link_loads = hyperpath._edges["volume"].values
            # Skims share the packed dense buffer and use the demand's zone IDs.
            self.__results[cls._id].skims = (
                hyperpath.skim_matrix.to_matrix(cls.matrix.index, cls.matrix.index_name)
                if hyperpath._skimming
                else None
            )
