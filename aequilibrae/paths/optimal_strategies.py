import logging
from aequilibrae.paths.public_transport import HyperpathGenerating


class OptimalStrategies:
    def __init__(self, assig_spec):
        self.__assig_spec = assig_spec  # type: TransitAssignment
        self.__logger = assig_spec.logger

        self.__classes = {
            cls._id: HyperpathGenerating(
                cls.graph.graph,
                cls.matrix.matrix[cls._id],  # FIXME: this is not the correct way to index the matrices
                trav_time=assig_spec._config["Time field"],
                freq=assig_spec._config["Frequency field"],
                threads=cls.results.cores,
            )
            for cls in self.__assig_spec.classes
        }

    def execute(self):
        for cls_id, hyperpath in self.__classes.items():
            self.__logger.info(f"Executing S&F assignment  for {cls_id}")

            hyperpath.execute()
