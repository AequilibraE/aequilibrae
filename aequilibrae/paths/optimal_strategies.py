import logging
from aequilibrae.paths.public_transport import HyperpathGenerating


class OptimalStrategies:
    def __init__(self, assig_spec):
        self.__assig_spec = assig_spec  # type: TransitAssignment
        self.__logger = assig_spec.logger
        self.__classes = {}
        self.__results = {}

        for cls in self.__assig_spec.classes:
            cls.results.prepare(cls.graph, cls.matrix)

            self.__results[cls._id] = cls.results
            self.__classes[cls._id] = HyperpathGenerating(
                cls.graph,
                cls.matrix.matrix[cls._id],  # FIXME: this is not the correct way to index the matrices
                assignment_config=assig_spec._config,
                threads=cls.results.cores,
            )

    def execute(self):
        for cls_id, hyperpath in self.__classes.items():
            self.__logger.info(f"Executing S&F assignment  for {cls_id}")

            hyperpath.execute()
            self.__results[cls_id].link_loads = hyperpath._edges["volume"].values

    # def run(self, origin=None, destination=None, volume=None):
    #     for cls_id, hyperpath in self.__classes.items():
    #         self.__logger.info(f"Executing S&F single run for {cls_id}")

    #         hyperpath.run(origin, destination, volume)
    #         self.__results[cls_id].link_loads.data["volume"] = hyperpath._edges["volume"].values
