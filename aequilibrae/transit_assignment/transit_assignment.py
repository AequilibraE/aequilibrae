import json
import socket

import numpy as np
import pandas as pd

from aequilibrae.context import get_active_project
from aequilibrae.paths.assignment_base import AssignmentBase
from aequilibrae.transit_assignment.optimal_strategies import OptimalStrategies
from aequilibrae.utils.core_setter import set_cores


class TransitAssignment(AssignmentBase):
    all_algorithms = ["optimal-strategies", "os"]

    def __init__(self, *args, project=None, **kwargs):
        super().__init__(*args, **kwargs)

        self._config["Skimming Fields"] = None

    def set_algorithm(self, algorithm: str):
        """
        Chooses the assignment algorithm. Currently only 'optimal-strategies' is available.

        'os' is also accepted as an alternative to 'optimal-strategies'

        :Arguments:
            **algorithm** (:obj:`str`): Algorithm to be used
        """
        algo_dict = {i: i for i in self.all_algorithms}
        algo_dict["os"] = "optimal-strategies"
        algo = algo_dict.get(algorithm.lower())

        if algo is None:
            raise AttributeError(f"Assignment algorithm not available. Choose from: {','.join(self.all_algorithms)}")

        self.algorithm = algo
        self._config["Algorithm"] = algo
        self.assignment = OptimalStrategies(self)

    def set_cores(self, cores: int) -> None:
        """Allows one to set the number of cores to be used AFTER transit classes have been added

        Inherited from :obj:`AssignmentResultsBase`

        :Arguments:
            **cores** (:obj:`int`): Number of CPU cores to use
        """
        if not self.classes:
            raise RuntimeError("You need load transit classes before overwriting the number of cores")

        self.cores = set_cores(cores)
        for c in self.classes:
            c.results.set_cores(self.cores)

    def info(self) -> dict:
        """Returns information for the transit assignment procedure

        Dictionary contains keys  'Algorithm', 'Classes', 'Computer name', 'Procedure ID'.

        The classes key is also a dictionary with all the user classes per transit class and their respective
        matrix totals

        :Returns:
            **info** (:obj:`dict`): Dictionary with summary information
        """

        classes = {}

        for cls in self.classes:
            uclass = {}

            if len(cls.matrix.view_names) == 1:
                uclass["matrix_totals"] = {nm: np.sum(cls.matrix.matrix_view[:, :]) for nm in cls.matrix.view_names}
            else:
                uclass["matrix_totals"] = {
                    nm: np.sum(cls.matrix.matrix_view[:, :, i]) for i, nm in enumerate(cls.matrix.view_names)
                }
            uclass["network mode"] = cls.graph.mode

            classes[cls._id] = uclass

        info = {
            "Algorithm": self.algorithm,
            "Classes": classes,
            "Computer name": socket.gethostname(),
            "Procedure ID": self.procedure_id,
        }
        return info

    def log_specification(self):
        self.logger.info("Transit Class specification")
        for cls in self.classes:
            self.logger.info(str(cls.info))

        self.logger.info("Transit Assignment specification")
        self.logger.info(self._config)

    def save_results(self, table_name: str, keep_zero_flows=True, project=None) -> None:
        """Saves the assignment results to results_database.sqlite

        Method fails if table exists

        :Arguments:
            **table_name** (:obj:`str`): Name of the table to hold this assignment result

            **keep_zero_flows** (:obj:`bool`): Whether we should keep records for zero flows. Defaults to ``True``

            **project** (:obj:`Project`, *Optional*): Project we want to save the results to.
            Defaults to the active project
        """

        df = self.results()
        if not keep_zero_flows:
            df = df[df.volume > 0]

        if not project:
            project = project or get_active_project()

        report = {"setup": self.info()}
        record = project.results.new_record(
            table_name=table_name,
            procedure="transit assignment",
            procedure_id=self.procedure_id,
            procedure_report=json.dumps(report),
            timestamp=self.procedure_date,
            description=self.description,
        )
        record.set_data(df)

    def results(self) -> pd.DataFrame:
        """Prepares the assignment results as a Pandas DataFrame

        :Returns:
            **DataFrame** (:obj:`pd.DataFrame`): Pandas DataFrame with all the assignment results indexed on *link_id*
        """
        assig_results = [
            pd.DataFrame(cls.results.get_load_results()).rename(columns={"volume": cls._id + "_volume"})
            for cls in self.classes
        ]

        return pd.concat(assig_results, axis=1)

    def set_time_field(self, time_field: str) -> None:
        """
        Sets the graph field that contains free flow travel time -> e.g. 'trav_time'

        :Arguments:
            **time_field** (:obj:`str`): Field name
        """
        super().set_time_field(time_field)
        self._config["Time field"] = time_field

    def set_frequency_field(self, frequency_field: str) -> None:
        """
        Sets the graph field that contains the frequency -> e.g. 'freq'

        :Arguments:
            **frequency_field** (:obj:`str`): Field name
        """
        self._check_field(frequency_field)
        self._config["Frequency field"] = frequency_field

    def set_skimming_fields(self, skimming_fields: list[str] = None) -> None:
        """
        Sets the skimming fields for the transit assignment.

        Also accepts predefined skimming fields:
            - discrete: 'boardings', 'alightings', 'inner_transfers', 'outer_transfers', and 'transfers'.
            - continuous: 'trav_time', 'on_board_trav_time', 'dwelling_time', 'egress_trav_time', 'access_trav_time',
              'walking_trav_time', 'transfer_time', 'in_vehicle_trav_time', and 'waiting_time'.

        Provide no argument to disable.

        :Arguments:
            **skimming_fields** (:obj:`list[str]`): Optional list of field names, or predefined skimming type.
        """

        if skimming_fields:
            if isinstance(skimming_fields, (tuple, set)):
                skimming_fields = list(skimming_fields)

            if not isinstance(skimming_fields, list):
                raise TypeError("Skimming Fields should be defined on a list, tuple or set")

        self._config["Skimming Fields"] = skimming_fields
