import logging
import socket

import pandas as pd

from aequilibrae._assignment import AssignmentBase
from aequilibrae.transit.optimal_strategies import OptimalStrategies
from aequilibrae.transit.transit_class import TransitClass
from aequilibrae.utils.core_setter import clamp_cores

logger = logging.getLogger(__name__)


class TransitAssignment(AssignmentBase):
    all_algorithms = ["optimal-strategies", "os"]

    def __init__(self, *args, project=None, **kwargs):
        super().__init__(*args, project=project, **kwargs)

        self._config["Skimming Fields"] = None

    def set_classes(self, classes: list[TransitClass]) -> None:
        """
        Sets Transit classes to be assigned

        :Arguments:
            **classes** (:obj:`List[TransitClass]`): List of TransitClass's for assignment
        """

        ids = {x._id for x in classes}
        if len(ids) < len(classes):
            raise ValueError("Classes need to be unique. Your list of classes has repeated items/IDs")
        self.classes = classes  # type: List[TransitClass]

    def add_class(self, transport_class: TransitClass) -> None:
        """
        Adds a Transit class to the assignment

        :Arguments:
            **transport_class** (:obj:`TransitClass`): Transit class
        """

        ids = [x._id for x in self.classes if x._id == transport_class._id]
        if len(ids) > 0:
            raise ValueError("Transport class already in the assignment")

        self.classes.append(transport_class)

    def _check_field(self, field: str, allow_zeros=False) -> None:
        """Throws exception if field is invalid."""
        if not self.classes:
            raise ValueError("You need add at least one transit class first")

        # Validation is performed when preparing classes in execute()
        # as graphs are built at that time

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

    def set_cores(
        self, cores: int, threading_threshold: int | None = None, elementwise_cores: int | None = None
    ) -> None:
        """Allows one to set the number of cores to be used AFTER transit classes have been added

        Inherited from :obj:`AssignmentResultsBase`

        :Arguments:
            **cores** (:obj:`int`): Number of CPU cores to use

            **threading_threshold** (:obj:`int`, `Optional`): Minimum number of array elements for elementwise
            operations to be threaded. Negative values disable threading for those operations. When not provided,
            the current value is kept

            **elementwise_cores** (:obj:`int`, `Optional`): Number of CPU cores for elementwise
            (``parallel_numpy``/VDF) operations. When not provided, it is resolved from the
            ``AEQ_ELEMENTWISE_CPUS`` environment variable or ``parameters.yml``, defaulting to at most
            8 threads
        """
        if not self.classes:
            raise RuntimeError("You need load transit classes before overwriting the number of cores")

        self.cores = clamp_cores(cores)
        for c in self.classes:
            c.results.set_cores(self.cores, threading_threshold, elementwise_cores)

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

            uclass["matrix_totals"] = cls.matrix.totals()
            uclass["network mode"] = getattr(cls.matrix.index, "name", None) or f"transit_{cls._id}"

            classes[cls._id] = uclass

        info = {
            "Algorithm": self.algorithm,
            "Classes": classes,
            "Computer name": socket.gethostname(),
            "Procedure ID": self.procedure_id,
        }
        return info

    def log_specification(self):
        logger.info("Transit Class specification")
        for cls in self.classes:
            logger.info(str(cls.info))

        logger.info("Transit Assignment specification")
        logger.info(self._config)

    def save_results(self, table_name: str, keep_zero_flows=True, project=None) -> None:
        """Saves the assignment results to results_database.sqlite

        Method fails if table exists

        :Arguments:
            **table_name** (:obj:`str`): Name of the table to hold this assignment result

            **keep_zero_flows** (:obj:`bool`): Whether we should keep records for zero flows. Defaults to ``True``

            **project** (:obj:`Project`, *Optional*): Project we want to save the results to.
            Defaults to the explicitly supplied constructor project
        """

        df = self.results()
        if not keep_zero_flows:
            df = df[df.volume > 0]

        project = self._resolve_project(project)

        report = {"setup": self.info()}
        project.results.create(
            table_name=table_name,
            data=df,
            procedure="transit assignment",
            procedure_id=self.procedure_id,
            procedure_report=report,
            timestamp=self.procedure_date,
            description=self.description,
        )

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
        if not isinstance(time_field, str) or not time_field:
            raise ValueError("time_field must be a nonempty string")
        # FIXME(transit): OptimalStrategies.execute still uses each prepared graph's
        # time field. Validate the selected field against it or rebuild its context.
        self.time_field = time_field
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

        # FIXME(transit): Forward skim changes, including None, to each prepared
        # graph's set_skimming_fields() before OptimalStrategies.execute uses its context.
        fields = list(skimming_fields or [])
        self._config["Skimming Fields"] = fields
