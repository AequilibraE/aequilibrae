import logging
import socket
from abc import ABC, abstractmethod
from datetime import datetime
from os import path
from typing import List, Dict, Union, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from numpy import nan_to_num

from aequilibrae import Parameters
from aequilibrae.context import get_active_project
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths.linear_approximation import LinearApproximation
from aequilibrae.paths.optimal_strategies import OptimalStrategies
from aequilibrae.paths.traffic_class import TrafficClass, TransportClassBase
from aequilibrae.paths.vdf import VDF, all_vdf_functions
from aequilibrae.utils.core_setter import set_cores
from aequilibrae.utils.db_utils import commit_and_close


class AssignmentBase(ABC):
    def __init__(self, project=None):
        self.procedure_id = uuid4().hex
        self.procedure_date = str(datetime.today())

        proj = project or get_active_project(must_exist=False)
        self.project = proj

        self.parameters = proj.parameters if proj else Parameters().parameters
        self.logger = proj.logger if proj else logging.getLogger("aequilibrae")

        self.classes: List[TrafficClass] = []
        self.algorithm: str = None
        self.time_field: str = None
        self.assignment: Union[LinearApproximation, OptimalStrategies] = None
        self.free_flow_tt: np.ndarray = None
        self.total_flow: np.ndarray = None
        self.cores: int = None
        self._config = {}

        self.description: str = ""

    def algorithms_available(self) -> list:
        """
        Returns all algorithms available for use

        :Returns:
            :obj:`list`: List of string values to be used with **set_algorithm**
        """
        return self.all_algorithms

    @abstractmethod
    def set_algorithm(self, algorithm: str):
        pass

    @abstractmethod
    def set_cores(self, cores: int) -> None:
        pass

    def execute(self, log_specification=True) -> None:
        """Processes assignment"""
        if log_specification:
            self.log_specification()
        self.assignment.execute()

    @abstractmethod
    def log_specification(self):
        pass

    @abstractmethod
    def save_results(self, table_name: str, keep_zero_flows=True, project=None) -> None:
        pass

    @abstractmethod
    def results(self) -> pd.DataFrame:
        pass

    def report(self) -> pd.DataFrame:
        """Returns the assignment convergence report

        :Returns:
           **DataFrame** (:obj:`pd.DataFrame`): Convergence report
        """
        return pd.DataFrame(self.assignment.convergence_report)

    @abstractmethod
    def info(self) -> dict:
        pass

    def set_classes(self, classes: List[TransportClassBase]) -> None:
        """
        Sets Transport classes to be assigned

        :Arguments:
            **classes** (:obj:`List[TransportClassBase]`): List of TransportClass's for assignment
        """

        ids = {x._id for x in classes}
        if len(ids) < len(classes):
            raise ValueError("Classes need to be unique. Your list of classes has repeated items/IDs")
        self.classes = classes  # type: List[TransportClassBase]

    def add_class(self, transport_class: TransportClassBase) -> None:
        """
        Adds a Transport class to the assignment

        :Arguments:
            **transport_class** (:obj:`TransportClassBase`): Transport class
        """

        ids = [x._id for x in self.classes if x._id == transport_class._id]
        if len(ids) > 0:
            raise ValueError("Transport class already in the assignment")

        self.classes.append(transport_class)

    def _check_field(self, field: str, allow_zeros=False) -> None:
        """Throws expection if field is invalid."""
        if not self.classes:
            raise ValueError("You need add at least one transport class first")

        for c in self.classes:
            if field not in c.graph.graph.columns:
                raise ValueError(f"'{field}' not in graph for '{c._id}'")

            if np.any(np.isnan(c.graph.graph[field].values)):
                raise ValueError(f"At least one link for {field} is NaN for '{c._id}'")

            if c.graph.graph[field].values.min() <= 0 and not allow_zeros:
                raise ValueError(f"There is at least one link with zero or negative {field} for '{c._id}'")

    def set_time_field(self, time_field: str) -> None:
        self._check_field(time_field)
        c = self.classes[0]
        self.free_flow_tt = np.zeros(c.graph.graph.shape[0], c.graph.default_types("float"))
        self.free_flow_tt[c.graph.graph.__supernet_id__] = c.graph.graph[time_field]
        self.total_flow = np.zeros(self.free_flow_tt.shape[0], np.float64)
        self.time_field = time_field

    def get_skim_results(self) -> list:
        """Prepares the assignment skim results for all classes

        :Returns:
            **skim list** (:obj:`list`): Lists of all skims with the results for each class
        """
        return {cls._id: cls.results.skims for cls in self.classes}


class TrafficAssignment(AssignmentBase):
    """Traffic assignment class.

    For a comprehensive example on use, see the Use examples page.

    .. code-block:: python

        >>> from aequilibrae.paths import TrafficAssignment, TrafficClass

        >>> project = create_example(project_path)
        >>> project.network.build_graphs()

        >>> graph = project.network.graphs['c'] # we grab the graph for cars
        >>> graph.set_graph('free_flow_time') # let's say we want to minimize time
        >>> graph.set_skimming(['free_flow_time', 'distance']) # And will skim time and distance
        >>> graph.set_blocked_centroid_flows(True)

        >>> proj_matrices = project.matrices

        >>> demand = proj_matrices.get_matrix("demand_omx")

        # We will only assign one user class stored as 'matrix' inside the OMX file
        >>> demand.computational_view(['matrix'])

        # Creates the assignment class
        >>> assigclass = TrafficClass("car", graph, demand)

        >>> assig = TrafficAssignment()

        # The first thing to do is to add at list of traffic classes to be assigned
        >>> assig.set_classes([assigclass])

        # Then we set the volume delay function
        >>> assig.set_vdf("BPR")  # This is not case-sensitive

        # And its parameters
        >>> assig.set_vdf_parameters({"alpha": "b", "beta": "power"})

        # The capacity and free flow travel times as they exist in the graph
        >>> assig.set_capacity_field("capacity")
        >>> assig.set_time_field("free_flow_time")

        # And the algorithm we want to use to assign
        >>> assig.set_algorithm('bfw')

        >>> assig.max_iter = 10
        >>> assig.rgap_target = 0.00001

        >>> assig.execute() # we then execute the assignment

        # If you want, it is possible to access the convergence report
        >>> convergence_report = pd.DataFrame(assig.assignment.convergence_report)

        # Assignment results can be viewed as a Pandas DataFrame
        >>> results_df = assig.results()

        # Information on the assignment setup can be recovered with
        >>> info = assig.info()

        # Or save it directly to the results database
        >>> results = assig.save_results(table_name='base_year_assignment')

        # skims are here
        >>> avg_skims = assigclass.results.skims # blended ones
        >>> last_skims = assigclass._aon_results.skims # those for the last iteration
    """

    bpr_parameters = ["alpha", "beta"]
    all_algorithms = ["all-or-nothing", "msa", "frank-wolfe", "fw", "cfw", "bfw"]

    def __init__(self, project=None) -> None:
        """"""
        self.__dict__["_TrafficAssignment__initalised"] = False
        super().__init__(project=project)

        proj = project or get_active_project(must_exist=False)

        par = proj.parameters if proj else Parameters().parameters
        parameters = par["assignment"]["equilibrium"]

        self.rgap_target = parameters["rgap"]
        self.max_iter = parameters["maximum_iterations"]
        self.vdf = VDF()
        self.vdf_parameters = None  # type: list
        self.capacity_field = None  # type: str
        self.capacity = None  # type: np.ndarray
        self.congested_time = None  # type: np.ndarray
        self.save_path_files = False  # type: bool
        self.preloads = None  # type: pd.DataFrame

        self.steps_below_needed_to_terminate = 1

        self._config = {}

        self.__initalised = True

    def __setattr__(self, name, value) -> None:
        # Special methods (__method__) cannot be override at runtime, instead we'll
        # just set a variable to indicate if the checking of attributes should be performed
        if self.__initalised:
            check, value, message = self.__check_attributes(name, value)
            if not check:
                raise ValueError(message)
        super().__setattr__(name, value)

    def __check_attributes(self, instance, value):
        if instance == "rgap_target":
            if not isinstance(value, float):
                return False, value, "Relative gap needs to be a float"
            if isinstance(self.assignment, LinearApproximation):
                self.assignment.rgap_target = value
        elif instance == "max_iter":
            if not isinstance(value, int):
                return False, value, "Number of iterations needs to be an integer"
            if isinstance(self.assignment, LinearApproximation):
                self.assignment.max_iter = value
        elif instance == "vdf":
            v = value.lower()
            if v not in all_vdf_functions:
                return False, value, f"Volume-delay function {value} is not available"
            value = VDF()
            value.function = v
        elif instance == "classes":
            if isinstance(value, TrafficClass):
                value = [value]
            elif isinstance(value, list):
                for v in value:
                    if not isinstance(v, TrafficClass):
                        return False, value, "Traffic classes need to be proper AssignmentClass objects"
            else:
                raise ValueError("Traffic classes need to be proper AssignmentClass objects")
        elif instance == "vdf_parameters":
            if not self.__validate_parameters(value):
                return False, value, f"Parameter set is not valid: {value} "
        elif instance in ["time_field", "capacity_field"] and not isinstance(value, str):
            return False, value, f"Value for {instance} is not string"
        elif instance == "cores" and not isinstance(value, int):
            return False, value, f"Value for {instance} is not integer"
        elif instance == "save_path_files" and not isinstance(value, bool):
            return False, value, f"Value for {instance} is not boolean"
        if instance not in self.__dict__:
            return False, value, f"TrafficAssignment class does not have property {instance}"
        return True, value, ""

    def set_vdf(self, vdf_function: str) -> None:
        """
        Sets the Volume-delay function to be used

        :Arguments:
            **vdf_function** (:obj:`str`): Name of the VDF to be used
        """
        self.vdf = vdf_function

    def set_classes(self, classes: List[TrafficClass]) -> None:
        """
        Sets Traffic classes to be assigned

        :Arguments:
            **classes** (:obj:`List[TrafficClass]`): List of Traffic classes for assignment
        """

        ids = {x._id for x in classes}
        if len(ids) < len(classes):
            raise ValueError("Classes need to be unique. Your list of classes has repeated items/IDs")
        self.classes = classes  # type: List[TrafficClass]

    def add_class(self, traffic_class: TrafficClass) -> None:
        """
        Adds a traffic class to the assignment

        :Arguments:
            **traffic_class** (:obj:`TrafficClass`): Traffic class
        """

        ids = [x._id for x in self.classes if x._id == traffic_class._id]
        if len(ids) > 0:
            raise ValueError("Traffic class already in the assignment")

        self.classes.append(traffic_class)

    # TODO: Create procedure to check that travel times, capacities and vdf parameters are equal across all graphs
    # TODO: We also need procedures to check that all graphs are compatible (i.e. originated from the same network)
    def set_algorithm(self, algorithm: str):
        """
        Chooses the assignment algorithm. e.g. 'frank-wolfe', 'bfw', 'msa'

        'fw' is also accepted as an alternative to 'frank-wolfe'

        :Arguments:
            **algorithm** (:obj:`str`): Algorithm to be used
        """

        # First we instantiate the arrays we will be using over and over

        algo_dict = {i: i for i in self.all_algorithms}
        algo_dict["fw"] = "frank-wolfe"
        algo = algo_dict.get(algorithm.lower())

        if algo is None:
            raise AttributeError(f"Assignment algorithm not available. Choose from: {','.join(self.all_algorithms)}")

        if algo in ["all-or-nothing", "msa", "frank-wolfe", "cfw", "bfw"]:
            self.assignment = LinearApproximation(self, algo, project=self.project)
        else:
            raise ValueError("Algorithm not listed in the case selection")

        self.__dict__["algorithm"] = algo
        self._config["Algorithm"] = algo
        self._config["Maximum iterations"] = self.assignment.max_iter
        self._config["Target RGAP"] = self.assignment.rgap_target

    def set_vdf_parameters(self, par: dict) -> None:
        """
        Sets the parameters for the Volume-delay function.

        Parameter values can be scalars (same values for the entire network) or network field names
        (link-specific values) - Examples: {'alpha': 0.15, 'beta': 4.0} or  {'alpha': 'alpha', 'beta': 'beta'}

        :Arguments:
            **par** (:obj:`dict`): Dictionary with all parameters for the chosen VDF

        """
        if self.classes is None or self.vdf.function.lower() not in all_vdf_functions:
            raise RuntimeError(
                "Before setting vdf parameters, you need to set traffic classes and choose a VDF function"
            )
        self.__dict__["vdf_parameters"] = par
        self._config["VDF parameters"] = par
        pars = []
        if self.vdf.function in ["BPR", "BPR2", "CONICAL", "INRETS"]:
            for p1 in ["alpha", "beta"]:
                if p1 not in par:
                    raise ValueError(f"{p1} should exist in the set of parameters provided")
                p = par[p1]
                if isinstance(self.vdf_parameters[p1], str):
                    c = self.classes[0]
                    array = np.zeros(c.graph.graph.shape[0], c.graph.default_types("float"))
                    array[c.graph.graph.__supernet_id__] = c.graph.graph[p]
                else:
                    array = np.zeros(self.classes[0].graph.graph.shape[0], np.float64)
                    array.fill(self.vdf_parameters[p1])
                pars.append(array)

                if np.any(np.isnan(array)):
                    raise ValueError(f"At least one {p1} is NaN")

                if p1 == "alpha":
                    if array.min() < 0:
                        raise ValueError(f"At least one {p1} is smaller than zero")
                else:
                    if array.min() < 1:
                        raise ValueError(f"At least one {p1} is smaller than one. Results will make no sense")

        self.__dict__["vdf_parameters"] = pars
        self._config["VDF function"] = self.vdf.function.lower()

    def set_cores(self, cores: int) -> None:
        """Allows one to set the number of cores to be used AFTER traffic classes have been added

        Inherited from :obj:`AssignmentResultsBase`

        :Arguments:
            **cores** (:obj:`int`): Number of CPU cores to use
        """
        if not self.classes:
            raise RuntimeError("You need load traffic classes before overwriting the number of cores")

        self.cores = set_cores(cores)
        for c in self.classes:
            c.results.set_cores(self.cores)
            c._aon_results.set_cores(self.cores)

    def set_save_path_files(self, save_it: bool) -> None:
        """Turn path saving on or off.

        :Arguments:
            **save_it** (:obj:`bool`): Boolean to indicate whether paths should be saved
        """
        if self.classes is None:
            raise RuntimeError("You need to set traffic classes before turning path saving on or off")

        # self.save_path_files = save_it
        for c in self.classes:
            c._aon_results.save_path_file = save_it

    def set_path_file_format(self, file_format: str) -> None:
        """Specify path saving format. Either parquet or feather.

        :Arguments:
            **file_format** (:obj:`str`): Name of file format to use for path files
        """
        if self.classes is None:
            raise RuntimeError("You need to set traffic classes before specifying path saving options")

        if file_format == "feather":
            for c in self.classes:
                c._aon_results.write_feather = True
        elif file_format == "parquet":
            for c in self.classes:
                c._aon_results.write_feather = False
        else:
            raise TypeError(f"Unsupported path file format {file_format} - only feather or parquet available.")

    def set_time_field(self, time_field: str) -> None:
        """
        Sets the graph field that contains free flow travel time -> e.g. 'fftime'

        :Arguments:
            **time_field** (:obj:`str`): Field name
        """

        super().set_time_field(time_field)

        self.__dict__["congested_time"] = np.array(self.free_flow_tt, copy=True)
        self._config["Time field"] = time_field

    def set_capacity_field(self, capacity_field: str) -> None:
        """
        Sets the graph field that contains link capacity for the assignment period -> e.g. 'capacity1h'

        :Arguments:
            **capacity_field** (:obj:`str`): Field name
        """
        super()._check_field(capacity_field)
        c = self.classes[0]

        self.cores = c.results.cores
        self.capacity = np.zeros(c.graph.graph.shape[0], c.graph.default_types("float"))
        self.capacity[c.graph.graph.__supernet_id__] = c.graph.graph[capacity_field]
        self.capacity_field = capacity_field
        self._config["Number of cores"] = c.results.cores
        self._config["Capacity field"] = capacity_field

    def add_preload(self, preload: pd.DataFrame, name: str = None) -> None:
        """
        Given a dataframe of 'link_id', 'direction' and 'preload', merge into current preloads dataframe.

        :Arguments:
            **preload** (:obj:`pd.DataFrame`): dataframe mapping 'link_id' & 'direction' to 'preload'
            **name** (:obj:`str`): Name for particular preload (optional - default name will be chosen if not specified)
        """
        # Create preloads dataframe in correct order if not already initialised
        if self.preloads is None:
            g = self.classes[0].graph.graph
            self.preloads = g.sort_values(by="__supernet_id__")[["link_id", "direction"]].copy()

        # Check that columns of preload are link_id, direction, preload:
        expected = {"link_id", "direction", "preload"}
        missing = expected - set(preload.columns)
        additional = set(preload.columns) - expected
        if missing:
            raise ValueError(f"Input preload dataframe is missing columns: {missing}\nexpected columns are {expected}")
        elif additional:
            raise ValueError(
                f"Input preload dataframe has additional columns: {additional}\nexpected columns are {expected}"
            )

        # Reject empty preloads
        if len(preload) == 0:
            raise ValueError("Cannot set empty preload!")

        # Check name is not already used (generate new name if needed):
        name = (
            name if name else f"preload_{len(self.preloads.columns) - 1}"
        )  # -1 -> remove keys to get 1 indexed preload columns
        if name in self.preloads.columns:
            raise ValueError(f"New preload has duplicate name - already used names are: {self.preload.columns}")
        preload.rename(columns={"preload": name}, inplace=True)

        # Merge onto current preload dataframe
        self.preloads = pd.merge(self.preloads, preload, on=["link_id", "direction"], how="left")
        self.preloads[name] = self.preloads[name].fillna(0)

        # Enable preload to be added before or after specifyig the algorithm
        if self.assignment is not None:
            if self.assignment.preload is None:
                self.assignment.preload = self.preloads[name].to_numpy()
            else:
                self.assignment.preload += self.preloads[name]

    # TODO: This function actually needs to return a human-readable dictionary, and not one with
    #       tons of classes. Feeds into the class above
    # def load_assignment_spec(self, specs: dict) -> None:
    #     pass
    # def get_spec(self) -> dict:
    #     """Gets the entire specification of the assignment"""
    #     return deepcopy(self.__dict__)

    def skim_congested(self, skim_fields=None, return_matrices=False) -> Optional[dict]:
        """
        Skims the congested network. The user can add a list of skims to be computed, which
        will be added to the congested time and the assignment cost from the last iteration of
        the assignment.

        The matrices are always stored internally in the AequilibraE objects to be saved to the
        project if needed. If return_matrices is set to True, the matrices are also returned.

        :Arguments:
            **skim_fields** (:obj:`Union[None, str]`): Name of the skims to use. If None, uses default only
            **return_matrices** (:obj:`Bool`): Returns a dictionary with skims. Defaults to False.
        """
        data = {}
        for assig_class in self.classes:
            skimmer = assig_class.skim_congested(skim_fields)
            assig_class._aon_results.skims = skimmer.results.skims
            data[assig_class._id] = skimmer.results.skims
        if return_matrices:
            return data

    def __validate_parameters(self, kwargs) -> bool:
        if self.vdf == "":
            raise ValueError("First you need to set the Volume-Delay Function to use")

        par = list(kwargs.keys())
        q = [x for x in par if x not in self.bpr_parameters] + [x for x in self.bpr_parameters if x not in par]
        if len(q) > 0:
            raise ValueError("List of functions {} for vdf {} has an inadequate set of parameters".format(q, self.vdf))
        return True

    def log_specification(self):
        self.logger.info("Traffic Class specification")
        for cls in self.classes:
            self.logger.info(str(cls.info))

        self.logger.info("Traffic Assignment specification")
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
            df = df[df.PCE_tot > 0]

        if not project:
            project = self.project or get_active_project()

        res_path = path.join(project.project_base_path, "results_database.sqlite")
        with commit_and_close(res_path, missing_ok=True) as conn:
            df.to_sql(table_name, conn)

        report = {"convergence": str(self.assignment.convergence_report), "setup": str(self.info())}
        data = [table_name, "traffic assignment", self.procedure_id, str(report), self.procedure_date, self.description]
        with self.project.db_connection as conn:
            conn.execute(
                """Insert into results(table_name, procedure, procedure_id, procedure_report, timestamp,
                                                description) Values(?,?,?,?,?,?)""",
                data,
            )

    def results(self) -> pd.DataFrame:
        """Prepares the assignment results as a Pandas DataFrame

        :Returns:
            **DataFrame** (:obj:`pd.DataFrame`): Pandas DataFrame with all the assignment results indexed on `link_id`
        """

        idx = self.classes[0].graph.graph.__supernet_id__
        assig_results = [cls.results.get_load_results() for cls in self.classes]

        class1 = self.classes[0]
        res1 = assig_results[0]

        tot_flow = self.assignment.fw_total_flow[idx]
        voc = tot_flow / self.capacity[idx]
        congested_time = self.congested_time[idx]
        free_flow_tt = self.free_flow_tt[idx]
        preload = np.full(len(tot_flow), np.nan) if self.assignment.preload is None else self.assignment.preload

        fields = [
            "Preload_AB",
            "Preload_BA",
            "Preload_tot",
            "Congested_Time_AB",
            "Congested_Time_BA",
            "Congested_Time_Max",
            "Delay_factor_AB",
            "Delay_factor_BA",
            "Delay_factor_Max",
            "VOC_AB",
            "VOC_BA",
            "VOC_max",
            "PCE_AB",
            "PCE_BA",
            "PCE_tot",
        ]

        agg = pd.DataFrame([], columns=fields, index=res1.index[:]).astype(float)
        agg.fillna(0.0, inplace=True)

        # Use the first class to get a graph -> network link ID mapping
        m = class1.results.get_graph_to_network_mapping()
        graph_ab, graph_ba = m.graph_ab_idx, m.graph_ba_idx
        agg["Preload_AB"].values[m.network_ab_idx] = nan_to_num(preload[m.graph_ab_idx])
        agg["Preload_BA"].values[m.network_ba_idx] = nan_to_num(preload[m.graph_ba_idx])
        agg.loc[:, "Preload_tot"] = np.nansum([agg.Preload_AB, agg.Preload_BA], axis=0)

        agg["Congested_Time_AB"].values[m.network_ab_idx] = nan_to_num(congested_time[m.graph_ab_idx])
        agg["Congested_Time_BA"].values[m.network_ba_idx] = nan_to_num(congested_time[m.graph_ba_idx])
        agg.loc[:, "Congested_Time_Max"] = np.nanmax([agg.Congested_Time_AB, agg.Congested_Time_BA], axis=0)

        agg["Delay_factor_AB"].values[m.network_ab_idx] = nan_to_num(congested_time[graph_ab] / free_flow_tt[graph_ab])
        agg["Delay_factor_BA"].values[m.network_ba_idx] = nan_to_num(congested_time[graph_ba] / free_flow_tt[graph_ba])
        agg.loc[:, "Delay_factor_Max"] = np.nanmax([agg.Delay_factor_AB, agg.Delay_factor_BA], axis=0)

        agg["VOC_AB"].values[m.network_ab_idx] = nan_to_num(voc[m.graph_ab_idx])
        agg["VOC_BA"].values[m.network_ba_idx] = nan_to_num(voc[m.graph_ba_idx])
        agg.loc[:, "VOC_max"] = np.nanmax([agg.VOC_AB, agg.VOC_BA], axis=0)

        agg["PCE_AB"].values[m.network_ab_idx] = nan_to_num(tot_flow[m.graph_ab_idx])
        agg["PCE_BA"].values[m.network_ba_idx] = nan_to_num(tot_flow[m.graph_ba_idx])
        agg.loc[:, "PCE_tot"] = np.nansum([agg.PCE_AB, agg.PCE_BA], axis=0)

        assig_results.append(agg)
        return pd.concat(assig_results, axis=1).rename_axis("link_id")

    def info(self) -> dict:
        """Returns information for the traffic assignment procedure

        Dictionary contains keys  'Algorithm', 'Classes', 'Computer name', 'Procedure ID',
        'Maximum iterations' and 'Target RGap'.

        The classes key is also a dictionary with all the user classes per traffic class and their respective
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
            uclass["Value-of-time"] = cls.vot
            uclass["PCE"] = cls.pce
            if cls.fixed_cost_field:
                uclass["Fixed cost field"] = cls.fixed_cost_field
                uclass["Fixed cost multiplier"] = cls.fc_multiplier
            uclass["save_path_files"] = cls._aon_results.save_path_file
            uclass["path_file_feather_format"] = cls._aon_results.write_feather

            classes[cls._id] = uclass

        info = {
            "Algorithm": self.algorithm,
            "Classes": classes,
            "Computer name": socket.gethostname(),
            "Maximum iterations": self.assignment.max_iter,
            "Procedure ID": self.procedure_id,
            "Target RGap": self.assignment.rgap_target,
        }
        return info

    def save_skims(self, matrix_name: str, which_ones="final", format="omx", project=None) -> None:
        """Saves the skims (if any) to the skim folder and registers in the matrix list

        :Arguments:
            **name** (:obj:`str`): Name of the matrix record to hold this matrix (same name used for file name)

            **which_ones** (:obj:`str`, *Optional*): {'final': Results of the final iteration, 'blended': Averaged results
            for all iterations, 'all': Saves skims for both the final iteration and the blended ones}.
            Default is 'final'

            **format** (:obj:`str`, *Optional*): File format ('aem' or 'omx'). Default is 'omx'

            **project** (:obj:`Project`, *Optional*): Project we want to save the results to.
            Defaults to the active project
        """
        mat_format = format.lower()
        if mat_format not in ["omx", "aem"]:
            raise ValueError("Matrix needs to be either OMX or native AequilibraE")
            raise ImportError("OpenMatrix is not available on your system")

        if not project:
            project = self.project or get_active_project()

        mats = project.matrices

        for cls in self.classes:
            file_name = f"{matrix_name}_{cls._id}.{mat_format}"

            export_name = path.join(mats.fldr, file_name)

            if path.isfile(export_name):
                raise FileExistsError(f"{file_name} already exists. Choose a different name or matrix format")

            if mats.check_exists(matrix_name):
                raise FileExistsError(f"{matrix_name} already exists. Choose a different name")

            avg_skims = cls.results.skims  # type: AequilibraeMatrix

            # The ones for the last iteration are here
            last_skims = cls._aon_results.skims  # type: AequilibraeMatrix

            names = []
            if which_ones in ["final", "all"]:
                for core in last_skims.names:
                    names.append(f"{core}_final")

            if which_ones in ["blended", "all"]:
                for core in avg_skims.names:
                    names.append(f"{core}_blended")

            if not names:
                continue
            # Assembling a single final skim file can be done like this
            # We will want only the time for the last iteration and the distance averaged out for all iterations
            working_name = export_name if mat_format == "aem" else AequilibraeMatrix().random_name()

            kwargs = {
                "file_name": working_name,
                "zones": self.classes[0].graph.centroids.shape[0],
                "matrix_names": names,
                "memory_only": False,
            }

            # Create the matrix to manipulate
            out_skims = AequilibraeMatrix()
            out_skims.create_empty(**kwargs)

            out_skims.index[:] = self.classes[0].graph.centroids[:]
            out_skims.description = f"Assignment skim from procedure ID {self.procedure_id}. Class name {cls._id}"

            if which_ones in ["final", "all"]:
                for core in last_skims.names:
                    out_skims.matrix[f"{core}_final"][:, :] = last_skims.matrix[core][:, :]

            if which_ones in ["blended", "all"]:
                for core in avg_skims.names:
                    out_skims.matrix[f"{core}_blended"][:, :] = avg_skims.matrix[core][:, :]

            out_skims.matrices.flush()  # Make sure that all data went to the disk

            # If it were supposed to be an OMX, we export to one
            if mat_format == "omx":
                out_skims.export(export_name)

            out_skims.description = f"Skimming for assignment procedure. Class {cls._id}"
            # Now we create the appropriate record

            record = mats.new_record(f"{matrix_name}_{cls._id}", file_name)
            record.procedure_id = self.procedure_id
            record.timestamp = self.procedure_date
            record.procedure = "Traffic Assignment"
            record.description = out_skims.description
            record.save()

    def select_link_flows(self) -> Dict[str, pd.DataFrame]:
        """
        Returns a dataframe of the select link flows for each class
        """
        class_flows = []  # stores the df for each class
        for cls in self.classes:
            # Save OD_matrices
            if cls._selected_links is None:
                continue
            # Create Values table
            df = cls.results.get_sl_results()
            # Remap the dataframe names to add the prefix classname for each class
            cls_cols = {x: cls._id + "_" + x if (x != "index") else "link_id" for x in df.columns}
            df.rename(columns=cls_cols, inplace=True)
            class_flows.append(df)
        return pd.concat(class_flows, axis=1).rename_axis("link_id")

    def save_select_link_flows(self, table_name: str, project=None) -> None:
        """
        Saves the select link link flows for all classes into the results database.

        :Arguments:
            **table_name** (:obj:`str`): Name of the table being inserted to. Note the traffic class

            **project** (:obj:`Project`, *Optional*): Project we want to save the results to.
            Defaults to the active project
        """

        if not project:
            project = self.project or get_active_project()
        df = self.select_link_flows()

        res_path = path.join(project.project_base_path, "results_database.sqlite")
        with commit_and_close(res_path, missing_ok=True) as conn:
            df.to_sql(table_name, conn)

        # Create description table
        self.description = f"Select link analysis from {self.procedure_id}"
        report = {}
        data = [
            table_name,
            "select link",
            f"{self.procedure_id}_sl",
            str(report),
            self.procedure_date,
            self.description,
        ]
        with self.project.db_connection as conn:
            conn.execute(
                """Insert into results(table_name, procedure, procedure_id, procedure_report, timestamp,
                                            description) Values(?,?,?,?,?,?)""",
                data,
            )

    def save_select_link_matrices(self, matrix_name: str, project=None) -> None:
        """
        Saves the Select Link matrices for each TrafficClass in the current TrafficAssignment class
        into OMX format.

        :Arguments:
            **name** (:obj:`str`): name of the matrices

            **project** (:obj:`Project`, *Optional*): Project we want to save the results to.
            Defaults to the active project
        """
        if not project:
            project = self.project or get_active_project()

        mats = project.matrices

        file_name = f"{matrix_name}.omx"

        export_name = path.join(mats.fldr, file_name)

        if path.isfile(export_name):
            raise FileExistsError(f"{file_name} already exists. Choose a different name or matrix format")

        if mats.check_exists(matrix_name):
            raise FileExistsError(f"{matrix_name} already exists. Choose a different name")

        names = [f"{key}_{cls._id}" for cls in self.classes for key in cls._selected_links.keys()]

        kwargs = {
            "file_name": AequilibraeMatrix().random_name(),
            "zones": self.classes[0].graph.centroids.shape[0],
            "matrix_names": names,
            "memory_only": False,
        }

        # Create the matrix to manipulate
        out_skims = AequilibraeMatrix()
        out_skims.create_empty(**kwargs)

        out_skims.index[:] = self.classes[0].graph.centroids[:]

        for cls in self.classes:
            if cls._selected_links is None:
                continue

            res = cls.results.select_link_od

            for mat in res.names:
                out_skims.matrix[f"{mat}_{cls._id}"][:, :] = res.get_matrix(mat)[:, :, 0]

        out_skims.matrices.flush()  # Make sure that all data went to the disk
        out_skims.description = f"Select link matrix from procedure ID {self.procedure_id}_sl."

        out_skims.export(export_name)

    def save_select_link_results(self, name: str) -> None:
        """
        Saves both the Select Link matrices and flow results at the same time, using the same name.

        :Arguments:
            **name** (:obj:`str`): name of the matrices
        """
        self.save_select_link_flows(name)
        self.save_select_link_matrices(name)


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

        res_path = path.join(project.project_base_path, "results_database.sqlite")
        with commit_and_close(res_path, missing_ok=True) as conn:
            df.to_sql(table_name, conn)

        report = {"setup": self.info()}
        data = [table_name, "transit assignment", self.procedure_id, str(report), self.procedure_date, self.description]
        with commit_and_close(path.join(project.project_base_path, "public_transport.sqlite")) as conn:
            conn.execute(
                """Insert into results(table_name, procedure, procedure_id, procedure_report, timestamp,
                                            description) Values(?,?,?,?,?,?)""",
                data,
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
