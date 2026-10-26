import json
import logging
import socket
from abc import ABC, abstractmethod
from datetime import datetime
from os import path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import numpy as np
import pandas as pd
from numpy import nan_to_num

from aequilibrae.context import get_active_project
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.parameters import Parameters
from aequilibrae.paths.assignment_context import PreparedTrafficClass, validate_assignment_demand
from aequilibrae.paths.linear_approximation import LinearApproximation
from aequilibrae.paths.network_data import NetworkData
from aequilibrae.paths.optimal_strategies import OptimalStrategies
from aequilibrae.paths.traffic_class import TrafficClass, TransportClassBase
from aequilibrae.paths.vdf import VDF
from aequilibrae.utils.core_setter import clamp_cores, resolve_cores


def _assign_aggregation_fields(
    aggregation,
    field_ab,
    field_ba,
    graph_ab_values,
    graph_ba_values,
    network_ab_idx,
    network_ba_idx,
):
    ab = aggregation[field_ab].to_numpy(copy=True)
    ba = aggregation[field_ba].to_numpy(copy=True)
    ab[network_ab_idx] = nan_to_num(graph_ab_values)
    ba[network_ba_idx] = nan_to_num(graph_ba_values)
    aggregation[field_ab] = ab
    aggregation[field_ba] = ba


def _safe_delay_factor(congested_time, free_flow_time):
    return np.divide(
        congested_time,
        free_flow_time,
        out=np.zeros_like(congested_time),
        where=free_flow_time != 0,
    )


logger = logging.getLogger(__name__)


class AssignmentBase(ABC):
    def __init__(self, project=None):
        self.procedure_id = uuid4().hex
        self.procedure_date = str(datetime.today())

        proj = project or get_active_project(must_exist=False)
        self.project = proj

        self.parameters = proj.parameters if proj else Parameters().parameters

        self.classes: List[TrafficClass] = []
        self.algorithm: str = None
        self.time_field: str = None
        self.assignment: Union[LinearApproximation, OptimalStrategies] = None
        self.free_flow_tt: np.ndarray = None
        self.total_flow: np.ndarray = None
        self.cores: int = None
        self.elementwise_cores: int = None
        self.threading_threshold: int = None
        self._config = {}

        self.description: str = ""

    def algorithms_available(self) -> list:
        """
        Returns all algorithms available for use

        :Returns:
            :obj:`list`: Available algorithm names
        """
        return self.all_algorithms

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
        if self.assignment is None:
            raise RuntimeError("Run the assignment before requesting its report")
        return pd.DataFrame(self.assignment.convergence_report)

    @abstractmethod
    def info(self) -> dict:
        pass

    def get_skim_results(self) -> dict:
        """Return assignment skims by class name."""
        return {cls._id: cls.results.skims for cls in self.classes}


class TrafficAssignment(AssignmentBase):
    """Traffic assignment class.

    The constructor validates the network fields, modes, demand, and VDF parameters.
    ``execute()`` prepares compact routing contexts and returns link results.
    Classes with matching routing policies share graph and reporting mappings.
    Each class owns its costs, demand, skims, and output buffers.

    .. code-block:: python

        >>> from aequilibrae.paths import TrafficAssignment, TrafficClass
        >>> from aequilibrae.paths.vdf import bpr
        >>> project = create_example(project_path)
        >>> network = project.network.build(modes=["c"])
        >>> demand = project.matrices.get_matrix("demand_omx", subset=["matrix"])
        >>> cars = TrafficClass(
        ...     mode="c", cost="free_flow_time", demand=demand,
        ...     skimming=["free_flow_time", "distance"],
        ...     block_centroid_flows=False,
        ... )
        >>> assignment = TrafficAssignment(
        ...     network=network, classes=[cars], time_field="free_flow_time",
        ...     capacity_field="capacity", vdf=bpr,
        ...     name_mapping={"alpha": "b", "beta": "power"}, algorithm="bfw",
        ... )
        >>> results = assignment.execute(max_iterations=10, rgap=1e-5, cores=1)
        >>> report = assignment.report()
        >>> project.close()

    """

    all_algorithms = ["all-or-nothing", "msa", "frank-wolfe", "cfw", "bfw"]
    all_bfw_conjugacies = ["approximate", "exact"]

    def __init__(
        self,
        *,
        network: NetworkData,
        classes: list[TrafficClass],
        time_field: str,
        capacity_field: str,
        vdf: VDF,
        algorithm: str,
        name_mapping=None,
        preload=None,
        project=None,
    ) -> None:
        """Validate a complete assignment configuration without preparing graphs.

        ``name_mapping`` maps VDF parameter names to network fields or constants.
        Without a mapping, matching network fields or VDF defaults supply values.
        Preload arrays use directed links in ascending link-ID and direction order.
        """
        super().__init__(project=project)
        if not isinstance(network, NetworkData):
            raise TypeError("network must be a NetworkData")
        classes = tuple(classes)
        if not classes or any(not isinstance(cls, TrafficClass) for cls in classes):
            raise TypeError("classes must contain at least one TrafficClass")
        if len({cls.name for cls in classes}) != len(classes):
            raise ValueError("Traffic class names must be unique")
        if not isinstance(vdf, VDF):
            raise TypeError("vdf must be a VDF")
        if not isinstance(algorithm, str) or algorithm.lower() not in self.all_algorithms:
            raise ValueError(f"Assignment algorithm must be one of {self.all_algorithms}")
        self.network = network
        self.classes = tuple(classes)
        self.time_field = time_field
        self.capacity_field = capacity_field
        self.vdf = vdf
        self.algorithm = algorithm.lower()
        self._vdf_sources = dict(name_mapping or {})
        if self._vdf_sources.keys() - vdf.spec.keys():
            raise ValueError("name_mapping contains unknown VDF parameters")
        for name, settings in vdf.spec.items():
            if name not in self._vdf_sources:
                self._vdf_sources[name] = name if name in network.fields else settings.get("fill_NA")
        self.preload = (
            None
            if preload is None
            else (preload.copy() if isinstance(preload, pd.DataFrame) else np.array(preload, dtype=float, copy=True))
        )
        self._class_states = []
        self.vdf_parameters = None
        self.capacity = None
        self.congested_time = None
        defaults = self.parameters["assignment"]["equilibrium"]
        self.max_iter = defaults["maximum_iterations"]
        self.rgap_target = defaults["rgap"]
        self.bfw_conjugacy = "approximate"
        self.steps_below_needed_to_terminate = 1
        self._validate_network_configuration()
        self._config = {
            "Algorithm": self.algorithm,
            "Time field": time_field,
            "Capacity field": capacity_field,
            "VDF function": vdf.name,
            "VDF parameters": self._vdf_sources.copy(),
        }

    @property
    def class_results(self):
        """Return results by traffic class name after execution."""
        if not self._class_states:
            raise RuntimeError("Run the assignment before requesting class results")
        return {state.configuration.name: state.results for state in self._class_states}

    def get_skim_results(self):
        return {name: results.skims for name, results in self.class_results.items()}

    def _validate_network_configuration(self):
        network = self.network
        for field in (self.time_field, self.capacity_field):
            values = network.field_values(field)
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(f"Field {field} must contain positive finite values")
        for cls in self.classes:
            network.validate_mode(cls.mode)
            network.validate_centroids(cls.demand.index)
            validate_assignment_demand(cls.demand, cls.demand.index)
            for field in [cls.cost, *cls.skimming]:
                values = network.field_values(field)
                if not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError(f"Field {field} must contain nonnegative finite values")
            if cls.fixed_cost_field:
                values = network.field_values(cls.fixed_cost_field)
                if not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError("Fixed costs must be nonnegative and finite")
            for members in cls.select_link_sets.values():
                for link, direction in members:
                    rows = network.links[network.links.link_id == link]
                    if (
                        rows.empty
                        or not rows.modes.fillna("").astype(str).str.contains(cls.mode, regex=False).all()
                        or (direction != 0 and (rows.direction == -direction).any())
                    ):
                        raise ValueError(f"Selected link {(link, direction)} is not available for mode {cls.mode}")
        arrays = {}
        for name in self.vdf.spec:
            value = self._vdf_sources[name]
            if value is None:
                raise ValueError(f"{name} should exist in the set of parameters provided")
            if not isinstance(value, str) and np.ndim(value) != 0:
                raise ValueError(f"VDF parameter {name} requires a field name or numeric constant")
            arrays[name] = network.field_values(value) if isinstance(value, str) else np.array([value], dtype=float)
        self._validate_vdf_arrays(arrays)
        self._validate_network_preload()

    def _validate_network_preload(self):
        network = self.network
        preload = self.preload
        if preload is not None:
            if isinstance(preload, pd.DataFrame):
                if set(preload.columns) != {"link_id", "direction", "preload"} or preload.empty:
                    raise ValueError("Preload requires link_id, direction and preload columns")
                if preload.duplicated(["link_id", "direction"]).any():
                    raise ValueError("Preload link and direction pairs must be unique")
                values = preload.preload.to_numpy(dtype=float)
                for row in preload.itertuples():
                    links = network.links[network.links.link_id == row.link_id]
                    if row.direction not in (-1, 1) or links.empty or (links.direction == -row.direction).any():
                        raise ValueError("Preload contains an invalid link or direction")
            else:
                values = np.asarray(preload, dtype=float)
                count = sum((network.links.direction != -d).sum() for d in (-1, 1))
                if values.shape != (count,):
                    raise ValueError("Preload must have one value per directed link, ordered by link ID and direction")
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Preload must be nonnegative and finite")

    def execute(
        self,
        *,
        max_iterations=None,
        rgap=None,
        cores=None,
        threading_threshold=None,
        elementwise_cores=None,
        bfw_conjugacy="approximate",
        steps_below_needed_to_terminate=1,
        log_specification=True,
    ) -> pd.DataFrame:
        """Prepare graphs, run assignment, and return link results.

        Execution options apply to this run. Each run owns fresh graphs and outputs.
        """
        defaults = self.parameters["assignment"]["equilibrium"]
        max_iterations = defaults["maximum_iterations"] if max_iterations is None else max_iterations
        rgap = defaults["rgap"] if rgap is None else rgap
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if isinstance(rgap, bool) or not isinstance(rgap, (int, float)) or not np.isfinite(rgap) or rgap < 0:
            raise ValueError("rgap must be nonnegative and finite")
        if bfw_conjugacy not in self.all_bfw_conjugacies:
            raise ValueError(f"bfw_conjugacy must be one of {self.all_bfw_conjugacies}")
        if (
            isinstance(steps_below_needed_to_terminate, bool)
            or not isinstance(steps_below_needed_to_terminate, int)
            or steps_below_needed_to_terminate < 1
        ):
            raise ValueError("steps_below_needed_to_terminate must be a positive integer")
        if isinstance(cores, bool):
            raise ValueError("cores must be an integer")
        self._validate_network_configuration()
        self.max_iter = max_iterations
        self.rgap_target = float(rgap)
        self.bfw_conjugacy = bfw_conjugacy
        self.steps_below_needed_to_terminate = steps_below_needed_to_terminate
        self.cores = resolve_cores(self.parameters["system"]) if cores is None else clamp_cores(cores)
        network = self.network._prepare()
        self._prepared_network = network
        states = []
        for cls in self.classes:
            state = PreparedTrafficClass(cls, network)
            for results in (state.results, state._aon_results):
                results.set_cores(self.cores, threading_threshold, elementwise_cores)
            states.append(state)
        self._class_states = states
        first = states[0]
        self.elementwise_cores = first.results.elementwise_cores
        self.threading_threshold = first.results.threading_threshold
        self.free_flow_tt = self._network_field(self.time_field)
        self.capacity = self._network_field(self.capacity_field)
        self.total_flow = np.zeros_like(self.free_flow_tt)
        self.congested_time = self.free_flow_tt.copy()
        self.vdf_parameters = {
            name: self._network_field(value) if isinstance(value, str) else np.full_like(self.free_flow_tt, value)
            for name, value in self._vdf_sources.items()
        }
        self.assignment = LinearApproximation(self, self.algorithm, project=self.project)
        self._config.update(
            {
                "Maximum iterations": self.max_iter,
                "Target RGAP": self.rgap_target,
                "BFW conjugacy": self.bfw_conjugacy,
                "Number of cores": self.cores,
            }
        )
        super().execute(log_specification)
        self.total_flow = self.assignment.total_flow.copy()
        return self.results()

    def _network_field(self, field):
        # VDF callables receive writable memoryviews. Keep their buffers separate
        # from the immutable attributes shared by class routing inputs.
        return self._prepared_network.field_values(field).copy()

    def _prepare_preload(self):
        if self.preload is None:
            return None
        if not isinstance(self.preload, pd.DataFrame):
            return self.preload.copy()
        network = self._prepared_network
        directed = pd.DataFrame({"link_id": network.link_ids, "direction": network.directions})
        return directed.merge(self.preload, on=["link_id", "direction"], how="left")["preload"].fillna(0).to_numpy()

    def _validate_vdf_arrays(self, vdf_link_attributes):
        for parameter_name, parameter_data in self.vdf.spec.items():
            array = vdf_link_attributes[parameter_name]
            if not np.isfinite(array).all():
                raise ValueError(f"At least one {parameter_name} is not finite")
            if "bounds" not in parameter_data:
                continue
            minimum, maximum = parameter_data["bounds"]

            if parameter_data.get("inclusive_lower", True):
                if array.min() < minimum:
                    raise ValueError(f"At least one {parameter_name} is less than {minimum}")
            else:
                if array.min() <= minimum:
                    raise ValueError(f"At least one {parameter_name} is less than or equal to {minimum}")

            if parameter_data.get("inclusive_upper", True):
                if array.max() > maximum:
                    raise ValueError(f"At least one {parameter_name} is greater than {maximum}")
            else:
                if array.max() >= maximum:
                    raise ValueError(f"At least one {parameter_name} is greater than or equal to {maximum}")

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
        for assig_class in self._class_states:
            data[assig_class.configuration.name] = assig_class.skim_congested(skim_fields)
        if return_matrices:
            return data

    def log_specification(self):
        logger.info("Traffic Class specification")
        for cls in self._class_states:
            logger.info(str(cls.configuration.info))

        logger.info("Traffic Assignment specification")
        logger.info(self._config)

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

        report = {"convergence": self.assignment.convergence_report, "setup": self.info()}
        record = project.results.create(
            table_name=table_name,
            data=df,
            procedure="traffic assignment",
            procedure_id=self.procedure_id,
            procedure_report=json.dumps(report),
            timestamp=self.procedure_date,
            description=self.description,
        )
        return record

    def results(self) -> pd.DataFrame:
        """Prepares the assignment results as a Pandas DataFrame

        :Returns:
            **DataFrame** (:obj:`pd.DataFrame`): Pandas DataFrame with all the assignment results indexed on `link_id`
        """

        if self.assignment is None:
            raise RuntimeError("Run the assignment before requesting its results")
        assig_results = [cls.results.get_load_results() for cls in self._class_states]

        class1 = self._class_states[0]
        res1 = assig_results[0]

        tot_flow = self.assignment.total_flow
        capacity = self.capacity
        voc = tot_flow / capacity
        congested_time = self.congested_time
        free_flow_tt = self.free_flow_tt
        if self.assignment.preload is None:
            preload = np.full(len(tot_flow), np.nan)
        else:
            preload = self.assignment.preload

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
        graph_ab_idx, graph_ba_idx = m.graph_ab_idx, m.graph_ba_idx

        _assign_aggregation_fields(
            agg,
            "Preload_AB",
            "Preload_BA",
            preload[m.graph_ab_idx],
            preload[m.graph_ba_idx],
            m.network_ab_idx,
            m.network_ba_idx,
        )
        agg["Preload_tot"] = np.nansum([agg.Preload_AB, agg.Preload_BA], axis=0)

        _assign_aggregation_fields(
            agg,
            "Congested_Time_AB",
            "Congested_Time_BA",
            congested_time[m.graph_ab_idx],
            congested_time[m.graph_ba_idx],
            m.network_ab_idx,
            m.network_ba_idx,
        )
        agg["Congested_Time_Max"] = np.nanmax([agg.Congested_Time_AB, agg.Congested_Time_BA], axis=0)

        delay_factor_ab = _safe_delay_factor(congested_time[graph_ab_idx], free_flow_tt[graph_ab_idx])
        delay_factor_ba = _safe_delay_factor(congested_time[graph_ba_idx], free_flow_tt[graph_ba_idx])
        _assign_aggregation_fields(
            agg,
            "Delay_factor_AB",
            "Delay_factor_BA",
            delay_factor_ab,
            delay_factor_ba,
            m.network_ab_idx,
            m.network_ba_idx,
        )
        agg["Delay_factor_Max"] = np.nanmax([agg.Delay_factor_AB, agg.Delay_factor_BA], axis=0)

        _assign_aggregation_fields(
            agg,
            "VOC_AB",
            "VOC_BA",
            voc[m.graph_ab_idx],
            voc[m.graph_ba_idx],
            m.network_ab_idx,
            m.network_ba_idx,
        )
        agg["VOC_max"] = np.nanmax([agg.VOC_AB, agg.VOC_BA], axis=0)

        _assign_aggregation_fields(
            agg,
            "PCE_AB",
            "PCE_BA",
            tot_flow[m.graph_ab_idx],
            tot_flow[m.graph_ba_idx],
            m.network_ab_idx,
            m.network_ba_idx,
        )
        agg["PCE_tot"] = np.nansum([agg.PCE_AB, agg.PCE_BA], axis=0)

        assig_results.append(agg)
        return pd.concat(assig_results, axis=1).rename_axis("link_id")

    def info(self) -> dict:
        """Return the assignment configuration and execution limits."""
        return {
            "Algorithm": self.algorithm,
            "Classes": {cls.name: cls.info for cls in self.classes},
            "Computer name": socket.gethostname(),
            "Maximum iterations": self.max_iter,
            "Procedure ID": self.procedure_id,
            "Target RGap": self.rgap_target,
        }

    def save_skims(self, matrix_name: str, which_ones="final", format="omx", project=None) -> List[Any]:
        """Saves the skims (if any) to the skim folder and registers in the matrix list

        :Arguments:
            **name** (:obj:`str`): Name of the matrix record to hold this matrix (same name used for file name)

            **which_ones** (:obj:`str`, *Optional*): 'final': Results of the final iteration,
                'blended': Averaged results for all iterations, 'all': Saves skims for both the final iteration and the
                blended ones. Default is 'final'

            **format** (:obj:`str`, *Optional*): File format ('omx'). Default is 'omx'

            **project** (:obj:`Project`, *Optional*): Project we want to save the results to.
                Defaults to the active project

        :Returns:
            **records** (:obj:`list`): Matrix records created for classes with skims to save.
        """
        if which_ones not in ("final", "blended", "all"):
            raise ValueError("which_ones must be final, blended or all")
        records = []
        mat_format = format.lower()
        if mat_format != "omx":
            raise ValueError("Only OMX matrix files are supported")

        if not project:
            project = self.project or get_active_project()

        mats = project.matrices

        for cls in self._class_states:
            file_name = f"{matrix_name}_{cls.configuration.name}.{mat_format}"

            export_name = mats.folder / file_name

            if path.isfile(export_name):
                raise FileExistsError(f"{file_name} already exists. Choose a different name or matrix format")

            if mats.get(matrix_name, default=None) is not None:
                raise FileExistsError(f"{matrix_name} already exists. Choose a different name")

            avg_skims = {} if cls.results.skims is None else cls.results.skims.matrices
            final = cls.congested_skims if cls.congested_skims is not None else cls._aon_results.skims
            last_skims = {} if final is None else final.matrices

            names = []
            if which_ones in ["final", "all"]:
                for core in last_skims:
                    names.append(f"{core}_final")

            if which_ones in ["blended", "all"]:
                for core in avg_skims:
                    names.append(f"{core}_blended")

            if not names:
                continue
            # Assembling a single final skim file can be done like this
            # We will want only the time for the last iteration and the distance averaged out for all iterations
            output_matrices = {}
            if which_ones in ["final", "all"]:
                output_matrices.update({f"{core}_final": values for core, values in last_skims.items()})
            if which_ones in ["blended", "all"]:
                output_matrices.update({f"{core}_blended": values for core, values in avg_skims.items()})
            out_skims = AequilibraeMatrix(index=cls.graph.mapping.centroids)
            out_skims.update(output_matrices)
            description = f"Skimming for assignment procedure. Class {cls.configuration.name}"
            out_skims.metadata["description"] = description

            record = mats.create(
                name=f"{matrix_name}_{cls.configuration.name}",
                file_name=file_name,
                matrix=out_skims,
                procedure_id=self.procedure_id,
                timestamp=self.procedure_date,
                procedure="Traffic Assignment",
                description=description,
            )
            records.append(record)

        return records

    def select_link_flows(self) -> Dict[str, pd.DataFrame]:
        """
        Returns a dataframe of the select link flows for each class
        """
        class_flows = []  # stores the df for each class
        for cls in self._class_states:
            # Save OD_matrices
            if cls._selected_links is None:
                continue
            # Create Values table
            df = cls.results.get_sl_results()
            # Remap the dataframe names to add the prefix classname for each class
            cls_cols = {x: cls.configuration.name + "_" + x if (x != "index") else "link_id" for x in df.columns}
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

        report = {}
        description = f"Select link analysis from {self.procedure_id}"
        project.results.create(
            table_name=table_name,
            data=df,
            procedure="traffic select link",
            procedure_id=f"{self.procedure_id}_sl",
            procedure_report=json.dumps(report),
            timestamp=self.procedure_date,
            description=description,
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

        export_name = mats.folder / file_name

        if path.isfile(export_name):
            raise FileExistsError(f"{file_name} already exists. Choose a different name or matrix format")

        if mats.get(matrix_name, default=False):
            raise FileExistsError(f"{matrix_name} already exists. Choose a different name")

        selected_classes = [cls for cls in self._class_states if cls.results.select_link_od is not None]
        if not selected_classes:
            return
        centroids = np.unique(np.concatenate([cls.graph.mapping.centroids for cls in selected_classes]))
        matrices = {}
        for cls in selected_classes:
            source_centroids = cls.graph.mapping.centroids
            positions = np.searchsorted(centroids, source_centroids)
            for name, values in cls.results.select_link_od.matrices.items():
                for column, class_name in enumerate(cls.results.classes["names"]):
                    key = f"{name}_{cls.configuration.name}_{class_name}"
                    if key in matrices:
                        raise ValueError(f"Duplicate selected OD export name: {key}")
                    if np.array_equal(source_centroids, centroids):
                        matrices[key] = values[:, :, column]
                    else:
                        matrix = np.zeros((len(centroids), len(centroids)), dtype=np.float64)
                        matrix[np.ix_(positions, positions)] = values[:, :, column]
                        matrices[key] = matrix
        if not matrices:
            return
        out_skims = AequilibraeMatrix(index=centroids)
        out_skims.update(matrices)
        out_skims.metadata["description"] = f"Select link matrix from procedure ID {self.procedure_id}_sl."
        out_skims.save_as_omx(export_name)

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
        if "Skimming Fields" in self._config:
            for transport_class in self.classes:
                transport_class.graph.set_skimming_fields(self._config["Skimming Fields"])

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
        if "Skimming Fields" in self._config:
            transport_class.graph.set_skimming_fields(self._config["Skimming Fields"])

    def _check_field(self, field: str, allow_zeros=False) -> None:
        """Throws exception if field is invalid."""
        if not self.classes:
            raise ValueError("You need add at least one transport class first")

        for c in self.classes:
            if field not in c.graph.graph.columns:
                raise ValueError(f"'{field}' not in graph for '{c._id}'")

            try:
                c.graph.graph[field] = pd.to_numeric(c.graph.graph[field], errors="raise").astype(np.float64)
                values = c.graph.graph[field].to_numpy(dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Field {field} must contain numeric values for '{c._id}'") from exc

            if np.any(pd.isna(values)):
                raise ValueError(f"At least one link for {field} is NaN for '{c._id}'")

            if values.min() <= 0 and not allow_zeros:
                raise ValueError(f"There is at least one link with zero or negative {field} for '{c._id}'")

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

            uclass["matrix_totals"] = {name: np.sum(cls.matrix[name]) for name in cls.matrix}
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
            Defaults to the active project
        """

        df = self.results()
        if not keep_zero_flows:
            df = df[df.volume > 0]

        if not project:
            project = project or get_active_project()

        report = {"setup": self.info()}
        project.results.create(
            table_name=table_name,
            data=df,
            procedure="transit assignment",
            procedure_id=self.procedure_id,
            procedure_report=json.dumps(report),
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
        self._check_field(time_field)
        c = self.classes[0]
        self.free_flow_tt = np.zeros(c.graph.graph.shape[0], "float64")
        self.free_flow_tt[c.graph.graph.__supernet_id__] = c.graph.graph[time_field]
        self.total_flow = np.zeros(self.free_flow_tt.shape[0], np.float64)
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

        fields = list(skimming_fields or [])
        self._config["Skimming Fields"] = fields
        for transport_class in self.classes:
            transport_class.graph.set_skimming_fields(fields)
