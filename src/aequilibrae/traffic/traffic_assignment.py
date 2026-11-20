import logging
import socket
from os import path
from typing import Optional

import numpy as np
import pandas as pd
from numpy import nan_to_num

from aequilibrae._assignment import AssignmentBase
from aequilibrae.matrix.aequilibrae_matrix import AequilibraeMatrix, AequilibraeSparseMatrix
from aequilibrae.routing.network import Network
from aequilibrae.traffic.assignment_context import PreparedTrafficClass, validate_assignment_demand
from aequilibrae.traffic.linear_approximation import LinearApproximation
from aequilibrae.traffic.traffic_class import TrafficClass
from aequilibrae.traffic.vdf import VDF
from aequilibrae.utils.core_setter import (
    clamp_cores,
    resolve_cores,
    resolve_elementwise_cores,
    resolve_threading_threshold,
)


def _safe_delay_factor(congested_time, free_flow_time):
    return np.divide(
        congested_time,
        free_flow_time,
        out=np.zeros_like(congested_time),
        where=free_flow_time != 0,
    )


logger = logging.getLogger(__name__)


class TrafficAssignment(AssignmentBase):
    """Traffic assignment class.

    The constructor retains a supplied standalone Network, then checks
    compatibility. Do not mutate that Network during use.
    ``execute()`` prepares each class's routing graph and returns link results.
    Classes share the network reporting mapping, but build their own graph mappings.
    Each class owns its costs, demand, skims, and output buffers.

    For a comprehensive example on use, see the Use examples page.

    .. code-block:: python

        >>> from aequilibrae.traffic import TrafficAssignment, TrafficClass
        >>> from aequilibrae.traffic.vdf import bpr
        >>> project = create_example(project_path)
        >>> network = project.road.build_network()
        >>> proj_matrices = project.matrices

        # We will only assign one user class stored as 'matrix' inside the OMX file
        >>> demand = proj_matrices.get_matrix("demand.omx", subset=["matrix"])

        # Creates the assignment class
        >>> assigclass = TrafficClass(
        ...     name="car", mode="c", cost="free_flow_time", demand=demand,
        ...     skimming=['free_flow_time', 'distance'], graph_options={'block_centroid_flows': False},
        ... )
        >>> assig = TrafficAssignment(
        ...     network=network, project=project, classes=[assigclass], time_field='free_flow_time',
        ...     capacity_field='capacity', vdf=bpr,
        ...     name_mapping={"alpha": "b", "beta": "power"}, algorithm='bfw',
        ... )
        >>> results_df = assig.execute(max_iterations=10, rgap=0.00001)

        # If you want, it is possible to access the convergence report
        >>> convergence_report = pd.DataFrame(assig.assignment.convergence_report)

        # Assignment results can be viewed as a Pandas DataFrame
        >>> results_df = assig.results()

        # Information on the assignment setup can be recovered with
        >>> info = assig.info()

        # Or save it directly to the results database
        >>> results = assig.save_results(table_name='base_year_assignment')

        # skims are here
        >>> avg_skims = assig.class_results['car'].skims # blended ones

        >>> project.close()

    """

    all_algorithms = ["all-or-nothing", "msa", "frank-wolfe", "fw", "cfw", "bfw"]
    all_bfw_conjugacies = ["approximate", "exact"]
    _result_fields = (
        ("Preload", "Preload_tot", np.nansum),
        ("Congested_Time", "Congested_Time_Max", np.nanmax),
        ("Delay_factor", "Delay_factor_Max", np.nanmax),
        ("VOC", "VOC_max", np.nanmax),
        ("PCE", "PCE_tot", np.nansum),
    )

    def __init__(
        self,
        *,
        network: Network,
        classes: list[TrafficClass],
        time_field: str,
        capacity_field: str,
        vdf: VDF,
        algorithm: str,
        name_mapping=None,
        preload=None,
        project=None,
    ) -> None:
        """Retain standalone network data, then validate without preparing routing topology.

        ``name_mapping`` maps VDF parameter names to network fields or constants.
        Without a mapping, VDF defaults supply values.
        Preload arrays use directed links in network mapping order.
        """
        if not isinstance(network, Network):
            raise TypeError("network must be a built routing Network; call project.road.build_network() first")
        super().__init__(project=project)
        classes = tuple(classes)
        if not classes or any(not isinstance(cls, TrafficClass) for cls in classes):
            raise TypeError("classes must contain at least one TrafficClass")
        if len({cls.name for cls in classes}) != len(classes):
            raise ValueError("Traffic class names must be unique")
        if not isinstance(vdf, VDF):
            raise TypeError("vdf must be a VDF")
        for label, field in (("time_field", time_field), ("capacity_field", capacity_field)):
            if not isinstance(field, str) or not field:
                raise ValueError(f"{label} must be a nonempty string")
        if not isinstance(algorithm, str) or algorithm.lower() not in self.all_algorithms:
            raise ValueError(f"Assignment algorithm must be one of {self.all_algorithms}")
        self.network = network
        self.classes = tuple(classes)
        self.time_field = time_field
        self.capacity_field = capacity_field
        self.vdf = vdf
        self.algorithm = "frank-wolfe" if algorithm.lower() == "fw" else algorithm.lower()
        self._vdf_sources = dict(name_mapping or {})
        if self._vdf_sources.keys() - vdf.spec.keys():
            raise ValueError("name_mapping contains unknown VDF parameters")
        for name, settings in vdf.spec.items():
            if name not in self._vdf_sources:
                self._vdf_sources[name] = settings.get("fill_NA")
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
        self._config = {
            "Algorithm": self.algorithm,
            "Time field": time_field,
            "Capacity field": capacity_field,
            "VDF function": vdf.name,
            "VDF parameters": self._vdf_sources.copy(),
        }
        self._validate_network_configuration(self.network)

    @property
    def class_results(self):
        """Return results by traffic class name after execution."""
        if not self._class_states:
            raise RuntimeError("Run the assignment before requesting class results")
        return {state.configuration.name: state.results for state in self._class_states}

    def get_skim_results(self):
        return {name: results.skims for name, results in self.class_results.items()}

    def _validate_network_configuration(self, network):
        result_columns = [
            f"{name}_{direction}" for cls in self.classes for name in cls.demand for direction in ("ab", "ba", "tot")
        ]
        result_columns.extend(
            column for name, total_name, _ in self._result_fields for column in (f"{name}_AB", f"{name}_BA", total_name)
        )
        # SQLite result tables treat column names as case-insensitive.
        if len(result_columns) != len({column.casefold() for column in result_columns}):
            raise ValueError(
                "Demand matrix names must produce unique result columns across classes and network metrics"
            )
        centroids = network.centroids if network.centroids is not None else self.classes[0].demand.index
        fields = [self.time_field, self.capacity_field]
        for cls in self.classes:
            fields.extend([cls.cost, *cls.skimming])
            if cls.fixed_cost_field:
                fields.append(cls.fixed_cost_field)
        fields.extend(value for value in self._vdf_sources.values() if isinstance(value, str))
        field_arrays = network._field_arrays(fields)
        field_values = field_arrays.__getitem__
        for field in (self.time_field, self.capacity_field):
            values = field_values(field)
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(f"Field {field} must contain positive finite values")
        for cls in self.classes:
            network.validate_mode(cls.mode)
            network.validate_centroids(cls.demand.index)
            validate_assignment_demand(cls.demand, centroids)
            network._validate_turns(cls.mode)
            values = field_values(cls.cost)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"Field {cls.cost} must contain nonnegative finite values")
            for field in cls.skimming:
                if not np.isfinite(field_values(field)).all():
                    raise ValueError(f"Skim field {field} must contain finite values")
            if cls.fixed_cost_field:
                values = field_values(cls.fixed_cost_field)
                if np.isinf(values).any() or np.any(values < 0):
                    raise ValueError("Fixed costs must be nonnegative and finite where present")
                if np.isnan(values).any():
                    logger.warning(f"Cost field {cls.fixed_cost_field} has NaN values. Converted to zero")
            if cls.select_link_sets:
                available = dict(
                    network.links.loc[network.mode_mask(cls.mode), ["link_id", "direction"]].itertuples(
                        index=False, name=None
                    )
                )
                for members in cls.select_link_sets.values():
                    for link, direction in members:
                        if link not in available or (direction != 0 and available[link] == -direction):
                            raise ValueError(f"Selected link {(link, direction)} is not available for mode {cls.mode}")
        arrays = {}
        for name in self.vdf.spec:
            value = self._vdf_sources[name]
            if value is None:
                raise ValueError(f"{name} should exist in the set of parameters provided")
            if not isinstance(value, str) and np.ndim(value) != 0:
                raise ValueError(f"VDF parameter {name} requires a field name or numeric constant")
            arrays[name] = field_values(value) if isinstance(value, str) else np.array([value], dtype=float)
        self._validate_vdf_arrays(arrays)
        self._validate_network_preload(network)
        return field_arrays

    def _validate_network_preload(self, network):
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
                if values.shape != (network.mapping.link_count,):
                    raise ValueError("Preload must have one value per directed link in network mapping order")
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

        Execution options apply to this run. Source data comes from the network
        retained by the constructor. Each run owns fresh routing contexts and outputs.

        ``bfw_conjugacy`` chooses how BFW solves for its direction coefficients. Ignored by every other algorithm.

        * ``"approximate"`` (default) - appendix A of Mitradjieva & Lindberg, which assumes the two previous
          directions are already conjugate with respect to the *current* Hessian. They were made conjugate with
          respect to the previous one, so the assumption holds only in the limit; the paper notes BFW is
          therefore "correct only in the limit".

        * ``"exact"`` - solve both conjugacy conditions jointly as a 2x2 system, keeping the cross term the
          appendix drops. Costs two extra Hessian contractions per iteration, which is negligible next to the
          shortest-path build. The system is singular when the two previous directions are H-parallel, in which
          case the iteration restarts from Frank-Wolfe.

        Either way, negative coefficients are clamped to zero to keep the weights on the simplex, and that
        clamping breaks strict biconjugacy whenever it triggers.

        When the algorithm is BFW the convergence report gains ``conjugacy_prev``, ``conjugacy_prev2`` and
        ``hessian_drift`` columns to compare the two variants; see ``LinearApproximation``.

        :Arguments:
            **cores** (:obj:`int`): Number of CPU cores to use

            **threading_threshold** (:obj:`int`, `Optional`): Minimum number of array elements for elementwise
            operations to be threaded. Negative values disable threading for those operations.

            **elementwise_cores** (:obj:`int`, `Optional`): Number of CPU cores for elementwise
            (``parallel_numpy``/VDF) operations. When not provided, it is resolved from the
            ``AEQ_ELEMENTWISE_CPUS`` environment variable or ``parameters.yml``, defaulting to at most
            8 threads
        """
        defaults = self.parameters["assignment"]["equilibrium"]
        max_iterations = defaults["maximum_iterations"] if max_iterations is None else max_iterations
        rgap = defaults["rgap"] if rgap is None else rgap
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if isinstance(rgap, bool) or not isinstance(rgap, (int, float)) or not np.isfinite(rgap) or rgap < 0:
            raise ValueError("rgap must be nonnegative and finite")
        if not isinstance(bfw_conjugacy, str) or bfw_conjugacy.lower() not in self.all_bfw_conjugacies:
            raise ValueError(f"bfw_conjugacy must be one of {self.all_bfw_conjugacies}")
        if (
            isinstance(steps_below_needed_to_terminate, bool)
            or not isinstance(steps_below_needed_to_terminate, int)
            or steps_below_needed_to_terminate < 1
        ):
            raise ValueError("steps_below_needed_to_terminate must be a positive integer")
        if isinstance(cores, bool):
            raise ValueError("cores must be an integer")
        network = self.network
        fields = self._validate_network_configuration(network)
        self.max_iter = max_iterations
        self.rgap_target = float(rgap)
        self.bfw_conjugacy = bfw_conjugacy.lower()
        self.steps_below_needed_to_terminate = steps_below_needed_to_terminate
        system_parameters = self.parameters["system"]
        self.cores = resolve_cores(system_parameters) if cores is None else clamp_cores(cores)
        if threading_threshold is None:
            threading_threshold = resolve_threading_threshold(system_parameters)
        if elementwise_cores is None:
            elementwise_cores = resolve_elementwise_cores(system_parameters, self.cores)
        self._prepared_network = network
        states = []
        for cls in self.classes:
            state = PreparedTrafficClass(cls, network, fields)
            for results in (state.results, state._aon_results):
                results.set_cores(self.cores, threading_threshold, elementwise_cores)
            states.append(state)
        self._class_states = states
        first = states[0]
        self.elementwise_cores = first.results.elementwise_cores
        self.threading_threshold = first.results.threading_threshold
        # VDF memoryviews need writable buffers, separate from shared class attributes.
        self.free_flow_tt = fields[self.time_field].copy()
        self.capacity = fields[self.capacity_field].copy()
        self.total_flow = np.zeros_like(self.free_flow_tt)
        self.congested_time = self.free_flow_tt.copy()
        self.vdf_parameters = {
            name: fields[value].copy() if isinstance(value, str) else np.full_like(self.free_flow_tt, value)
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
            Defaults to the explicitly supplied constructor project
        """

        df = self.results()
        if not keep_zero_flows:
            df = df[df.PCE_tot > 0]

        project = self._resolve_project(project)

        report = {"convergence": self.assignment.convergence_report, "setup": self.info()}
        record = project.results.create(
            table_name=table_name,
            data=df,
            procedure="traffic assignment",
            procedure_id=self.procedure_id,
            procedure_report=report,
            timestamp=self.procedure_date,
            description=self.description,
        )
        return record

    def results(self) -> pd.DataFrame:
        """Prepares the assignment results as a Pandas DataFrame

        Class load columns use demand matrix names without a traffic-class prefix.
        Names must be unique across classes and shared network metrics.

        :Returns:
            **DataFrame** (:obj:`pd.DataFrame`): Pandas DataFrame with all the assignment results indexed on `link_id`
        """

        if self.assignment is None:
            raise RuntimeError("Run the assignment before requesting its results")
        assig_results = [cls.results.get_load_results() for cls in self._class_states]

        network = self._prepared_network.mapping
        tot_flow = self.assignment.total_flow
        preload = self.assignment.preload
        if preload is None:
            preload = np.zeros(network.link_count)

        # Report shared network values independently of each class's routing graph.
        values = np.column_stack(
            (
                preload,
                self.congested_time,
                _safe_delay_factor(self.congested_time, self.free_flow_tt),
                tot_flow / self.capacity,
                tot_flow,
            )
        )

        indices = network.directed_link_indices
        valid = indices >= 0
        directional = np.zeros((len(indices), 2, values.shape[1]), dtype=values.dtype)
        directional[valid] = nan_to_num(values[indices[valid]])
        ab, ba = directional[:, 0], directional[:, 1]

        columns = {}
        for index, (name, total_name, reduce) in enumerate(self._result_fields):
            columns[f"{name}_AB"] = ab[:, index]
            columns[f"{name}_BA"] = ba[:, index]
            columns[total_name] = reduce([ab[:, index], ba[:, index]], axis=0)

        assig_results.append(pd.DataFrame(columns, index=network.physical_link_ids))

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
        return {
            "Algorithm": self.algorithm,
            "Classes": {cls.name: cls.info for cls in self.classes},
            "Computer name": socket.gethostname(),
            "Maximum iterations": self.max_iter,
            "Procedure ID": self.procedure_id,
            "Target RGap": self.rgap_target,
        }

    def save_skims(self, matrix_name: str, which_ones="final", format="omx", project=None) -> list[str]:
        """Saves the skims (if any) to the project's matrix folder

        :Arguments:
            **matrix_name** (:obj:`str`): Base file name for the matrices, suffixed with each traffic class ID

            **which_ones** (:obj:`str`, *Optional*): 'final': Results of the final iteration,
                'blended': Averaged results for all iterations, 'all': Saves skims for both the final iteration and the
                blended ones. Default is 'final'

            **format** (:obj:`str`, *Optional*): File format ('omx'). Default is 'omx'

            **project** (:obj:`Project`, *Optional*): Project we want to save the results to.
                Defaults to the explicitly supplied constructor project

        :Returns:
            **file_names** (:obj:`list`): OMX file names saved for classes with skims to save.
        """
        if which_ones not in ("final", "blended", "all"):
            raise ValueError("which_ones must be final, blended or all")
        file_names = []
        mat_format = format.lower()
        if mat_format != "omx":
            raise ValueError("Only OMX matrix files are supported")

        project = self._resolve_project(project)

        mats = project.matrices

        for cls in self._class_states:
            file_name = f"{matrix_name}_{cls.configuration.name}.{mat_format}"

            export_name = mats.folder / file_name

            if path.isfile(export_name):
                raise FileExistsError(f"{file_name} already exists. Choose a different name or matrix format")

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
            out_skims = AequilibraeMatrix(index=cls.configuration.demand.index)
            out_skims.update(output_matrices)
            description = f"Skimming for assignment procedure. Class {cls.configuration.name}"
            out_skims.metadata["description"] = description

            mats.save_matrix(
                file_name=file_name,
                matrix=out_skims,
                procedure_id=self.procedure_id,
                timestamp=self.procedure_date,
                procedure="Traffic Assignment",
                description=description,
            )
            file_names.append(file_name)

        return file_names

    def select_link_flows(self) -> dict[str, pd.DataFrame]:
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
            df = df.rename(columns=cls_cols)
            class_flows.append(df)
        return pd.concat(class_flows, axis=1).rename_axis("link_id")

    def save_select_link_flows(self, table_name: str, project=None) -> None:
        """
        Saves the select link link flows for all classes into the results database.

        :Arguments:
            **table_name** (:obj:`str`): Name of the table being inserted to. Note the traffic class

            **project** (:obj:`Project`, *Optional*): Project we want to save the results to.
            Defaults to the explicitly supplied constructor project
        """

        project = self._resolve_project(project)
        df = self.select_link_flows()

        report = {}
        description = f"Select link analysis from {self.procedure_id}"
        project.results.create(
            table_name=table_name,
            data=df,
            procedure="traffic select link",
            procedure_id=f"{self.procedure_id}_sl",
            procedure_report=report,
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
            Defaults to the explicitly supplied constructor project
        """
        project = self._resolve_project(project)

        mats = project.matrices

        file_name = f"{matrix_name}.omx"

        export_name = mats.folder / file_name

        if path.isfile(export_name):
            raise FileExistsError(f"{file_name} already exists. Choose a different name or matrix format")

        selected_classes = [cls for cls in self._class_states if cls.results.select_link_od is not None]
        if not selected_classes:
            return
        centroids = selected_classes[0].configuration.demand.index
        matrices = {}
        for cls in selected_classes:
            selected = cls.results.select_link_od
            for name, values in selected.matrices.items():
                for class_name, matrix in values.items():
                    key = f"{name}_{cls._id}_{class_name}"
                    if key in matrices:
                        raise ValueError(f"Duplicate selected OD export name: {key}")
                    # Classes share the export's centroid order, so sparse
                    # coordinates do not need remapping.
                    matrices[key] = matrix.copy()
        if not matrices:
            return
        out_skims = AequilibraeSparseMatrix(index=centroids)
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
