from os import environ, path
import importlib.util as iutil
from typing import List
from uuid import uuid4
import sqlite3
from datetime import datetime
import socket
import numpy as np
import pandas as pd
from aequilibrae.project.database_connection import ENVIRON_VAR
from aequilibrae.paths.all_or_nothing import allOrNothing
from aequilibrae.paths.linear_approximation import LinearApproximation
from aequilibrae.paths.vdf import VDF, all_vdf_functions
from aequilibrae.paths.traffic_class import TrafficClass
from aequilibrae.matrix import AequilibraeData
from aequilibrae.project.database_connection import database_connection
from aequilibrae import Parameters
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.project.data import Matrices

spec = iutil.find_spec("openmatrix")
has_omx = spec is not None


class TrafficAssignment(object):
    """Traffic assignment class

    For a comprehensive example on use, see the Use examples page.
    ::

        from os.path import join
        from aequilibrae.matrix import AequilibraeMatrix
        from aequilibrae.paths import TrafficAssignment, TrafficClass


        fldr = 'D:/release/Sample models/sioux_falls_2020_02_15'
        proj_name = 'SiouxFalls.sqlite'
        dt_fldr = '0_tntp_data'
        prj_fldr = '1_project'

        demand = AequilibraeMatrix()
        demand.load(join(fldr, dt_fldr, 'demand.omx'))
        demand.computational_view(['matrix']) # We will only assign one user class stored as 'matrix' inside the OMX file

        project = Project()
        project.load(join(fldr, prj_fldr))
        project.network.build_graphs()

        graph = project.network.graphs['c'] # we grab the graph for cars
        graph.set_graph('free_flow_time') # let's say we want to minimize time
        graph.set_skimming(['free_flow_time', 'distance']) # And will skim time and distance
        graph.set_blocked_centroid_flows(True)

        # Creates the assignment class
        assigclass = TrafficClass(graph, demand)

        assig = TrafficAssignment()
        # The first thing to do is to add at list of traffic classes to be assigned
        assig.set_classes([assigclass])

        assig.set_vdf("BPR")  # This is not case-sensitive # Then we set the volume delay function

        assig.set_vdf_parameters({"alpha": "b", "beta": "power"}) # And its parameters

        assig.set_capacity_field("capacity") # The capacity and free flow travel times as they exist in the graph
        assig.set_time_field("free_flow_time")

        # And the algorithm we want to use to assign
        assig.set_algorithm('bfw')

        # since I haven't checked the parameters file, let's make sure convergence criteria is good
        assig.max_iter = 1000
        assig.rgap_target = 0.00001

        assig.execute() # we then execute the assignment

        # Convergence report is here
        import pandas as pd
        convergence_report = pd.DataFrame(assig.assignment.convergence_report)
        convergence_report.head()

        # Assignment results can be viewed as a Pandas DataFrame
        results_df = assig.results()

        # information on the assignment setup can be recovered with
        info = assig.info()

        # Or save it directly to the results database
        results = assig.save_results(table_name='example_from_the_documentation')

        # skims are here
        avg_skims = assigclass.results.skims # blended ones
        last_skims = assigclass._aon_results.skims # those for the last iteration
    """

    bpr_parameters = ["alpha", "beta"]
    all_algorithms = ["all-or-nothing", "msa", "frank-wolfe", "fw", "cfw", "bfw"]

    def __init__(self) -> None:
        parameters = Parameters().parameters["assignment"]["equilibrium"]
        self.__dict__["rgap_target"] = parameters["rgap"]
        self.__dict__["max_iter"] = parameters["maximum_iterations"]
        self.__dict__["vdf"] = VDF()
        self.__dict__["classes"] = []  # type: List[TrafficClass]
        self.__dict__["algorithm"] = None  # type: str
        self.__dict__["vdf_parameters"] = None  # type: list
        self.__dict__["time_field"] = None  # type: str
        self.__dict__["capacity_field"] = None  # type: str
        self.__dict__["assignment"] = None  # type: LinearApproximation
        self.__dict__["capacity"] = None  # type: np.ndarray
        self.__dict__["free_flow_tt"] = None  # type: np.ndarray
        self.__dict__["total_flow"] = None  # type: np.ndarray
        self.__dict__["congested_time"] = None  # type: np.ndarray
        self.__dict__["cores"] = None  # type: int
        self.__dict__["save_path_files"] = False  # type: bool

        self.__dict__["procedure_id"] = uuid4().hex
        self.__dict__["description"] = ""
        self.__dict__["procedure_date"] = str(datetime.today())
        self.__dict__["steps_below_needed_to_terminate"] = 1

    def __setattr__(self, instance, value) -> None:

        check, value, message = self.__check_attributes(instance, value)
        if check:
            self.__dict__[instance] = value
        else:
            raise ValueError(message)

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
        elif instance in ["time_field", "capacity_field"]:
            if not isinstance(value, str):
                return False, value, f"Value for {instance} is not string"
        elif instance == "cores":
            if not isinstance(value, int):
                return False, value, f"Value for {instance} is not integer"
        elif instance == "save_path_files":
            if not isinstance(value, bool):
                return False, value, f"Value for {instance} is not boolean"
        if instance not in self.__dict__:
            return False, value, f"trafficAssignment class does not have property {instance}"
        return True, value, ""

    def set_vdf(self, vdf_function: str) -> None:
        """
        Sets the Volume-delay function to be used

        Args:
            vdf_function(:obj:`str`:) Name of the VDF to be used
        """
        self.vdf = vdf_function

    def set_classes(self, classes: List[TrafficClass]) -> None:
        """
        Sets Traffic classes to be assigned

        Args:
            classes (:obj:`List[TrafficClass]`:) List of Traffic classes for assignment
        """

        ids = set([x.__id__ for x in classes])
        if len(ids) < len(classes):
            raise Exception("Classes need to be unique. Your list of classes has repeated items/IDs")
        self.classes = classes  # type: List[TrafficClass]

    def add_class(self, traffic_class: TrafficClass) -> None:
        """
        Adds a traffic class to the assignment

        Args:
            traffic_class (:obj:`TrafficClass`:) Traffic class
        """

        ids = [x.__id__ for x in self.classes if x.__id__ == traffic_class.__id__]
        if len(ids) > 0:
            raise Exception("Traffic class already in the assignment")

        self.classes.append(traffic_class)

    def algorithms_available(self) -> list:
        """
        Returns all algorithms available for use

        Returns:
            :obj:`list`: List of string values to be used with **set_algorithm**
        """
        return self.all_algorithms

    # TODO: Create procedure to check that travel times, capacities and vdf parameters are equal across all graphs
    # TODO: We also need procedures to check that all graphs are compatible (i.e. originated from the same network)
    def set_algorithm(self, algorithm: str):
        """
        Chooses the assignment algorithm. e.g. 'frank-wolfe', 'bfw', 'msa'

        'fw' is also accepted as an alternative to 'frank-wolfe'

        Args:
            algorithm (:obj:`list`): Algorithm to be used
        """

        # First we instantiate the arrays we will be using over and over

        algo_dict = {i: i for i in self.all_algorithms}
        algo_dict["fw"] = "frank-wolfe"
        algo = algo_dict.get(algorithm.lower())

        if algo is None:
            raise AttributeError(f"Assignment algorithm not available. Choose from: {','.join(self.all_algorithms)}")

        if algo in ["all-or-nothing", "msa", "frank-wolfe", "cfw", "bfw"]:
            self.assignment = LinearApproximation(self, algo)
        else:
            raise Exception("Algorithm not listed in the case selection")

        self.__dict__["algorithm"] = algo

    def set_vdf_parameters(self, par: dict) -> None:
        """
        Sets the parameters for the Volume-delay function.

        Parameter values can be scalars (same values for the entire network) or network field names
        (link-specific values) - Examples: {'alpha': 0.15, 'beta': 4.0} or  {'alpha': 'alpha', 'beta': 'beta'}

        Args:
            par (:obj:`dict`): Dictionary with all parameters for the chosen VDF

        """
        if self.classes is None or self.vdf.function.lower() not in all_vdf_functions:
            raise Exception("Before setting vdf parameters, you need to set traffic classes and choose a VDF function")
        self.__dict__["vdf_parameters"] = par
        pars = []
        if self.vdf.function in ["BPR", "CONICAL"]:
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

    def set_cores(self, cores: int) -> None:
        """Allows one to set the number of cores to be used AFTER traffic classes have been added

            Inherited from :obj:`AssignmentResults`

        Args:
            cores (:obj:`int`): Number of CPU cores to use
        """
        if not self.classes:
            raise Exception("You need load traffic classes before overwriting the number of cores")

        self.cores = cores
        for c in self.classes:
            c.results.set_cores(cores)
            c._aon_results.set_cores(cores)

    def set_save_path_files(self, save_it: bool) -> None:
        """Turn path saving on or off.

        Args:
            save_it (:obj:`bool`): Boolean to indicate whether paths should be saved
        """
        if self.classes is None:
            raise Exception("You need to set traffic classes before turning path saving on or off")

        # self.save_path_files = save_it
        for c in self.classes:
            c._aon_results.save_path_file = save_it

    def set_path_file_format(self, file_format: str) -> None:
        """Specify path saving format. Either parquet wiht compression gzip or feather.

        Args:
            file_format (:obj:`str`): Name of file format to use for path files
        """
        if self.classes is None:
            raise Exception("You need to set traffic classes before specifying path saving options")

        if file_format == "feather":
            for c in self.classes:
                c._aon_results.write_feather = True
        elif file_format == "parquet":
            for c in self.classes:
                c._aon_results.write_feather = False
        else:
            raise Exception(f"Unsupported path file format {file_format} - only feather or parquet available.")

    def set_time_field(self, time_field: str) -> None:
        """
        Sets the graph field that contains free flow travel time -> e.g. 'fftime'

        Args:
            time_field (:obj:`str`): Field name
        """

        if not self.classes:
            raise ValueError("Your need add at least one traffic classes first")

        c = self.classes[0]
        if time_field not in c.graph.graph.columns:
            raise ValueError("Field not in graph")

        if np.any(np.isnan(c.graph.graph[time_field].values)):
            raise ValueError("At least one link free-flow time is NaN")

        if c.graph.graph[time_field].values.min() <= 0:
            raise ValueError("There is at least one link with zero or negative free-flow time")

        self.__dict__["free_flow_tt"] = np.zeros(c.graph.graph.shape[0], c.graph.default_types("float"))
        self.__dict__["free_flow_tt"][c.graph.graph.__supernet_id__] = c.graph.graph[time_field]
        self.__dict__["congested_time"] = np.array(self.free_flow_tt, copy=True)
        self.__dict__["total_flow"] = np.zeros(self.free_flow_tt.shape[0], np.float64)
        self.time_field = time_field

    def set_capacity_field(self, capacity_field: str) -> None:
        """
        Sets the graph field that contains link capacity for the assignment period -> e.g. 'capacity1h'

        Args:
            capacity_field (:obj:`str`): Field name
        """

        if not self.classes:
            raise ValueError("Your need add at least one traffic classes first")

        c = self.classes[0]
        if capacity_field not in c.graph.graph.columns:
            raise ValueError("Field not in graph")

        if np.any(np.isnan(c.graph.graph[capacity_field].values)):
            raise ValueError("At least one link capacity is NaN")

        if c.graph.graph[capacity_field].values.min() <= 0:
            raise ValueError("There is at least one link with zero or negative capacity")

        self.__dict__["cores"] = c.results.cores
        self.__dict__["capacity"] = np.zeros(c.graph.graph.shape[0], c.graph.default_types("float"))
        self.__dict__["capacity"][c.graph.graph.__supernet_id__] = c.graph.graph[capacity_field]
        self.capacity_field = capacity_field

    # TODO: This function actually needs to return a human-readable dictionary, and not one with
    #       tons of classes. Feeds into the class above
    # def load_assignment_spec(self, specs: dict) -> None:
    #     pass
    # def get_spec(self) -> dict:
    #     """Gets the entire specification of the assignment"""
    #     return deepcopy(self.__dict__)

    def __validate_parameters(self, kwargs) -> bool:
        if self.vdf == "":
            raise ValueError("First you need to set the Volume-Delay Function to use")

        par = list(kwargs.keys())
        if self.vdf.function == "BPR":
            q = [x for x in par if x not in self.bpr_parameters] + [x for x in self.bpr_parameters if x not in par]
        if len(q) > 0:
            raise ValueError("List of functions {} for vdf {} has an inadequate set of parameters".format(q, self.vdf))
        return True

    def execute(self) -> None:
        """Processes assignment"""
        self.assignment.execute()

    def save_results(self, table_name: str) -> None:
        """Saves the assignment results to results_database.sqlite

        Method fails if table exists

        Args:
            table_name (:obj:`str`): Name of the table to hold this assignment result
        """
        df = self.results()
        conn = sqlite3.connect(path.join(environ[ENVIRON_VAR], "results_database.sqlite"))
        df.to_sql(table_name, conn)
        conn.close()

        conn = database_connection()
        report = {"convergence": str(self.assignment.convergence_report), "setup": str(self.info())}
        data = [table_name, "traffic assignment", self.procedure_id, str(report), self.procedure_date, self.description]
        conn.execute(
            """Insert into results(table_name, procedure, procedure_id, procedure_report, timestamp,
                                            description) Values(?,?,?,?,?,?)""",
            data,
        )
        conn.commit()
        conn.close()

    def results(self) -> pd.DataFrame:
        """Prepares the assignment results as a Pandas DataFrame

        Returns:
            *DataFrame* (:obj:`pd.DataFrame`): Pandas dataframe with all the assignment results indexed on link_id
        """

        idx = self.classes[0].graph.graph.__supernet_id__
        assig_results = [cls.results.get_load_results() for cls in self.classes]

        class1 = self.classes[0]
        res1 = assig_results[0]

        tot_flow = self.assignment.fw_total_flow[idx]
        voc = self.assignment.fw_total_flow / self.capacity[idx]
        congested_time = self.congested_time[idx]
        free_flow_tt = self.free_flow_tt[idx]

        entries = res1.data.shape[0]
        fields = [
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

        types = [np.float64] * len(fields)
        agg = AequilibraeData()
        agg.create_empty(memory_mode=True, entries=entries, field_names=fields, data_types=types)
        agg.data.fill(np.nan)
        agg.index[:] = res1.data.index[:]

        link_ids = class1.results.lids
        ABs = class1.results.direcs > 0
        BAs = class1.results.direcs < 0

        indexing = np.zeros(int(link_ids.max()) + 1, np.uint64)
        indexing[agg.index[:]] = np.arange(entries)

        # Indices of links BA and AB
        ab_ids = indexing[link_ids[ABs]]
        ba_ids = indexing[link_ids[BAs]]

        agg.data["Congested_Time_AB"][ab_ids] = np.nan_to_num(congested_time[ABs])
        agg.data["Congested_Time_BA"][ba_ids] = np.nan_to_num(congested_time[BAs])
        agg.data["Congested_Time_Max"][:] = np.nanmax([agg.data.Congested_Time_AB, agg.data.Congested_Time_BA], axis=0)

        agg.data["Delay_factor_AB"][ab_ids] = np.nan_to_num(congested_time[ABs] / free_flow_tt[ABs])
        agg.data["Delay_factor_BA"][ba_ids] = np.nan_to_num(congested_time[BAs] / free_flow_tt[BAs])
        agg.data["Delay_factor_Max"][:] = np.nanmax([agg.data.Delay_factor_AB, agg.data.Delay_factor_BA], axis=0)

        agg.data["VOC_AB"][ab_ids] = np.nan_to_num(voc[ABs])
        agg.data["VOC_BA"][ba_ids] = np.nan_to_num(voc[BAs])
        agg.data["VOC_max"][:] = np.nanmax([agg.data.VOC_AB, agg.data.VOC_BA], axis=0)

        agg.data["PCE_AB"][ab_ids] = np.nan_to_num(tot_flow[ABs])
        agg.data["PCE_BA"][ba_ids] = np.nan_to_num(tot_flow[BAs])
        agg.data["PCE_tot"][:] = np.nansum([agg.data.PCE_AB, agg.data.PCE_BA], axis=0)

        assig_results.append(agg)

        dfs = [pd.DataFrame(aed.data) for aed in assig_results]
        dfs = [df.rename(columns={"index": "link_id"}).set_index("link_id") for df in dfs]
        df = pd.concat(dfs, axis=1)

        return df

    def report(self) -> pd.DataFrame:
        """Returns the assignment convergence report

         Returns:
            *DataFrame* (:obj:`pd.DataFrame`): Convergence report
        """
        return pd.DataFrame(self.assignment.convergence_report)

    def info(self) -> dict:
        """ Returns information for the traffic assignment procedure

        Dictionary contains keys  'Algorithm', 'Classes', 'Computer name', 'Procedure ID',
        'Maximum iterations' and 'Target RGap'.

        The classes key is also a dictionary with all the user classes per traffic class and their respective
        matrix totals

        Returns:
            *info* (:obj:`dict`): Pandas dataframe with all the assignment results indexed on link_id
        """

        classes = {}

        for cls in self.classes:
            uclass = {}

            if len(cls.matrix.view_names) == 1:
                uclass['matrix_totals'] = {nm: np.sum(cls.matrix.matrix_view[:, :]) for nm in
                                           cls.matrix.view_names}
            else:
                uclass['matrix_totals'] = {nm: np.sum(cls.matrix.matrix_view[:, :, i]) for i, nm in
                                           enumerate(cls.matrix.view_names)}
            uclass['network mode'] = cls.graph.mode
            uclass['Value-of-time'] = cls.vot
            uclass['PCE'] = cls.pce
            if cls.fixed_cost_field:
                uclass['Fixed cost field'] = cls.fixed_cost_field
                uclass['Fixed cost multiplier'] = cls.fc_multiplier
            classes[cls.__id__] = uclass

        info = {
            "Algorithm": self.algorithm,
            "Classes": classes,
            "Computer name": socket.gethostname(),
            "Maximum iterations": self.assignment.max_iter,
            "Procedure ID": self.procedure_id,
            "Target RGap": self.assignment.rgap_target,
        }
        return info

    def save_skims(self, matrix_name: str, which_ones="final", format="omx") -> None:
        """Saves the skims (if any) to the skim folder and registers in the matrix list

        Args:
            name (:obj:`str`): Name of the matrix record to hold this matrix (same name used for file name)
            which_ones (:obj:`str`,optional): {'final': Results of the final iteration, 'blended': Averaged results for
            all iterations, 'all': Saves skims for both the final iteration and the blended ones} Default is 'final'
            *format* (:obj:`str`, `Optional`): File format ('aem' or 'omx'). Default is 'omx'
        """

        mat_format = format.lower()
        if mat_format not in ["omx", "aem"]:
            raise ValueError("Matrix needs to be either OMX or native AequilibraE")
        if mat_format == "omx" and not has_omx:
            raise ImportError("OpenMatrix is not available on your system")

        file_name = f"{matrix_name}.{mat_format}"

        mats = Matrices()
        export_name = path.join(mats.fldr, file_name)

        if path.isfile(export_name):
            raise FileExistsError(f"{file_name} already exists. Choose a different name or matrix format")

        if mats.check_exists(matrix_name):
            raise FileExistsError(f"{matrix_name} already exists. Choose a different name")

        avg_skims = self.classes[0].results.skims  # type: AequilibraeMatrix

        # The ones for the last iteration are here
        last_skims = self.classes[0]._aon_results.skims  # type: AequilibraeMatrix

        names = []
        if which_ones in ["final", "all"]:
            for core in last_skims.names:
                names.append(f"{core}_final")

        if which_ones in ["blended", "all"]:
            for core in avg_skims.names:
                names.append(f"{core}_blended")

        if not names:
            raise ValueError("No skims to save")
        # Assembling a single final skim file can be done like this
        # We will want only the time for the last iteration and the distance averaged out for all iterations
        working_name = export_name if mat_format == "aem" else AequilibraeMatrix().random_name()

        kwargs = {"file_name": working_name, "zones": self.classes[0].graph.centroids.shape[0], "matrix_names": names}

        # Create the matrix to manipulate
        out_skims = AequilibraeMatrix()
        out_skims.create_empty(**kwargs)

        out_skims.index[:] = self.classes[0].graph.centroids[:]
        out_skims.description = f"Assignment skim from procedure ID {self.procedure_id}"

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

        # Now we create the appropriate record
        record = mats.new_record(matrix_name, file_name)
        record.procedure_id = self.procedure_id
        record.timestamp = self.procedure_date
        record.procedure = "Traffic Assignment"
        record.description = "Skimming for assignment procedure"
        record.save()
