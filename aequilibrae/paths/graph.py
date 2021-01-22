import logging
import pickle
import uuid
from datetime import datetime
from warnings import warn
from typing import List
import numpy as np
import pandas as pd
from aequilibrae.starts_logging import logger
from .__version__ import binary_version as VERSION


class Graph(object):
    """
    Graph class
    """

    def __init__(self):
        self.logger = logging.getLogger("aequilibrae")
        self.__integer_type = np.int64
        self.__float_type = np.float64

        self.required_default_fields = ["link_id", "a_node", "b_node", "direction", "id"]
        self.__required_default_types = [
            self.__integer_type,
            self.__integer_type,
            self.__integer_type,
            np.int8,
            self.__integer_type,
        ]
        self.other_fields = ""
        self.mode = ""
        self.date = str(datetime.now())

        self.description = "No description added so far"

        self.num_links = -1
        self.num_nodes = -1
        self.num_zones = -1
        self.network = pd.DataFrame([])  # This method will hold ALL information on the network
        self.graph = pd.DataFrame([])  # This method will hold an array with ALL fields in the graph.

        # These are the fields actually used in computing paths
        self.all_nodes = False  # Holds an array with all nodes in the original network
        self.nodes_to_indices = False  # Holds the reverse of the all_nodes
        self.fs = False  # This method will hold the forward star for the graph
        self.b_node = False  # b node for each directed link

        self.cost = None  # This array holds the values being used in the shortest path routine
        self.capacity = None  # Array holds the capacity for links
        self.free_flow_time = None  # Array holds the free flow travel time by link
        self.skims = None
        # sake of the Cython code
        self.skim_fields = []  # List of skim fields to be used in computation
        self.cost_field = False  # Name of the cost field
        self.ids = np.array(0)  # 1-D Array with link IDs (sequence from 0 to N-1)

        self.block_centroid_flows = True
        self.penalty_through_centroids = np.inf

        self.centroids = None  # NumPy array of centroid IDs

        self.network_ok = False
        self.__version__ = VERSION

        # Randomly generate a unique Graph ID randomly
        self.__id__ = uuid.uuid4().hex
        self.__source__ = None  # Name of the file that originated the graph

        # In case the graph is generated in QGIS, it is useful to have the name of the layer and fields that originated
        # it
        self.__field_name__ = None
        self.__layer_name__ = None

    def default_types(self, tp: str):
        """
        Returns the default integer and float types used for computation

        Args:
            tp (:obj:`str`): data type. 'int' or 'float'
        """
        if tp == "int":
            return self.__integer_type
        elif tp == "float":
            return self.__float_type
        else:
            raise ValueError("It must be either a int or a float")

    def prepare_graph(self, centroids: np.ndarray) -> None:
        """
        Prepares the graph for a computation for a certain set of centroids

        Under the hood, if sets all centroids to have IDs from 1 through **n**,
        which should correspond to the index of the matrix being assigned.

        This is what enables having any node IDs as centroids, and it relies on
        the inference that all links connected to these nodes are centroid
        connectors.

        Args:
            centroids (:obj:`np.ndarray`): Array with centroid IDs. Mandatory type Int64, unique and positive
        """
        self.__network_error_checking__()

        # Creates the centroids
        if centroids is not None and isinstance(centroids, np.ndarray):
            if np.issubdtype(centroids.dtype, np.integer):
                if centroids.shape[0] > 0:
                    if centroids.min() <= 0:
                        raise ValueError("Centroid IDs need to be positive")
                    else:
                        if centroids.shape[0] != np.unique(centroids).shape[0]:
                            raise ValueError("Centroid IDs are not unique")
                self.centroids = np.array(list(centroids), np.uint32)
            else:
                raise ValueError("Centroids need to be an array of integers 64 bits")
        else:
            raise ValueError("Centroids need to be a NumPy array of integers 64 bits")
        self.__build_derived_properties()

        all_titles = list(self.network.columns)

        not_pos = self.network.loc[self.network.direction != 1, :]
        not_negs = self.network.loc[self.network.direction != -1, :]

        names, types = self.__build_column_names(all_titles)
        neg_names = []
        for name in names:
            if name in not_pos.columns:
                neg_names.append(name)
            elif name + "_ba" in not_pos.columns:
                neg_names.append(name + "_ba")
        not_pos = pd.DataFrame(not_pos, copy=True)[neg_names]
        not_pos.columns = names
        not_pos.loc[:, "direction"] = 1

        pos_names = []
        for name in names:
            if name in not_negs.columns:
                pos_names.append(name)
            elif name + "_ab" in not_negs.columns:
                pos_names.append(name + "_ab")
        not_negs = pd.DataFrame(not_negs, copy=True)[pos_names]
        not_negs.columns = names
        not_negs.loc[:, "direction"] = 1

        df = pd.concat([not_negs, not_pos])

        # Now we take care of centroids
        nodes = np.unique(np.hstack((df.a_node.values, df.b_node.values))).astype(self.__integer_type)
        nodes = np.setdiff1d(nodes, centroids, assume_unique=True)
        self.all_nodes = np.hstack((centroids, nodes)).astype(self.__integer_type)

        self.num_nodes = self.all_nodes.shape[0]
        self.nodes_to_indices = np.empty(int(self.all_nodes.max()) + 1, self.__integer_type)
        self.nodes_to_indices.fill(-1)
        self.nodes_to_indices[self.all_nodes] = np.arange(self.num_nodes)

        self.num_links = df.shape[0]

        df.loc[:, "a_node"] = self.nodes_to_indices[df.a_node][:]
        df.loc[:, "b_node"] = self.nodes_to_indices[df.b_node][:]
        df = df.sort_values(by=["a_node", "b_node"])
        df.index = np.arange(df.shape[0])
        df.loc[:, "id"] = np.arange(df.shape[0])
        self.fs = np.empty(self.num_nodes + 1, dtype=self.__integer_type)
        self.fs.fill(-1)
        x = np.arange(self.num_nodes + 1, dtype=self.__integer_type)

        _, _, x = np.intersect1d(x, df.a_node.values, assume_unique=False, return_indices=True)
        self.fs[: x.shape[0]] = x[:]
        self.fs[self.num_nodes] = df.shape[0]
        for i in range(self.num_nodes, 1, -1):
            if self.fs[i - 1] == -1:
                self.fs[i - 1] = self.fs[i]

        nans = ",".join([i for i in df.columns if df[i].isnull().any().any()])
        if nans:
            logger.warning(f"Field(s) {nans} has(ve) at least one NaN value. Check your computations")
        self.graph = df
        self.ids = self.graph.id.values.astype(self.__integer_type)
        self.__build_arrays_for_computation()

    def exclude_links(self, links: list) -> None:
        """
        Excludes a list of links from a graph by setting their B node equal to their A node

        Args:
            links (:obj:`list`): List of link IDs to be excluded from the graph
        """
        filter = self.network.link_id.isin(links)
        # We check is the list makes sense in order to warn the user
        if filter.sum() != len(set(links)):
            warn("At least one link does not exist in the network and therefore cannot be excluded")

        self.network.loc[filter, "b_node"] = self.network.loc[filter, "a_node"]

        self.prepare_graph(self.centroids)
        self.set_blocked_centroid_flows(self.block_centroid_flows)
        self.__id__ = uuid.uuid4().hex

    def __build_column_names(self, all_titles: [str]) -> (list, list):
        fields = [x for x in self.required_default_fields]
        types = [x for x in self.__required_default_types]
        for column in all_titles:
            if column not in self.required_default_fields and column[0:-3] not in self.required_default_fields:
                if column[-3:] == "_ab":
                    if column[:-3] + "_ba" in all_titles:
                        fields.append(column[:-3])
                        types.append(self.network[column].dtype)
                    else:
                        raise ValueError("Field {} exists for ab direction but does not exist for ba".format(column))
                elif column[-3:] == "_ba":
                    if column[:-3] + "_ab" not in all_titles:
                        raise ValueError("Field {} exists for ba direction but does not exist for ab".format(column))
                else:
                    fields.append(column)
                    types.append(self.network[column].dtype)
        return fields, types

    def __build_arrays_for_computation(self):
        self._comp_b_node = np.array(self.graph.b_node.values.astype(self.__integer_type), copy=True)
        self._comp_link_id = np.array(self.graph.link_id.values.astype(self.__integer_type), copy=True)
        self._comp_direction = np.array(self.graph.direction.values.astype(self.__integer_type), copy=True)

    def __build_dtype(self, all_titles) -> list:
        dtype = [
            ("link_id", self.__integer_type),
            ("a_node", self.__integer_type),
            ("b_node", self.__integer_type),
            ("direction", np.int8),
            ("id", self.__integer_type),
        ]
        for i in all_titles:
            if i not in self.required_default_fields and i[0:-3] not in self.required_default_fields:
                if i[-3:] == "_ab":
                    if i[:-3] + "_ba" in all_titles:
                        dtype.append((i[:-3], self.network[i].dtype))
                    else:
                        raise ValueError("Field {} exists for ab direction but does not exist for ba".format(i))
                elif i[-3:] == "_ba":
                    if i[:-3] + "_ab" not in all_titles:
                        raise ValueError("Field {} exists for ba direction but does not exist for ab".format(i))
                else:
                    dtype.append((i, self.network[i].dtype))
        return dtype

    def set_graph(self, cost_field) -> None:
        """
        Sets the field to be used for path computation

        Args:
            cost_field (:obj:`str`): Field name. Must be numeric
        """
        if cost_field in self.graph.columns:
            self.cost_field = cost_field
            if self.graph[cost_field].dtype == self.__float_type:
                self.cost = self.graph[cost_field].values
            else:
                self.cost = self.graph[cost_field].astype(self.__float_type).values
                warn("Cost field with wrong type. Converting to float64")
        else:
            raise ValueError("cost_field not available in the graph:" + str(self.graph.columns))

        self.__build_derived_properties()

    def set_skimming(self, skim_fields: list) -> None:
        """
        Sets the list of skims to be computed

        Args:
            skim_fields (:obj:`list`): Fields must be numeric
        """
        if not skim_fields:
            self.skim_fields = []
            self.skims = None

        if isinstance(skim_fields, str):
            skim_fields = [skim_fields]
        elif not isinstance(skim_fields, list):
            raise ValueError("You need to provide a list of skims or the same of a single field")

        # Check if list of fields make sense
        k = [x for x in skim_fields if x not in self.graph.columns]
        if k:
            raise ValueError("At least one of the skim fields does not exist in the graph: {}".format(",".join(k)))

        t = [x for x in skim_fields if self.graph[x].dtype != self.__float_type]

        self.skims = np.zeros((self.num_links, len(skim_fields) + 1), self.__float_type)

        if t:
            Warning("Some skim field with wrong type. Converting to float64")
            for i, j in enumerate(skim_fields):
                self.skims[:, i] = self.graph[j].astype(self.__float_type).values[:]
        else:
            for i, j in enumerate(skim_fields):
                self.skims[:, i] = self.graph[j].values[:]
        self.skim_fields = skim_fields

    def set_blocked_centroid_flows(self, block_centroid_flows) -> None:
        """
        Chooses whether we want to block paths to go through centroids or not.

        Default value is True

        Args:
            block_centroid_flows (:obj:`bool`): Blocking or not
        """
        if isinstance(block_centroid_flows, bool):
            if self.num_zones == 0:
                warn("No centroids in the model. Nothing to block")
            else:
                self.block_centroid_flows = block_centroid_flows
                self.b_node = np.array(self.graph["b_node"], self.__integer_type)
        else:
            raise TypeError("Blocking flows through centroids needs to be boolean")

    # Procedure to pickle graph and save to disk
    def save_to_disk(self, filename: str) -> None:
        """
        Saves graph to disk

        Args:
            filename (:obj:`str`): Path to file. Usual file extension is *aeg*
        """
        mygraph = {}
        mygraph["description"] = self.description
        mygraph["num_links"] = self.num_links
        mygraph["num_nodes"] = self.num_nodes
        mygraph["network"] = self.network
        mygraph["graph"] = self.graph
        mygraph["all_nodes"] = self.all_nodes
        mygraph["nodes_to_indices"] = self.nodes_to_indices
        mygraph["num_nodes"] = self.num_nodes
        mygraph["fs"] = self.fs
        mygraph["b_node"] = self.b_node
        mygraph["cost"] = self.cost
        mygraph["cost_field"] = self.cost_field
        mygraph["skims"] = self.skims
        mygraph["skim_fields"] = self.skim_fields
        mygraph["ids"] = self.ids
        mygraph["block_centroid_flows"] = self.block_centroid_flows
        mygraph["centroids"] = self.centroids
        mygraph["network_ok"] = self.network_ok
        mygraph["graph_id"] = self.__id__
        mygraph["graph_version"] = self.__version__
        mygraph["mode"] = self.mode

        with open(filename, "wb") as f:
            pickle.dump(mygraph, f)

    def load_from_disk(self, filename: str) -> None:
        """
        Loads graph from disk

        Args:
            filename (:obj:`str`): Path to file
        """
        with open(filename, "rb") as f:
            mygraph = pickle.load(f)
            self.description = mygraph["description"]
            self.num_links = mygraph["num_links"]
            self.num_nodes = mygraph["num_nodes"]
            self.network = mygraph["network"]
            self.graph = mygraph["graph"]
            self.all_nodes = mygraph["all_nodes"]
            self.nodes_to_indices = mygraph["nodes_to_indices"]
            self.num_nodes = mygraph["num_nodes"]
            self.fs = mygraph["fs"]
            self.b_node = mygraph["b_node"]
            self.cost = mygraph["cost"]
            self.cost_field = mygraph["cost_field"]
            self.skims = mygraph["skims"]
            self.skim_fields = mygraph["skim_fields"]
            self.ids = mygraph["ids"]
            self.block_centroid_flows = mygraph["block_centroid_flows"]
            self.centroids = mygraph["centroids"]
            self.network_ok = mygraph["network_ok"]
            self.__id__ = mygraph["graph_id"]
            self.__version__ = mygraph["graph_version"]
            self.mode = mygraph["mode"]
        self.__build_derived_properties()

    def __build_derived_properties(self):
        if self.centroids is not None:
            if self.centroids.shape:
                self.num_zones = self.centroids.shape[0]
            else:
                self.num_zones = 0

    def available_skims(self) -> List[str]:
        """
        Returns graph fields that are available to be set as skims

        Returns:
            *list* (:obj:`str`): Field names
        """
        return [x for x in self.graph.columns if x not in ["link_id", "a_node", "b_node", "direction", "id"]]

    # We check if all minimum fields are there
    def __network_error_checking__(self):

        # Checking field names
        has_fields = self.network.columns
        must_fields = ["link_id", "a_node", "b_node", "direction"]
        for field in must_fields:
            if field not in has_fields:
                raise ValueError(f"could not find field {field} in the network array")

        # Uniqueness of the id
        link_ids = self.network["link_id"].astype(np.int)
        if link_ids.shape[0] != np.unique(link_ids).shape[0]:
            raise ValueError('"link_id" field not unique')

            # Direction values
        if np.max(self.network["direction"]) > 1 or np.min(self.network["direction"]) < -1:
            raise ValueError('"direction" field not limited to (-1,0,1) values')

        if "id" not in self.network.columns:
            self.network = self.network.assign(id=np.nan)

    def __determine_types__(self, new_type, current_type):

        if new_type.isdigit():
            new_type = int(new_type)
        else:
            try:
                new_type = float(new_type)
            except ValueError as verr:
                self.logger.warning("Could not convert {} - {}".format(new_type, verr.__str__()))
        nt = type(new_type)
        def_type = None
        if nt == int:
            def_type = int
            if current_type == float:
                def_type == float
            elif current_type == str:
                def_type = str
        elif nt == float:
            def_type = float
            if current_type == str:
                def_type = str
        elif nt == str:
            def_type = str
        else:
            raise ValueError("WRONG TYPE OR NULL VALUE")
        return def_type
