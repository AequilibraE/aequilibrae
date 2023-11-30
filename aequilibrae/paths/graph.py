from os.path import join
import pickle
import uuid
from datetime import datetime
from typing import List, Tuple
import numpy as np
import pandas as pd
from aequilibrae.context import get_logger
from aequilibrae.paths.AoN import build_compressed_graph


class Graph(object):
    """
    Graph class
    """

    def __init__(self, logger=None):
        self.logger = logger or get_logger()
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

        self.compact_num_links = -1
        self.compact_num_nodes = -1

        self.network = pd.DataFrame([])  # This method will hold ALL information on the network
        self.graph = pd.DataFrame([])  # This method will hold an array with ALL fields in the graph.

        self.compact_graph = pd.DataFrame([])  # This method will hold an array with ALL fields in the graph.

        # These are the fields actually used in computing paths
        self.all_nodes = np.array(0)  # Holds an array with all nodes in the original network
        self.nodes_to_indices = np.array(0, np.int64)  # Holds the reverse of the all_nodes
        self.fs = np.array([])  # This method will hold the forward star for the graph
        self.cost = np.array([])  # This array holds the values being used in the shortest path routine
        self.skims = None

        self.lonlat_index = pd.DataFrame([])  # Holds a node_id to lon/lat coord index for nodes within this graph

        self.compact_all_nodes = np.array(0)  # Holds an array with all nodes in the original network
        self.compact_nodes_to_indices = np.array(0)  # Holds the reverse of the all_nodes
        self.compact_fs = np.array([])  # This method will hold the forward star for the graph
        self.compact_cost = np.array([])  # This array holds the values being used in the shortest path routine
        self.compact_skims = None

        self.capacity = np.array([])  # Array holds the capacity for links
        self.free_flow_time = np.array([])  # Array holds the free flow travel time by link

        # sake of the Cython code
        self.skim_fields = []  # List of skim fields to be used in computation
        self.cost_field = False  # Name of the cost field

        self.block_centroid_flows = True
        self.penalty_through_centroids = np.inf

        self.centroids = None  # NumPy array of centroid IDs

        self.g_link_crosswalk = np.array([])  # 4 a link ID in the BIG graph, a corresponding link in the compressed 1

        # Randomly generate a unique Graph ID randomly
        self.__id__ = uuid.uuid4().hex

    def default_types(self, tp: str):
        """
        Returns the default integer and float types used for computation

        :Arguments:
            **tp** (:obj:`str`): data type. 'int' or 'float'
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

        :Arguments:
            **centroids** (:obj:`np.ndarray`): Array with centroid IDs. Mandatory type Int64, unique and positive
        """
        self.__network_error_checking__()

        # Creates the centroids

        if centroids is None or not isinstance(centroids, np.ndarray):
            raise ValueError("Centroids need to be a NumPy array of integers 64 bits")
        if not np.issubdtype(centroids.dtype, np.integer):
            raise ValueError("Centroids need to be a NumPy array of integers 64 bits")
        if centroids.shape[0] == 0:
            raise ValueError("You need at least one centroid")
        if centroids.min() <= 0:
            raise ValueError("Centroid IDs need to be positive")
        if centroids.shape[0] != np.unique(centroids).shape[0]:
            raise ValueError("Centroid IDs are not unique")
        self.centroids = np.array(centroids, np.uint32)

        self.network = self.network.astype(
            {
                "direction": np.int8,
                "a_node": self.__integer_type,
                "b_node": self.__integer_type,
                "link_id": self.__integer_type,
            }
        )

        properties = self.__build_directed_graph(self.network, centroids)
        self.all_nodes, self.num_nodes, self.nodes_to_indices, self.fs, self.graph = properties

        # We generate IDs that we KNOW will be constant across modes
        self.graph.sort_values(by=["link_id", "direction"], inplace=True)
        self.graph["__supernet_id__"] = np.arange(self.graph.shape[0]).astype(self.__integer_type)
        self.graph.sort_values(by=["a_node", "b_node"], inplace=True)

        self.num_links = self.graph.shape[0]
        self.__build_derived_properties()

        self.__build_compressed_graph()
        self.compact_num_links = self.compact_graph.shape[0]

    def __build_compressed_graph(self):
        build_compressed_graph(self)

        # We build a groupby to save time later
        self.__graph_groupby = self.graph.groupby(["__compressed_id__"])

    def __build_directed_graph(self, network: pd.DataFrame, centroids: np.ndarray):
        all_titles = list(network.columns)

        not_pos = network.loc[network.direction != 1, :]
        not_negs = network.loc[network.direction != -1, :]

        names, types = self.__build_column_names(all_titles)
        neg_names = []
        for name in names:
            if name in not_pos.columns:
                neg_names.append(name)
            elif name + "_ba" in not_pos.columns:
                neg_names.append(name + "_ba")
        not_pos = pd.DataFrame(not_pos, copy=True)[neg_names]
        not_pos.columns = names
        not_pos.loc[:, "direction"] = -1
        aux = np.array(not_pos.a_node.values, copy=True)
        not_pos.loc[:, "a_node"] = not_pos.loc[:, "b_node"]
        not_pos.loc[:, "b_node"] = aux[:]
        del aux

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
        all_nodes = np.hstack((centroids, nodes)).astype(self.__integer_type)

        num_nodes = all_nodes.shape[0]

        nodes_to_indices = np.full(int(all_nodes.max()) + 1, -1, dtype=np.int64)
        nlist = np.arange(num_nodes)
        nodes_to_indices[all_nodes] = nlist

        df.a_node = nodes_to_indices[df.a_node.values]
        df.b_node = nodes_to_indices[df.b_node.values]
        df = df.sort_values(by=["a_node", "b_node"])
        df.index = np.arange(df.shape[0])
        df["id"] = np.arange(df.shape[0])
        fs = np.empty(num_nodes + 1, dtype=self.__integer_type)
        fs.fill(-1)
        y, x, _ = np.intersect1d(df.a_node.values, nlist, assume_unique=False, return_indices=True)
        fs[y] = x[:]
        fs[-1] = df.shape[0]
        for i in range(num_nodes, 1, -1):
            if fs[i - 1] == -1:
                fs[i - 1] = fs[i]

        nans = ", ".join([i for i in df.columns if df[i].isnull().any().any()])
        if nans:
            self.logger.warning(f"Field(s) {nans} has(ve) at least one NaN value. Check your computations")

        df.loc[:, "b_node"] = df.b_node.values.astype(self.__integer_type)
        df.loc[:, "id"] = df.id.values.astype(self.__integer_type)
        df.loc[:, "link_id"] = df.link_id.values.astype(self.__integer_type)
        df["direction"] = df.direction.values.astype(np.int8)

        return all_nodes, num_nodes, nodes_to_indices, fs, df

    def exclude_links(self, links: list) -> None:
        """
        Excludes a list of links from a graph by setting their B node equal to their A node

        :Arguments:
            **links** (:obj:`list`): List of link IDs to be excluded from the graph
        """
        filter = self.network.link_id.isin(links)
        # We check is the list makes sense in order to warn the user
        if filter.sum() != len(set(links)):
            self.logger.warning("At least one link does not exist in the network and therefore cannot be excluded")

        self.network.loc[filter, "b_node"] = self.network.loc[filter, "a_node"]

        if self.centroids is not None:
            self.prepare_graph(self.centroids)
            self.set_blocked_centroid_flows(self.block_centroid_flows)
        self.__id__ = uuid.uuid4().hex

    def __build_column_names(self, all_titles: List[str]) -> Tuple[list, list]:
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

        :Arguments:
            **cost_field** (:obj:`str`): Field name. Must be numeric
        """
        if cost_field in self.graph.columns:
            self.cost_field = cost_field
            self.compact_cost = np.zeros(self.compact_graph.id.max() + 2, self.__float_type)
            df = self.__graph_groupby.sum(numeric_only=True)[[cost_field]].reset_index()
            self.compact_cost[df.index.values] = df[cost_field].values
            if self.graph[cost_field].dtype == self.__float_type:
                self.cost = np.array(self.graph[cost_field].values, copy=True)
            else:
                self.cost = np.array(self.graph[cost_field].values, dtype=self.__float_type)
                self.logger.warning("Cost field with wrong type. Converting to float64")
        else:
            raise ValueError("cost_field not available in the graph:" + str(self.graph.columns))

        self.__build_derived_properties()

    def set_skimming(self, skim_fields: list) -> None:
        """
        Sets the list of skims to be computed

        Skimming with A* may produce results that differ from tradditional Dijkstra's due to its use a heuristic.

        :Arguments:
            **skim_fields** (:obj:`list`): Fields must be numeric
        """
        if not skim_fields:
            self.skim_fields = []
            self.skims = np.array([])

        if isinstance(skim_fields, str):
            skim_fields = [skim_fields]
        elif not isinstance(skim_fields, list):
            raise ValueError("You need to provide a list of skims or the same of a single field")

        # Check if list of fields make sense
        k = [x for x in skim_fields if x not in self.graph.columns]
        if k:
            raise ValueError("At least one of the skim fields does not exist in the graph: {}".format(",".join(k)))

        self.compact_skims = np.zeros((self.compact_num_links + 1, len(skim_fields) + 1), self.__float_type)
        df = self.__graph_groupby.sum(numeric_only=True)[skim_fields].reset_index()
        for i, skm in enumerate(skim_fields):
            self.compact_skims[df.index.values, i] = df[skm].values.astype(self.__float_type)

        self.skims = np.zeros((self.num_links, len(skim_fields) + 1), self.__float_type)
        t = [x for x in skim_fields if self.graph[x].dtype != self.__float_type]
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

        :Arguments:
            **block_centroid_flows** (:obj:`bool`): Blocking or not
        """
        if not isinstance(block_centroid_flows, bool):
            raise TypeError("Blocking flows through centroids needs to be boolean")
        if self.num_zones == 0:
            self.logger.warning("No centroids in the model. Nothing to block")
            return
        self.block_centroid_flows = block_centroid_flows

    # Procedure to pickle graph and save to disk
    def save_to_disk(self, filename: str) -> None:
        """
        Saves graph to disk

        :Arguments:
            **filename** (:obj:`str`): Path to file. Usual file extension is *aeg*
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
        mygraph["cost"] = self.cost
        mygraph["cost_field"] = self.cost_field
        mygraph["skims"] = self.skims
        mygraph["skim_fields"] = self.skim_fields
        mygraph["block_centroid_flows"] = self.block_centroid_flows
        mygraph["centroids"] = self.centroids
        mygraph["graph_id"] = self.__id__
        mygraph["mode"] = self.mode

        with open(filename, "wb") as f:
            pickle.dump(mygraph, f)

    def load_from_disk(self, filename: str) -> None:
        """
        Loads graph from disk

        :Arguments:
            **filename** (:obj:`str`): Path to file
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
            self.cost = mygraph["cost"]
            self.cost_field = mygraph["cost_field"]
            self.skims = mygraph["skims"]
            self.skim_fields = mygraph["skim_fields"]
            self.block_centroid_flows = mygraph["block_centroid_flows"]
            self.centroids = mygraph["centroids"]
            self.__id__ = mygraph["graph_id"]
            self.mode = mygraph["mode"]
        self.__build_derived_properties()

    def __build_derived_properties(self):
        if self.centroids is None:
            return
        self.num_zones = self.centroids.shape[0] if self.centroids.shape else 0

    def available_skims(self) -> List[str]:
        """
        Returns graph fields that are available to be set as skims

        :Returns:
            **list** (:obj:`str`): Field names
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
        link_ids = self.network["link_id"].astype(int)
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

    def save_compressed_correspondence(self, path, mode_name, mode_id):
        """Save graph and nodes_to_indices to disk"""
        graph_path = join(path, f"correspondence_c{mode_name}_{mode_id}.feather")
        self.graph.to_feather(graph_path)
        node_path = join(path, f"nodes_to_indices_c{mode_name}_{mode_id}.feather")
        pd.DataFrame(self.nodes_to_indices, columns=["node_index"]).to_feather(node_path)
