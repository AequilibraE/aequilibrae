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

        self.compact_num_links = -1
        self.compact_num_nodes = -1

        self.network = pd.DataFrame([])  # This method will hold ALL information on the network
        self.graph = pd.DataFrame([])  # This method will hold an array with ALL fields in the graph.

        self.compact_graph = pd.DataFrame([])  # This method will hold an array with ALL fields in the graph.

        # These are the fields actually used in computing paths
        self.all_nodes = np.array(0)  # Holds an array with all nodes in the original network
        self.nodes_to_indices = np.array(0)  # Holds the reverse of the all_nodes
        self.fs = np.array([])  # This method will hold the forward star for the graph
        self.cost = np.array([])  # This array holds the values being used in the shortest path routine
        self.skims = None

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

        self.__version__ = VERSION

        # Randomly generate a unique Graph ID randomly
        self.__id__ = uuid.uuid4().hex

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

        properties = self.__build_directed_graph(self.network, centroids)
        self.all_nodes, self.num_nodes, self.nodes_to_indices, self.fs, self.graph = properties

        # We generate IDs that we KNOW will be constant across modes
        self.graph.sort_values(by=["link_id", "direction"], inplace=True)
        self.graph.loc[:, "__supernet_id__"] = np.arange(self.graph.shape[0]).astype(self.__integer_type)
        self.graph.sort_values(by=["a_node", "b_node"], inplace=True)

        self.num_links = self.graph.shape[0]
        self.__build_derived_properties()

        self.__build_compressed_graph()
        self.compact_num_links = self.compact_graph.shape[0]

    def __build_compressed_graph(self):
        # Build link index
        link_idx = np.empty(self.network.link_id.max() + 1).astype(np.int)
        link_idx[self.network.link_id] = np.arange(self.network.shape[0])

        nodes = np.hstack([self.network.a_node.values, self.network.b_node.values])
        links = np.hstack([self.network.link_id.values, self.network.link_id.values])
        counts = np.bincount(nodes)

        idx = np.argsort(nodes)
        all_nodes = nodes[idx]
        all_links = links[idx]
        links_index = np.empty(all_nodes.max() + 2, np.int64)
        links_index.fill(-1)
        nlist = np.arange(all_nodes.max() + 2)

        y, x, _ = np.intersect1d(all_nodes, nlist, assume_unique=False, return_indices=True)
        links_index[y] = x[:]
        links_index[-1] = all_links.shape[0]

        for i in range(all_nodes.max() + 1, 0, -1):
            links_index[i - 1] = links_index[i] if links_index[i - 1] == -1 else links_index[i - 1]

        # We keep all centroids for sure
        counts[self.centroids] = 999

        truth = (counts == 2).astype(np.int)
        link_edge = truth[self.network.a_node.values] + truth[self.network.b_node.values]
        link_edge = self.network.link_id.values[link_edge == 1]

        simplified_links = np.repeat(-1, self.network.link_id.max() + 1)
        simplified_directions = np.zeros(self.network.link_id.max() + 1, np.int)

        compressed_dir = np.zeros(self.network.link_id.max() + 1, np.int)
        compressed_a_node = np.zeros(self.network.link_id.max() + 1, np.int)
        compressed_b_node = np.zeros(self.network.link_id.max() + 1, np.int)

        slink = 0
        major_nodes = {}
        tot = 0
        tot_graph_add = 0
        for pre_link in link_edge:
            if simplified_links[pre_link] >= 0:
                continue
            ab_dir = 1
            ba_dir = 1
            lidx = link_idx[pre_link]
            a_node = self.network.a_node.values[lidx]
            b_node = self.network.b_node.values[lidx]
            drc = self.network.direction.values[lidx]
            n = a_node if counts[a_node] == 2 else b_node
            first_node = b_node if counts[a_node] == 2 else a_node

            ab_dir = 0 if (first_node == a_node and drc < 0) or (first_node == b_node and drc > 0) else ab_dir
            ba_dir = 0 if (first_node == a_node and drc > 0) or (first_node == b_node and drc < 0) else ba_dir

            while counts[n] == 2:
                # assert (simplified_links[pre_link] >= 0), "How the heck did this happen?"
                simplified_links[pre_link] = slink
                simplified_directions[pre_link] = -1 if a_node == n else 1

                # Gets the link from the list that is not the link we are coming from
                lnk = [all_links[k] for k in range(links_index[n], links_index[n + 1]) if pre_link != all_links[k]][0]
                pre_link = lnk
                lidx = link_idx[pre_link]
                a_node = self.network.a_node.values[lidx]
                b_node = self.network.b_node.values[lidx]
                drc = self.network.direction.values[lidx]
                ab_dir = 0 if (n == a_node and drc < 0) or (n == b_node and drc > 0) else ab_dir
                ba_dir = 0 if (n == a_node and drc > 0) or (n == b_node and drc < 0) else ba_dir
                n = (
                    self.network.a_node.values[lidx]
                    if n == self.network.b_node.values[lidx]
                    else self.network.b_node.values[lidx]
                )

            if max(ab_dir, ba_dir) < 1:
                tot += 1

            tot_graph_add += ab_dir + ba_dir
            simplified_links[pre_link] = slink
            simplified_directions[pre_link] = -1 if a_node == n else 1
            last_node = b_node if counts[a_node] == 2 else a_node
            major_nodes[slink] = [first_node, last_node]

            # Available directions are NOT indexed like the other arrays
            compressed_a_node[slink] = first_node
            compressed_b_node[slink] = last_node
            if ab_dir > 0:
                if ba_dir > 0:
                    compressed_dir[slink] = 0
                else:
                    compressed_dir[slink] = 1
            elif ba_dir > 0:
                compressed_dir[slink] = -1
            else:
                compressed_dir[slink] = -999
            slink += 1

        links_to_remove = np.argwhere(simplified_links >= 0)
        df = pd.DataFrame(self.network, copy=True)
        df = df[~df.link_id.isin(links_to_remove[:, 0])]
        df = df[df.a_node != df.b_node]

        comp_lnk = pd.DataFrame(
            {
                "a_node": compressed_a_node[:slink],
                "b_node": compressed_b_node[:slink],
                "direction": compressed_dir[:slink],
                "link_id": np.arange(slink),
            }
        )
        max_link_id = self.network.link_id.max() * 10
        comp_lnk.loc[:, "link_id"] += max_link_id

        df = pd.concat([df, comp_lnk])
        df = df[["id", "link_id", "a_node", "b_node", "direction"]]
        properties = self.__build_directed_graph(df, self.centroids)
        self.compact_all_nodes = properties[0]
        self.compact_num_nodes = properties[1]
        self.compact_nodes_to_indices = properties[2]
        self.compact_fs = properties[3]
        self.compact_graph = properties[4]

        crosswalk = pd.DataFrame(
            {
                "link_id": np.arange(simplified_directions.shape[0]),
                "link_direction": simplified_directions,
                "compressed_link": simplified_links,
                "compressed_direction": np.ones(simplified_directions.shape[0]).astype(np.int),
            }
        )

        crosswalk = crosswalk[crosswalk.compressed_link >= 0]
        crosswalk.loc[:, "compressed_link"] += max_link_id

        cw2 = pd.DataFrame(crosswalk, copy=True)
        cw2.loc[:, "link_direction"] *= -1
        cw2.loc[:, "compressed_direction"] = -1

        crosswalk = pd.concat([crosswalk, cw2])
        crosswalk = crosswalk.assign(key=crosswalk.compressed_link * crosswalk.compressed_direction)
        crosswalk.drop(["compressed_link", "compressed_direction"], axis=1, inplace=True)

        final_ids = pd.DataFrame(self.compact_graph[["id", "link_id", "direction"]], copy=True)
        final_ids = final_ids.assign(key=final_ids.link_id * final_ids.direction)
        final_ids.drop(["link_id", "direction"], axis=1, inplace=True)

        agg_crosswalk = crosswalk.merge(final_ids, on="key")
        agg_crosswalk.loc[:, "key"] = agg_crosswalk.link_id * agg_crosswalk.link_direction
        agg_crosswalk.drop(["link_id", "link_direction"], axis=1, inplace=True)

        direct_crosswalk = final_ids[final_ids.key.abs() < max_link_id]

        crosswalk = pd.concat([agg_crosswalk, direct_crosswalk])[["key", "id"]]
        crosswalk.columns = ["__graph_correlation_key__", "__compressed_id__"]

        self.graph = self.graph.assign(__graph_correlation_key__=self.graph.link_id * self.graph.direction)
        self.graph = self.graph.merge(crosswalk, on="__graph_correlation_key__", how="left")
        self.graph.drop(["__graph_correlation_key__"], axis=1, inplace=True)

        # If will refer all the links that have no correlation to an element beyond the last link
        # This element will always be zero during assignment
        self.graph.loc[self.graph.__compressed_id__.isna(), "__compressed_id__"] = self.compact_graph.id.max() + 1

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

        nodes_to_indices = np.repeat(-1, int(all_nodes.max()) + 1)
        nlist = np.arange(num_nodes)
        nodes_to_indices[all_nodes] = nlist

        df.loc[:, "a_node"] = nodes_to_indices[df.a_node.values][:]
        df.loc[:, "b_node"] = nodes_to_indices[df.b_node.values][:]
        df = df.sort_values(by=["a_node", "b_node"])
        df.index = np.arange(df.shape[0])
        df.loc[:, "id"] = np.arange(df.shape[0])
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
            logger.warning(f"Field(s) {nans} has(ve) at least one NaN value. Check your computations")

        df.loc[:, "b_node"] = df.b_node.values.astype(self.__integer_type)
        df.loc[:, "id"] = df.id.values.astype(self.__integer_type)
        df.loc[:, "link_id"] = df.link_id.values.astype(self.__integer_type)
        df.loc[:, "direction"] = df.direction.values.astype(np.int8)

        return all_nodes, num_nodes, nodes_to_indices, fs, df

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

        if self.centroids is not None:
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
            self.compact_cost = np.zeros(self.compact_graph.id.max() + 1, self.__float_type)
            df = self.__graph_groupby.sum()[[cost_field]].reset_index()
            self.compact_cost[df.index.values[:-1]] = df[cost_field].values[:-1]
            if self.graph[cost_field].dtype == self.__float_type:
                self.cost = np.array(self.graph[cost_field].values, copy=True)
            else:
                self.cost = np.array(self.graph[cost_field].values, dtype=self.__float_type)
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
            self.skims = np.array([])

        if isinstance(skim_fields, str):
            skim_fields = [skim_fields]
        elif not isinstance(skim_fields, list):
            raise ValueError("You need to provide a list of skims or the same of a single field")

        # Check if list of fields make sense
        k = [x for x in skim_fields if x not in self.graph.columns]
        if k:
            raise ValueError("At least one of the skim fields does not exist in the graph: {}".format(",".join(k)))

        self.compact_skims = np.zeros((self.compact_num_links, len(skim_fields) + 1), self.__float_type)
        df = self.__graph_groupby.sum()[skim_fields].reset_index()
        for i, skm in enumerate(skim_fields):
            self.compact_skims[df.index.values[:-1], i] = df[skm].values[:-1].astype(self.__float_type)

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

        Args:
            block_centroid_flows (:obj:`bool`): Blocking or not
        """
        if not isinstance(block_centroid_flows, bool):
            raise TypeError("Blocking flows through centroids needs to be boolean")
        if self.num_zones == 0:
            warn("No centroids in the model. Nothing to block")
            return
        self.block_centroid_flows = block_centroid_flows

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
        mygraph["cost"] = self.cost
        mygraph["cost_field"] = self.cost_field
        mygraph["skims"] = self.skims
        mygraph["skim_fields"] = self.skim_fields
        mygraph["block_centroid_flows"] = self.block_centroid_flows
        mygraph["centroids"] = self.centroids
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
            self.cost = mygraph["cost"]
            self.cost_field = mygraph["cost_field"]
            self.skims = mygraph["skims"]
            self.skim_fields = mygraph["skim_fields"]
            self.block_centroid_flows = mygraph["block_centroid_flows"]
            self.centroids = mygraph["centroids"]
            self.__id__ = mygraph["graph_id"]
            self.__version__ = mygraph["graph_version"]
            self.mode = mygraph["mode"]
        self.__build_derived_properties()

    def __build_derived_properties(self):
        if self.centroids is None:
            return
        self.num_zones = self.centroids.shape[0] if self.centroids.shape else 0

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

    def save_compressed_correspondence(self, path):
        self.graph[["link_id", "__supernet_id__", "__compressed_id__"]].to_feather(path)
