import logging
import pickle
import uuid
from datetime import datetime
from warnings import warn
from typing import List

import numpy as np

from .__version__ import binary_version as VERSION


class Graph(object):
    """
    Graph class
    """

    def __init__(self):
        self.logger = logging.getLogger("aequilibrae")
        self.__integer_type = np.int64
        self.__float_type = np.float64

        self.required_default_fields = []
        self.__reset_single_fields()
        self.other_fields = ""
        self.mode = ''
        self.date = str(datetime.now())

        self.description = "No description added so far"

        self.num_links = -1
        self.num_nodes = -1
        self.num_zones = -1
        self.network = False  # This method will hold ALL information on the network
        self.graph = False  # This method will hold an array with ALL fields in the graph.

        # These are the fields actually used in computing paths
        self.all_nodes = False  # Holds an array with all nodes in the original network
        self.nodes_to_indices = False  # Holds the reverse of the all_nodes
        self.fs = False  # This method will hold the forward star for the graph
        self.b_node = False  # b node for each directed link

        self.cost = None  # This array holds the values being used in the shortest path routine
        self.capacity = None  # Array holds the capacity for links
        self.free_flow_time = None  # Array holds the free flow travel time by link
        self.skims = np.zeros((1, 1), self.__float_type)  # Skimming array that we initialize with something for the
        # sake of the Cython code
        self.skim_fields = []  # List of skim fields to be used in computation
        self.cost_field = False  # Name of the cost field
        self.ids = False  # 1-D Array with link IDs (sequence from 0 to N-1)

        self.block_centroid_flows = True
        self.penalty_through_centroids = np.inf

        self.centroids = None  # NumPy array of centroid IDs

        self.status = "NO network loaded"
        self.network_ok = False
        self.type_loaded = False
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

    def create_from_geography(
            self,
            geo_file: str,
            id_field: str,
            dir_field: str,
            cost_field: str,
            centroids: np.ndarray,
            skim_fields=[],
            anode="A_NODE",
            bnode="B_NODE",
    ) -> None:
        """
        Creates a graph from a Shapefile. (Deprecated)

        Args:
            geo_file (:obj:`str`): Path to the geographic file to be used. File is usually the output of the network
                                   preparation tool from the AequilibraE plugin for QGIS

            id_field (:obj:`str`): Name of the field that has link IDs (must be unique)

            dir_field (:obj:`str`): Name of the field that has the link directions of flow ([-1, 0, 1])

            cost_field (:obj:`str`): Name of the field that has the cost data (field to minimized in shortest path)

            centroids (:obj:`np.ndarray`): Numpy Array with a list of centroids included in this graph

            skim_fields (:obj:`list`): List with the name of fields to be skimmed

            anode (:obj:`str`): Name of the field with information of A_Node of links (if different than *A_NODE*)

            bnode (:obj:`str`): Name of the field with information of B_Node of links (if different than *B_NODE*)
        """

        import shapefile

        cost_field_name = cost_field
        error = None
        geo_file_records = shapefile.Reader(geo_file)
        records = geo_file_records.records()

        def find_field_index(fields, field_name):
            for i, f in enumerate(fields):
                if f[0] == field_name:
                    return i - 1

            f = [str(x[0]) for x in fields]
            raise ValueError(field_name + " does not exist. Fields available are: " + ", ".join(f))

        # collect the fields in the network
        check_titles = [id_field, dir_field, anode, bnode, cost_field]
        id_field = find_field_index(geo_file_records.fields, id_field)
        dir_field = find_field_index(geo_file_records.fields, dir_field)
        cost_field = find_field_index(geo_file_records.fields, cost_field)
        anode = find_field_index(geo_file_records.fields, anode)
        bnode = find_field_index(geo_file_records.fields, bnode)

        # Appends all fields to the list of fields to be used
        all_types = [
            self.__integer_type,
            self.__integer_type,
            self.__integer_type,
            self.__float_type,
            np.int8,
        ]
        all_titles = [
            "link_id",
            "a_node",
            "b_node",
            cost_field_name,
            "direction",
        ]
        check_fields = [id_field, dir_field, anode, bnode, cost_field]
        types_to_check = [int, int, int, int, float]

        # Loads the skim index fields
        dict_field = {}
        for k in skim_fields:
            skim_index = find_field_index(geo_file_records.fields, k)
            check_fields.append(skim_index)
            check_titles.append(k)
            types_to_check.append(float)

            all_types.append(self.__float_type)
            all_titles.append((k))
            dict_field[k] = skim_index

        dt = [(t, d) for t, d in zip(all_titles, all_types)]

        # Check ID uniqueness and if there are any non-valid values
        all_ids = []
        for feat in records:
            for i, j in enumerate(check_fields):
                k = feat[j]
                if not isinstance(k, types_to_check[i]):
                    error = check_titles[i], "field has wrong type or empty values"
                    break
            all_ids.append(feat[check_fields[0]])
            if error is not None:
                break

        if error is None:
            # Checking uniqueness
            all_ids = np.array(all_ids, np.int)
            y = np.bincount(all_ids)
            if np.max(y) > 1:
                error = "IDs are not unique."

        if error is None:
            data = []

            for feat in records:
                line = []
                line.append(feat[id_field])
                line.append(feat[anode])
                line.append(feat[bnode])
                line.append(feat[cost_field])
                line.append(feat[dir_field])

                # We append the skims now
                for k in all_titles:
                    if k in dict_field:
                        line.append(feat[dict_field[k]])
                data.append(line)

            network = np.asarray(data)
            del data

            self.network = np.zeros(network.shape[0], dtype=dt)
            for k, t in enumerate(dt):
                self.network[t[0]] = network[:, k].astype(t[1])
            del network

            self.type_loaded = "SHAPEFILE"
            self.status = "OK"
            self.network_ok = True
            self.prepare_graph(centroids.astype(np.int64))
            self.__source__ = geo_file
            self.__field_name__ = None
            self.__layer_name__ = None
        if error is not None:
            raise ValueError(error)

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

        if not self.network_ok:
            raise ValueError("Network not yet properly loaded")
        else:
            all_titles = self.network.dtype.names

            if self.status == "OK":
                negs = self.network[self.network["direction"] == -1]
                poss = self.network[self.network["direction"] == 1]
                zers = self.network[self.network["direction"] == 0]

                self.num_links = negs.shape[0] + poss.shape[0]
                self.num_links += zers.shape[0] * 2

                dtype = self.__build_dtype(all_titles)

                self.graph = np.zeros(self.num_links, dtype=dtype)
                a1 = negs.shape[0]
                a2 = a1 + poss.shape[0]
                a3 = a2 + zers.shape[0]
                a4 = a3 + zers.shape[0]

                # Create the graph-specific node numbers
                self.all_nodes = np.unique(np.hstack((self.network["a_node"], self.network["b_node"]))).astype(
                    self.__integer_type
                )
                # We put the centroids as the first N elements of this array
                if self.num_zones:
                    for i in self.centroids:
                        self.all_nodes = np.delete(self.all_nodes, np.argwhere(self.all_nodes == i))

                self.all_nodes = np.hstack((centroids, self.all_nodes)).astype(self.__integer_type)
                self.num_nodes = self.all_nodes.shape[0]
                self.nodes_to_indices = np.empty(int(self.all_nodes.max()) + 1, self.__integer_type)
                self.nodes_to_indices.fill(-1)
                self.nodes_to_indices[self.all_nodes] = np.arange(self.num_nodes)

                for i in all_titles:
                    if i == "link_id":
                        self.graph[i][0:a1] = negs[i]
                        self.graph[i][a1:a2] = poss[i]
                        self.graph[i][a2:a3] = zers[i]
                        self.graph[i][a3:a4] = zers[i]

                    elif i == "a_node":
                        self.graph[i][0:a1] = self.nodes_to_indices[negs["b_node"]]
                        self.graph[i][a1:a2] = self.nodes_to_indices[poss[i]]
                        self.graph[i][a2:a3] = self.nodes_to_indices[zers["b_node"]]
                        self.graph[i][a3:a4] = self.nodes_to_indices[zers[i]]

                    elif i == "b_node":
                        self.graph[i][0:a1] = self.nodes_to_indices[negs["a_node"]]
                        self.graph[i][a1:a2] = self.nodes_to_indices[poss[i]]
                        self.graph[i][a2:a3] = self.nodes_to_indices[zers["a_node"]]
                        self.graph[i][a3:a4] = self.nodes_to_indices[zers[i]]

                    elif i == "direction":
                        self.graph[i][0:a1] = -1
                        self.graph[i][a1:a2] = 1
                        self.graph[i][a2:a3] = -1
                        self.graph[i][a3:a4] = 1
                    else:
                        if i[-3:] == "_ab":
                            self.graph[i[0:-3]][0:a1] = negs[i[0:-3] + "_ba"]
                            self.graph[i[0:-3]][a1:a2] = poss[i]
                            self.graph[i[0:-3]][a2:a3] = zers[i[0:-3] + "_ba"]
                            self.graph[i[0:-3]][a3:a4] = zers[i]
                        elif i[-3:] == "_ba":
                            pass
                        else:
                            if i in all_titles:
                                self.graph[i][0:a1] = negs[i]
                                self.graph[i][a1:a2] = poss[i]
                                self.graph[i][a2:a3] = zers[i]
                                self.graph[i][a3:a4] = zers[i]

                ind = np.lexsort((self.graph["b_node"], self.graph["a_node"]))
                self.graph = self.graph[ind]
                del ind
                self.graph["id"] = np.arange(self.num_links)
                self.fs = np.zeros(self.num_nodes + 1, dtype=self.__integer_type)

                a = self.graph["a_node"][0]
                p = 0
                k = 0

                for i in range(1, self.num_links):
                    if a != self.graph["a_node"][i]:
                        for j in range(p, self.graph["a_node"][i]):
                            self.fs[j + 1] = k
                        p = a
                        a = self.graph["a_node"][i]
                        k = i

                for j in range(p, self.num_nodes):
                    self.fs[j + 1] = k

                self.fs[self.num_nodes] = self.graph.shape[0]
                self.ids = self.graph["id"]
                self.b_node = np.array(self.graph["b_node"], self.__integer_type)
                for i in self.graph.dtype.names:
                    if np.any(np.isnan(self.graph[i])):
                        warn(f'Field {i} has at least one NaN value.  Your computation may be compromised')

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
                    if i[:-3] + '_ba' in all_titles:
                        dtype.append((i[:-3], self.network[i].dtype))
                    else:
                        raise ValueError('Field {} exists for ab direction but does not exist for ba'.format(i))
                elif i[-3:] == "_ba":
                    if i[:-3] + '_ab' not in all_titles:
                        raise ValueError('Field {} exists for ba direction but does not exist for ab'.format(i))
                else:
                    dtype.append((i, self.network[i].dtype))
        return dtype

    def set_graph(self, cost_field) -> None:
        """
        Sets the field to be used for path computation

        Args:
            cost_field (:obj:`str`): Field name. Must be numeric
        """
        if cost_field in self.graph.dtype.names:
            self.cost_field = cost_field
            if self.graph[cost_field].dtype == self.__float_type:
                self.cost = self.graph[cost_field]
            else:
                self.cost = self.graph[cost_field].astype(self.__float_type)
                Warning("Cost field with wrong type. Converting to float64")
        else:
            raise ValueError("cost_field not available in the graph:" + str(self.graph.dtype.names))

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
        k = [x for x in skim_fields if x not in self.graph.dtype.names]
        if k:
            raise ValueError("At least one of the skim fields does not exist in the graph: {}".format(",".join(k)))

        t = [x for x in skim_fields if self.graph[x].dtype != self.__float_type]

        self.skims = np.zeros((self.num_links, len(skim_fields) + 1), self.__float_type)

        if t:
            Warning("Some skim field with wrong type. Converting to float64")
            for i, j in enumerate(skim_fields):
                self.skims[:, i] = self.graph[j].astype(self.__float_type)
        else:
            for i, j in enumerate(skim_fields):
                self.skims[:, i] = self.graph[j]
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
                warn('No centroids in the model. Nothing to block')
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
        mygraph["status"] = self.status
        mygraph["network_ok"] = self.network_ok
        mygraph["type_loaded"] = self.type_loaded
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
            self.status = mygraph["status"]
            self.network_ok = mygraph["network_ok"]
            self.type_loaded = mygraph["type_loaded"]
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

    # We return the list of the fields that are the same for both directions to their initial states
    def __reset_single_fields(self):
        self.required_default_fields = ["link_id", "a_node", "b_node", "direction", "id"]

    # We add a new fields that is the same for both directions
    def __add_single_field(self, new_field):
        if new_field not in self.required_default_fields:
            self.required_default_fields.append(new_field)

    def available_skims(self) -> List[str]:
        """
        Returns graph fields that are available to be set as skims

        Returns:
            *list* (:obj:`str`): Field names
        """
        graph_fields = list(self.graph.dtype.names)
        return [x for x in graph_fields if x not in ["link_id", "a_node", "b_node", "direction", "id"]]

    # We check if all minimum fields are there
    def __network_error_checking__(self):

        # Checking field names
        has_fields = self.network.dtype.names
        must_fields = ["link_id", "a_node", "b_node", "direction"]
        for field in must_fields:
            if field not in has_fields:
                raise ValueError(f'could not find field {field} in the network array')

        # Uniqueness of the id
        link_ids = self.network["link_id"].astype(np.int)
        if link_ids.shape[0] != np.unique(link_ids).shape[0]:
            raise ValueError('"link_id" field not unique')

            # Direction values
        if np.max(self.network["direction"]) > 1 or np.min(self.network["direction"]) < -1:
            raise ValueError('"direction" field not limited to (-1,0,1) values')

    # Needed for when we load the graph directly
    def __graph_error_checking__(self):
        # Checking field names
        self.status = "graph loaded"
        has_fields = self.graph.dtype.names
        must_fields = ["link_id", "a_node", "b_node", "id"]
        for field in must_fields:
            if field not in has_fields:
                self.status = 'could not find field "%s" in the network array' % field

                # checking data types
        must_types = [self.__integer_type, self.__integer_type, self.__integer_type, self.__integer_type]
        for field, ytype in zip(must_fields, must_types):
            if self.graph[field].dtype != ytype:
                self.status = ('Field "%s" in the network array has the wrong type. '
                               'Please refer to the documentation' % field)

        # Uniqueness of the graph id
        a = self.graph["id"].astype(np.int)
        if a.shape[0] != np.unique(a).shape[0]:
            self.status = '"id" field not unique'

        a = np.bincount(self.graph["link_id"].astype(np.int))
        if np.max(a) > 2:
            self.status = '"link_id" field has more than one link per direction'

        if np.min(self.graph["id"]) != 0:
            self.status = '"id" field needs to start in 0 and go to number of links - 1'

        if np.max(self.graph["id"]) > self.graph["id"].shape[0] - 1:
            self.status = '"id" field needs to start in 0 and go to number of links - 1'

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
