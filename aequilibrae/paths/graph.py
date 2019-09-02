import logging
import pickle
import uuid
from datetime import datetime

import numpy as np

from .__version__ import binary_version as VERSION

""""""
"""-----------------------------------------------------------------------------------------------------------
 Package:    AequilibraE

 Name:       Transportation graph class
 Purpose:    Implement a standard graph class to support all network computation

 Original Author:  Pedro Camargo (c@margo.co)
 Contributors:
 Last edited by: Pedro Camargo

 Website:    www.AequilibraE.com
 Repository:  https://github.com/AequilibraE/AequilibraE

 Created:    05/June/2015
 Updated:    03/Dec/2017
 Copyright:   (c) AequilibraE authors
 Licence:     See LICENSE.TXT
 -----------------------------------------------------------------------------------------------------------"""


class Graph(object):
    """
    Graph class
    """

    def __init__(self):
        self.logger = logging.getLogger("aequilibrae")
        self.__integer_type = np.int64
        self.__float_type = np.float64

        self.required_default_fields = []
        self.reset_single_fields()
        self.other_fields = ""
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
        self.skims = False  # 2-D Array with the fields to be computed as skims
        self.skim_fields = False  # List of skim fields to be used in computation
        self.cost_field = False  # Name of the cost field
        self.ids = False  # 1-D Array with link IDs (sequence from 0 to N-1)

        self.block_centroid_flows = False
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
        if tp == "int":
            return self.__integer_type
        elif tp == "float":
            return self.__float_type
        else:
            raise ValueError("It must be either a int or a float")

    # Create a graph from a shapefile. To be upgraded to ANY geographic file in the future
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
        :param geo_file: Path to the geographic file to be used. File is usually the output of the network preparation
                         tool from the AequilibraE plugin for QGIS
        :param id_field: Name of the field that has link IDs (must be unique)
        :param dir_field: Name of the field that has the link directions of flow ([-1, 0, 1])
        :param cost_field: Name of the field that has the cost data (field to minimized in shortest para)
        :param centroids: Numpy Array with a list of centroids included in this graph
        :param skim_fields: List with the name of fields to be skimmed
        :param anode: Name of the field with information of A_Node of links (if different than *A_NODE*)
        :param bnode: Name of the field with information of B_Node of links (if different than *B_NODE*)
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
            return -1

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
            self.__float_type,
            np.int8,
        ]
        all_titles = [
            "link_id",
            "a_node",
            "b_node",
            cost_field_name.lower() + "_ab",
            cost_field_name.lower() + "_ba",
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
            all_types.append(self.__float_type)
            all_titles.append((k + "_ab"))
            all_titles.append((k + "_ba"))
            dict_field[k + "_ab"] = skim_index
            dict_field[k + "_ba"] = skim_index

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
            self.prepare_graph(centroids)
            self.__source__ = geo_file
            self.__field_name__ = None
            self.__layer_name__ = None
        if error is not None:
            raise ValueError(error)

    def prepare_graph(self, centroids: np.ndarray) -> None:
        """
        :param centroids: Array with centroid IDs. Mandatory type Int64, unique and positive
        """

        # Creates the centroids
        if centroids is not None and isinstance(centroids, np.ndarray):
            if np.issubdtype(centroids.dtype, np.integer):
                if centroids.min() <= 0:
                    raise ValueError("Centroid IDs need to be positive")
                else:
                    if centroids.shape[0] != np.unique(centroids).shape[0]:
                        raise ValueError("Centroid IDs are not unique")
                self.centroids = centroids
            else:
                raise ValueError("Centroids need to be an array of integers 64 bits")
        else:
            raise ValueError("Centroids need to be a NumPy array of integers 64 bits")
        self.build_derived_properties()

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

    def __build_dtype(self, all_titles):
        dtype = [
            ("link_id", self.__integer_type),
            ("a_node", self.__integer_type),
            ("b_node", self.__integer_type),
            ("direction", np.int8),
            ("id", self.__integer_type),
        ]
        for i in all_titles:
            if i not in self.required_default_fields and i[0:-3] not in self.required_default_fields:
                if i[-3:] != "_ab":
                    dtype.append((i[0:-3], self.network[i].dtype))
        return dtype

    # We set which are the fields that are going to be minimized in this file
    # TODO: Change the call for all the uses on this function
    def set_graph(self, cost_field=None, skim_fields=False, block_centroid_flows=None):
        """
        :type cost_field
        :type block_centroid_flows
        :type skim_fields: list of fields for skims
        :type self: object
        """
        if block_centroid_flows is not None:
            if isinstance(block_centroid_flows, bool):
                self.set_blocked_centroid_flows(block_centroid_flows)

            else:
                raise ValueError("block_c" "entroid_flows needs to be a boolean")

        if cost_field is not None:
            if cost_field in self.graph.dtype.names:
                self.cost_field = cost_field
                if self.graph[cost_field].dtype == self.__float_type:
                    self.cost = self.graph[cost_field]
                else:
                    self.cost = self.graph[cost_field].astype(self.__float_type)
                    Warning("Cost field with wrong type. Converting to float64")

            else:
                raise ValueError("cost_field not available in the graph:" + str(self.graph.dtype.names))

        if self.cost_field is not None:
            if not skim_fields:
                skim_fields = [self.cost_field]
            else:
                s = [self.cost_field]
                for i in skim_fields:
                    if i in self.graph.dtype.names:
                        if i not in s:
                            s.append(i)
                    else:
                        self.skim_fields = None
                        self.skims = None
                        raise ValueError("Skim", i, " not available in the graph:", self.graph.dtype.names)
                skim_fields = s
        else:
            if skim_fields:
                raise ValueError("Before setting skims, you need to set the cost field")

        t = False
        for i in skim_fields:
            if self.graph[i].dtype != self.__float_type:
                t = True

        self.skims = np.zeros((self.num_links, len(skim_fields) + 1), self.__float_type)

        if t:
            Warning("Some skim field with wrong type. Converting to float64")
            for i, j in enumerate(skim_fields):
                self.skims[:, i] = self.graph[j].astype(self.__float_type)
        else:
            for i, j in enumerate(skim_fields):
                self.skims[:, i] = self.graph[j]
        self.skim_fields = skim_fields

        self.build_derived_properties()
        return True

    def set_blocked_centroid_flows(self, blocking):
        if self.num_zones > 0:
            self.block_centroid_flows = blocking
            self.b_node = np.array(self.graph["b_node"], self.__integer_type)
        else:
            raise ValueError("You can only block flows through centroids after setting the centroids")

    # Procedure to pickle graph and save to disk
    def save_to_disk(self, filename):
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

        with open(filename, "wb") as f:
            pickle.dump(mygraph, f)

    def load_from_disk(self, filename):
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
        self.build_derived_properties()

    def build_derived_properties(self):
        if self.centroids is not None:
            self.num_zones = self.centroids.shape[0]

    # We return the list of the fields that are the same for both directions to their initial states
    def reset_single_fields(self):
        self.required_default_fields = ["link_id", "a_node", "b_node", "direction", "id"]

    # We add a new fields that is the same for both directions
    def add_single_field(self, new_field):
        if new_field not in self.required_default_fields:
            self.required_default_fields.append(new_field)

    def available_skims(self):
        graph_fields = list(self.graph.dtype.names)
        return [x for x in graph_fields if x not in ["link_id", "a_node", "b_node", "direction", "id"]]

    # We check if all minimum fields are there
    def __network_error_checking__(self):

        # Checking field names
        has_fields = self.network.dtype.names
        must_fields = ["link_id", "a_node", "b_node", "direction"]
        for field in must_fields:
            if field not in has_fields:
                self.status = 'could not find field "%s" in the network array' % field

                # checking data types
        must_types = [self.__integer_type, self.__integer_type, self.__integer_type, np.int8]
        for field, ytype in zip(must_fields, must_types):
            if self.network[field].dtype != ytype:
                self.status = (
                    'Field "%s" in the network array has the wrong type. Please refer to the documentation' % field
                )

                # Uniqueness of the id
        link_ids = self.network["link_id"].astype(np.int)
        if link_ids.shape[0] != np.unique(link_ids).shape[0]:
            self.status = '"link_id" field not unique'

            # Direction values
        if np.max(self.network["direction"]) > 1 or np.min(self.network["direction"]) < -1:
            self.status = '"direction" field not limited to (-1,0,1) values'

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
                self.status = (
                    'Field "%s" in the network array has the wrong type. Please refer to the documentation' % field
                )

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
