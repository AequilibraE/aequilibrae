import math
from typing import List
import numpy as np
from .haversine import haversine
from aequilibrae import logger
from aequilibrae.parameters import Parameters
from PyQt5.QtCore import QObject, pyqtSignal


class OSMBuilder(QObject):
    building = pyqtSignal(object)

    def __init__(self, osm_items: List, conn, node_start=10000) -> None:
        QObject.__init__(self, None)
        self.osm_items = osm_items
        self.conn = conn
        self.curr = self.conn.cursor()
        self.node_start = node_start
        self.report = []
        self.nodes = {}
        self.links = {}
        self.insert_qry = 'INSERT INTO {} ({}, geometry) VALUES({}, GeomFromText("{}", 4326))'

    def doWork(self):
        node_count = self.data_structures()
        nodes_to_add, node_ids = self.importing_links(node_count)
        self.import_nodes(nodes_to_add, node_ids)

    def data_structures(self):

        osmi = []
        self.building.emit(["text", "Consolidating geo elements"])
        self.building.emit(["maxValue", len(self.osm_items)])

        for i, x in enumerate(self.osm_items):
            osmi.append(x["elements"])
            self.building.emit(["Value", i])
        self.osm_items = sum(osmi, [])

        self.building.emit(["text", "Separating nodes and links"])
        self.building.emit(["maxValue", len(self.osm_items)])

        alinks = []
        n = []
        for i, x in enumerate(self.osm_items):
            if x["type"] == "way":
                alinks.append(x)
            elif x["type"] == "node":
                n.append(x)
            self.building.emit(["Value", i])

        self.osm_items = None

        self.building.emit(["text", "Setting data structures for nodes"])
        self.building.emit(["maxValue", len(n)])

        for i, node in enumerate(n):
            nid = node.pop("id")
            _ = node.pop("type")
            self.nodes[nid] = node
            self.building.emit(["Value", i])
        del n

        self.building.emit(["text", "Setting data structures for links"])
        self.building.emit(["maxValue", len(alinks)])

        all_nodes = []
        for i, link in enumerate(alinks):
            osm_id = link.pop("id")
            _ = link.pop("type")
            all_nodes.extend(link["nodes"])
            self.links[osm_id] = link
            self.building.emit(["Value", i])
        del alinks

        self.building.emit(["text", "Finalizing data structures"])

        node_count = self.unique_count(np.array(all_nodes))

        return node_count

    def importing_links(self, node_count):
        node_ids = {}

        vars = {}
        vars["link_id"] = 1
        table = "links"
        fields = self.get_link_fields()
        field_names = ",".join(fields)
        fn = ",".join(['"{}"'.format(x) for x in field_names.split(",")])

        self.building.emit(["text", "Adding network links"])
        self.building.emit(["maxValue", len(self.links)])

        nodes_to_add = set()
        counter = 0
        for osm_id, link in self.links.items():
            self.building.emit(["Value", counter])
            vars["osm_id"] = osm_id
            linknodes = link["nodes"]
            linktags = link["tags"]

            indices = np.searchsorted(node_count[:, 0], linknodes)
            nodedegree = node_count[indices, 1]

            # Makes sure that beginning and end are end nodes for a link
            nodedegree[0] = 2
            nodedegree[-1] = 2

            intersections = np.where(nodedegree > 1)[0]
            segments = intersections.shape[0] - 1

            for i in range(segments):
                ii = intersections[i]
                jj = intersections[i + 1]
                all_nodes = [linknodes[x] for x in range(ii, jj + 1)]

                vars["a_node"] = node_ids.get(linknodes[ii], self.node_start)
                if vars["a_node"] == self.node_start:
                    node_ids[linknodes[ii]] = vars["a_node"]
                    self.node_start += 1

                vars["b_node"] = node_ids.get(linknodes[jj], self.node_start)
                if vars["b_node"] == self.node_start:
                    node_ids[linknodes[jj]] = vars["b_node"]
                    self.node_start += 1

                vars["direction"] = (linktags.get("oneway") == "yes") * 1
                vars["length"] = sum(
                    [
                        haversine(
                            self.nodes[x]["lon"], self.nodes[x]["lat"], self.nodes[y]["lon"], self.nodes[y]["lat"]
                        )
                        for x, y in zip(all_nodes[1:], all_nodes[:-1])
                    ]
                )
                vars["name"] = linktags.get("name")
                if vars["name"] is not None:
                    vars["name"] = '"{}"'.format(vars["name"])

                geometry = ["{} {}".format(self.nodes[x]["lon"], self.nodes[x]["lat"]) for x in all_nodes]
                geometry = "LINESTRING ({})".format(", ".join(geometry))
                vars["link_type"] = linktags.get("highway")
                if vars["link_type"] is not None:
                    vars["link_type"] = '"{}"'.format(vars["link_type"])

                lanes = linktags.get("lanes", None)
                if lanes is None:
                    vars["lanes_ab"] = None
                    vars["lanes_ba"] = None
                else:
                    lanes = int(lanes)
                    vars["lanes_ab"] = linktags.get("lanes:forward", math.ceil(lanes / 2))
                    if not isinstance(vars["lanes_ab"], (int, float)):
                        vars["lanes_ab"] = math.ceil(lanes / 2)
                    vars["lanes_ba"] = linktags.get("lanes:backward", lanes - vars["lanes_ab"])
                    if not isinstance(vars["lanes_ba"], (int, float)):
                        vars["lanes_ab"] = lanes - vars["lanes_ab"]

                speed = linktags.get("maxspeed")
                vars["speed_ab"] = linktags.get("maxspeed:forward", speed)
                vars["speed_ba"] = linktags.get("maxspeed:backward", speed)

                if vars["speed_ab"] is not None:
                    vars["speed_ab"] = int(vars["speed_ab"])

                if vars["speed_ba"] is not None:
                    vars["speed_ba"] = int(vars["speed_ab"])

                vars["capacity_ab"] = None
                vars["capacity_ba"] = None

                attributes = [vars[x] for x in fields]

                attributes = ", ".join([str(x) for x in attributes])
                sql = self.insert_qry.format(table, fn, attributes, geometry)
                sql = sql.replace("None", "null")
                try:
                    self.curr.execute(sql)
                    nodes_to_add.update([linknodes[ii], linknodes[jj]])
                except Exception as e:
                    data = list(vars.values())
                    logger.error("error when inserting link {}. Error {}".format(data, e.args))
                    logger.error(sql)
                vars["link_id"] += 1
            counter += 1
        return nodes_to_add, node_ids

    def import_nodes(self, nodes_to_add, node_ids):
        table = "nodes"
        fields = self.get_node_fields()
        field_names = ",".join(fields)
        field_names = ",".join(['"{}"'.format(x) for x in field_names.split(",")])

        self.building.emit(["text", "Adding network nodes"])
        self.building.emit(["maxValue", len(nodes_to_add)])

        vars = {}
        for counter, osm_id in enumerate(nodes_to_add):
            self.building.emit(["Value", counter])
            vars["node_id"] = node_ids[osm_id]
            vars["osm_id"] = osm_id
            vars["is_centroid"] = 0
            geometry = "POINT({} {})".format(self.nodes[osm_id]["lon"], self.nodes[osm_id]["lat"])

            attributes = [vars.get(x) for x in fields]
            attributes = ", ".join([str(x) for x in attributes])
            sql = self.insert_qry.format(table, field_names, attributes, geometry)
            sql = sql.replace("None", "null")

            try:
                self.curr.execute(sql)
            except Exception as e:
                data = list(vars.values())
                logger.error("error when inserting NODE {}. Error {}".format(data, e.args))
                logger.error(sql)

        self.conn.commit()
        self.curr.close()

    @staticmethod
    def unique_count(a):
        # From: https://stackoverflow.com/a/21124789/1480643
        unique, inverse = np.unique(a, return_inverse=True)
        count = np.zeros(len(unique), np.int)
        np.add.at(count, inverse, 1)
        return np.vstack((unique, count)).T

    @staticmethod
    def get_link_fields():
        p = Parameters()
        fields = p.parameters["network"]["links"]["fields"]
        owf = [list(x.keys())[0] for x in fields["one-way"]]

        twf1 = ["{}_ab".format(list(x.keys())[0]) for x in fields["two-way"]]
        twf2 = ["{}_ba".format(list(x.keys())[0]) for x in fields["two-way"]]

        return owf + twf1 + twf2 + ["osm_id"]

    @staticmethod
    def get_node_fields():
        p = Parameters()
        fields = p.parameters["network"]["nodes"]["fields"]
        return fields + ["osm_id"]
