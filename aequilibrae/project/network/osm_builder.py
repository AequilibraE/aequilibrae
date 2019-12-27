import importlib.util as iutil
import os
import math
from typing import List
import yaml
import numpy as np
from .osm_utils.osm_params import overpass_endpoint, timeout, http_headers
from ...utils import WorkerThread
from .haversine import haversine
from aequilibrae import logger
from aequilibrae.parameters import Parameters

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None
if pyqt:
    from PyQt5.QtCore import pyqtSignal


class OSMBuilder(WorkerThread):
    if pyqt:
        building = pyqtSignal(object)

    def __init__(self, osm_items: List, conn, node_start=10000) -> None:
        super().__init__(self)
        self.osm_items = osm_items
        self.conn = conn
        self.node_start = node_start
        self.report = []

    def doWork(self):
        curr = self.conn.cursor()
        self.osm_items = [x["elements"] for x in self.osm_items]
        self.osm_items = sum(self.osm_items, [])

        alinks = [x for x in self.osm_items if x["type"] == "way"]
        n = [x for x in self.osm_items if x["type"] == "node"]
        self.osm_items = None

        nodes = {}
        for node in n:
            nid = node.pop("id")
            _ = node.pop("type")
            nodes[nid] = node
        del n

        links = {}
        all_nodes = []
        for link in alinks:
            osm_id = link.pop("id")
            _ = link.pop("type")
            all_nodes.extend(link["nodes"])
            links[osm_id] = link
        del alinks
        node_count = self.unique_count(np.array(all_nodes))
        node_ids = {x: i + self.node_start for i, x in enumerate(node_count[:, 0])}

        insert_qry = 'INSERT INTO {} ({}, geometry) VALUES({}, GeomFromText("{}", 4326))'

        vars = {}
        vars["link_id"] = 1
        table = "links"
        fields = self.get_link_fields()
        field_names = ",".join(fields)
        fn = ",".join(['"{}"'.format(x) for x in field_names.split(",")])

        nodes_to_add = set()
        for osm_id, link in links.items():
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

                vars["a_node"] = node_ids[linknodes[ii]]
                vars["b_node"] = node_ids[linknodes[jj]]
                vars["direction"] = (linktags.get("oneway") == "yes") * 1
                vars["length"] = sum(
                    [
                        haversine(nodes[x]["lon"], nodes[x]["lat"], nodes[y]["lon"], nodes[y]["lat"])
                        for x, y in zip(all_nodes[1:], all_nodes[:-1])
                    ]
                )
                vars["name"] = linktags.get("name")
                if vars["name"] is not None:
                    vars["name"] = '"{}"'.format(vars["name"])

                geometry = ["{} {}".format(nodes[x]["lon"], nodes[x]["lat"]) for x in all_nodes]
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
                sql = insert_qry.format(table, fn, attributes, geometry)
                sql = sql.replace("None", "null")
                try:
                    curr.execute(sql)
                    nodes_to_add.update([linknodes[ii], linknodes[jj]])
                except Exception as e:
                    data = list(vars.values())
                    logger.error("error when inserting link {}. Error {}".format(data, e.args))
                    logger.error(sql)
                vars["link_id"] += 1

        table = "nodes"
        fields = self.get_node_fields()
        field_names = ",".join(fields)
        field_names = ",".join(['"{}"'.format(x) for x in field_names.split(",")])

        vars = {}
        for osm_id in nodes_to_add:
            vars["node_id"] = node_ids[osm_id]
            vars["osm_id"] = osm_id
            vars["is_centroid"] = 0
            geometry = "POINT({} {})".format(nodes[osm_id]["lon"], nodes[osm_id]["lat"])

            attributes = [vars.get(x) for x in fields]
            attributes = ", ".join([str(x) for x in attributes])
            sql = insert_qry.format(table, field_names, attributes, geometry)
            sql = sql.replace("None", "null")

            try:
                curr.execute(sql)
            except Exception as e:
                data = list(vars.values())
                logger.error("error when inserting NODE {}. Error {}".format(data, e.args))
                logger.error(sql)

        self.conn.commit()
        curr.close()

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
