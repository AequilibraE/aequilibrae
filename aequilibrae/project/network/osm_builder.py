import math
from typing import List
import numpy as np
from .haversine import haversine
from aequilibrae import logger
from aequilibrae.parameters import Parameters
from ...utils import WorkerThread
import importlib.util as iutil
import sqlite3
from ..spatialite_connection import spatialite_connection

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None
if pyqt:
    from PyQt5.QtCore import pyqtSignal

spec = iutil.find_spec("qgis")
isqgis = spec is not None
if isqgis:
    import qgis


class OSMBuilder(WorkerThread):
    if pyqt:
        building = pyqtSignal(object)

    def __init__(self, osm_items: List, path: str, node_start=10000) -> None:
        WorkerThread.__init__(self, None)
        self.osm_items = osm_items
        self.path = path
        self.conn = None
        self.node_start = node_start
        self.report = []
        self.nodes = {}
        self.links = {}
        self.insert_qry = """INSERT INTO {} ({}, geometry) VALUES({}, GeomFromText("{}", 4326))"""

    def __emit_all(self, *args):
        if pyqt:
            self.building.emit(*args)

    def doWork(self):
        if isqgis:
            self.conn = qgis.utils.spatialite_connect(self.path)
        else:
            conn = sqlite3.connect(self.path)
            self.conn = spatialite_connection(conn)
        self.curr = self.conn.cursor()

        node_count = self.data_structures()
        nodes_to_add, node_ids = self.importing_links(node_count)
        self.import_nodes(nodes_to_add, node_ids)

    def data_structures(self):

        osmi = []
        logger.info("Consolidating geo elements")
        self.__emit_all(["text", "Consolidating geo elements"])
        self.__emit_all(["maxValue", len(self.osm_items)])

        for i, x in enumerate(self.osm_items):
            osmi.append(x["elements"])
            self.__emit_all(["Value", i])
        self.osm_items = sum(osmi, [])

        logger.info("Separating nodes and links")
        self.__emit_all(["text", "Separating nodes and links"])
        self.__emit_all(["maxValue", len(self.osm_items)])

        alinks = []
        n = []
        for i, x in enumerate(self.osm_items):
            if x["type"] == "way":
                alinks.append(x)
            elif x["type"] == "node":
                n.append(x)
            self.__emit_all(["Value", i])

        self.osm_items = None
        logger.info("Setting data structures for nodes")
        self.__emit_all(["text", "Setting data structures for nodes"])
        self.__emit_all(["maxValue", len(n)])

        for i, node in enumerate(n):
            nid = node.pop("id")
            _ = node.pop("type")
            self.nodes[nid] = node
            self.__emit_all(["Value", i])
        del n

        logger.info("Setting data structures for links")
        self.__emit_all(["text", "Setting data structures for links"])
        self.__emit_all(["maxValue", len(alinks)])

        all_nodes = []
        for i, link in enumerate(alinks):
            osm_id = link.pop("id")
            _ = link.pop("type")
            all_nodes.extend(link["nodes"])
            self.links[osm_id] = link
            self.__emit_all(["Value", i])
        del alinks

        logger.info("Finalizing data structures")
        self.__emit_all(["text", "Finalizing data structures"])

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

        logger.info("Adding network links")
        self.__emit_all(["text", "Adding network links"])
        L = len(list(self.links.keys()))
        self.__emit_all(["maxValue", L])

        nodes_to_add = set()
        counter = 0
        mode_codes, not_found_tags = self.modes_per_link_type()

        for osm_id, link in self.links.items():
            self.__emit_all(["Value", counter])
            counter += 1
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

            owf, twf = self.field_osm_source()

            # Attributes that are common to all individual links/segments
            vars["direction"] = (linktags.get("oneway") == "yes") * 1

            for k, v in owf.items():
                attr_value = linktags.get(v)
                if isinstance(attr_value, str):
                    attr_value = attr_value.replace('"', "'")
                    attr_value = '"{}"'.format(attr_value)

                vars[k] = attr_value

            for k, v in twf.items():
                val = linktags.get(v["osm_source"])
                for d1, d2 in [("ab", "forward"), ("ba", "backward")]:
                    vars["{}_{}".format(k, d1)] = self.__get_link_property(d2, val, linktags, v)

            vars["modes"] = mode_codes.get(linktags.get("highway"), not_found_tags)

            if len(vars["modes"]) > 0:
                for i in range(segments):
                    geometry, attributes = self.__build_link_data(vars, intersections, i, linknodes, node_ids, fields)
                    sql = self.insert_qry.format(table, fn, attributes, geometry)
                    sql = sql.replace("None", "null")
                    try:
                        self.curr.execute(sql)
                        nodes_to_add.update([linknodes[intersections[i]], linknodes[intersections[i + 1]]])
                    except Exception as e:
                        data = list(vars.values())
                        logger.error("error when inserting link {}. Error {}".format(data, e.args))
                        logger.error(sql)
                    vars["link_id"] += 1
            self.__emit_all(["text", f"{counter:,} of {L:,} super links added"])

        return nodes_to_add, node_ids

    def import_nodes(self, nodes_to_add, node_ids):
        table = "nodes"
        fields = self.get_node_fields()
        field_names = ",".join(fields)
        field_names = ",".join(['"{}"'.format(x) for x in field_names.split(",")])

        logger.info("Adding network nodes")
        self.__emit_all(["text", "Adding network nodes"])
        self.__emit_all(["maxValue", len(nodes_to_add)])

        vars = {}
        for counter, osm_id in enumerate(nodes_to_add):
            self.__emit_all(["Value", counter])
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
        self.__emit_all(["finished_threaded_procedure", 0])

    def __build_link_data(self, vars, intersections, i, linknodes, node_ids, fields):
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

        vars["distance"] = sum(
            [
                haversine(
                    self.nodes[x]["lon"], self.nodes[x]["lat"], self.nodes[y]["lon"], self.nodes[y]["lat"]
                )
                for x, y in zip(all_nodes[1:], all_nodes[:-1])
            ]
        )

        geometry = ["{} {}".format(self.nodes[x]["lon"], self.nodes[x]["lat"]) for x in all_nodes]
        geometry = "LINESTRING ({})".format(", ".join(geometry))

        attributes = [vars.get(x) for x in fields]

        attributes = ", ".join([str(x) for x in attributes])

        return geometry, attributes

    def __get_link_property(self, d2, val, linktags, v):
        vald = linktags.get("{}:{}".format(v["osm_source"], d2), val)
        if vald is not None:
            if vald.isdigit():
                if vald == val and v["osm_behaviour"] == "divide":
                    vald = float(val) / 2
            else:
                if isinstance(vald, str):
                    vald = vald.replace('"', "'")
                    vald = '"{}"'.format(vald)
        return vald

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
    def field_osm_source():
        p = Parameters()
        fields = p.parameters["network"]["links"]["fields"]

        owf = {
            list(x.keys())[0]: x[list(x.keys())[0]]["osm_source"]
            for x in fields["one-way"]
            if "osm_source" in x[list(x.keys())[0]]
        }

        twf = {}
        for x in fields["two-way"]:
            if "osm_source" in x[list(x.keys())[0]]:
                twf[list(x.keys())[0]] = {
                    "osm_source": x[list(x.keys())[0]]["osm_source"],
                    "osm_behaviour": x[list(x.keys())[0]]["osm_behaviour"],
                }
        return owf, twf

    def modes_per_link_type(self):
        p = Parameters()
        modes = p.parameters["network"]["osm"]["modes"]

        cursor = self.conn.cursor()
        cursor.execute("SELECT mode_name, mode_id from modes")
        mode_codes = cursor.fetchall()
        mode_codes = {p[0]: p[1] for p in mode_codes}

        type_list = {}
        notfound = ""
        for mode, val in modes.items():
            all_types = val["link_types"]
            md = mode_codes[mode]
            for tp in all_types:
                type_list[tp] = "{}{}".format(type_list.get(tp, ""), md)
            if val["unknown_tags"]:
                notfound += md

        type_list = {k: '"{}"'.format("".join(set(v))) for k, v in type_list.items()}

        return type_list, '"{}"'.format(notfound)

    @staticmethod
    def get_node_fields():
        p = Parameters()
        fields = p.parameters["network"]["nodes"]["fields"]
        fields = [list(x.keys())[0] for x in fields]
        return fields + ["osm_id"]
