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

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None
if pyqt:
    from PyQt5.QtCore import pyqtSignal


class OSMBuilder(WorkerThread):
    if pyqt:
        building = pyqtSignal(object)

    def __init__(self, osm_items: List, file_name: str, node_start=10000) -> None:
        super().__init__(self)
        self.osm_items = osm_items
        self.file_name = file_name
        self.node_start = node_start
        self.conn = None
        self.report = []

    def doWork(self):
        curr = self.conn.cursor()
        self.osm_items = [x["elements"] for x in self.osm_items]
        if len(self.osm_items) > 1:
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

        insert_qry = "INSERT INTO {} ({}, geometry) VALUES({}, {})"

        link_id = 1
        table = "links"
        fields = self.get_link_fields()
        field_names = ",".join(fields)

        for osm_id, link in links.items():
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
                geometry = [linknodes[x] for x in range(ii, jj + 1)]

                node_a = node_ids[linknodes[ii]]
                node_b = node_ids[linknodes[jj]]
            direction = (linktags.get("oneway") == "yes") * 1
            length = sum(
                [
                    haversine(nodes[x]["lon"], nodes[x]["lat"], nodes[y]["lon"], nodes[y]["lat"])
                    for x, y in zip(geometry[1:], geometry[:-1])
                ]
            )
            name = linktags.get("name")

            geometry = ["{} {}".format(nodes[x]["lon"], nodes[x]["lat"]) for x in geometry]
            geometry = "LINESTRING ({})".format(", ".join(geometry))
            link_type = linktags.get("highway")

            lanes = linktags.get("lanes", None)

            if lanes is None:
                lanes_ab = None
                lanes_ba = None
            else:
                lanes_ab = linktags.get("lanes:forward", math.ceil(lanes / 2))
                lanes_ba = linktags.get("lanes:backward", lanes - lanes_ab)

            speed = linktags.get("maxspeed")
            speed_ab = linktags.get("maxspeed:forward", speed)
            speed_ba = linktags.get("maxspeed:backward", speed)

            capacity_ab = None
            capacity_ba = None

            logger.info(
                "{}".format(
                    ",".join(
                        [
                            node_a,
                            node_b,
                            direction,
                            length,
                            link_type,
                            name,
                            lanes_ba,
                            speed_ab,
                            speed_ba,
                            capacity_ab,
                            capacity_ba,
                        ]
                    )
                )
            )
            attributes = [globals()[x] for x in fields]
            sql = insert_qry.format(table, field_names, attributes, geometry)

            try:
                curr.execute(sql)
            except Exception as e:
                logger.info("OSM link ID {} could not be added".format(osm_id))
                logger.error("error {}".format(e.args))
            link_id += 1

        self.conn.commit()
        curr.close()


# def insert_layer(self, conn, table, layer, layer_fields, string_fields, initializable_fields):
#     self.emit_messages(
#         message="Transferring features from " + table + "' layer", value=0, max_val=layer.featureCount()
#     )
#     curr = conn.cursor()
#
#     # We add the Nodes layer
#     field_titles = ", ".join(layer_fields.keys())
#     for j, f in enumerate(layer.getFeatures()):
#         self.emit_messages(value=j)
#         attributes = []
#         for k, val in layer_fields.items():
#             if val < 0:
#                 attributes.append(str(initializable_fields[k]))
#             else:
#                 if k in string_fields:
#                     attributes.append('"' + self.convert_data(f.attributes()[val]) + '"')
#                 else:
#                     attributes.append(self.convert_data(f.attributes()[val]))
#
#         attributes = ", ".join(attributes)
#         sql = "INSERT INTO " + table + " (" + field_titles + ", geometry) "
#         sql += "VALUES (" + attributes + ", "
#         sql += (
#                 "GeomFromText('" + f.geometry().asWkt().upper() + "', " + str(layer.crs().authid().split(":")[1]) + "))"
#         )
#
#         try:
#             a = curr.execute(sql)
#         except:
#             if f.id():
#                 msg = "feature with id " + str(f.id()) + " could not be added to layer " + table
#             else:
#                 msg = "feature with no node id present. It could not be added to layer " + table
#             self.report.append(msg)
#     conn.commit()
#     curr.close()


@staticmethod
def unique_count(a):
    # From: https://stackoverflow.com/a/21124789/1480643
    unique, inverse = np.unique(a, return_inverse=True)
    count = np.zeros(len(unique), np.int)
    np.add.at(count, inverse, 1)
    return np.vstack((unique, count)).T


@staticmethod
def get_link_fields():
    path = os.path.dirname(os.path.realpath(__file__))
    file = os.path.join(path, "network.yml")
    with open(file, "r") as yml:
        fields = yaml.load(yml, Loader=yaml.SafeLoader)
    fields = fields["network"]["links"]["fields"]
    owf = [list(x.keys())[0] for x in fields["one-way"]]

    twf1 = ["{}_ab".format(list(x.keys())[0]) for x in fields["two-way"]]
    twf2 = ["{}_ba".format(list(x.keys())[0]) for x in fields["two-way"]]

    return owf + twf1 + twf2 + ["osm_id"]


@staticmethod
def get_node_fields():
    path = os.path.dirname(os.path.realpath(__file__))
    file = os.path.join(path, "network.yml")
    with open(file, "r") as yml:
        fields = yaml.load(yml, Loader=yaml.SafeLoader)
    fields = fields["network"]["node"]["fields"]
    nf = [list(x.keys())[0] for x in fields]
    return nf + ["osm_id"]
