import gc
import importlib.util as iutil
import sqlite3
import string
from typing import List

import numpy as np
import pandas as pd

from aequilibrae.context import get_active_project
from aequilibrae.parameters import Parameters
from aequilibrae.project.network.link_types import LinkTypes
from aequilibrae.utils.spatialite_utils import connect_spatialite
from .haversine import haversine
from ...utils import WorkerThread

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

    def __init__(self, osm_items: List, path: str, node_start=10000, project=None) -> None:
        WorkerThread.__init__(self, None)
        self.project = project or get_active_project()
        self.logger = self.project.logger
        self.osm_items = osm_items
        self.path = path
        self.conn = None
        self.node_start = node_start
        self.__link_types = None  # type: LinkTypes
        self.report = []
        self.__model_link_types = []
        self.__model_link_type_ids = []
        self.__link_type_quick_reference = {}
        self.nodes = {}
        self.node_df = []
        self.links = {}
        self.insert_qry = """INSERT INTO {} ({}, geometry) VALUES({}, GeomFromText(?, 4326))"""

    def __emit_all(self, *args):
        if pyqt:
            self.building.emit(*args)

    def doWork(self):
        self.conn = connect_spatialite(self.path)
        self.curr = self.conn.cursor()
        self.__worksetup()
        node_count = self.data_structures()
        self.importing_links(node_count)
        self.__emit_all(["finished_threaded_procedure", 0])

    def data_structures(self):
        self.logger.info("Separating nodes and links")
        self.__emit_all(["text", "Separating nodes and links"])
        self.__emit_all(["maxValue", len(self.osm_items)])

        alinks = []
        n = []
        tot_items = len(self.osm_items)
        # When downloading data for entire countries, memory consumption can be quite intensive
        # So we get rid of everything we don't need
        for i in range(tot_items, 0, -1):
            item = self.osm_items.pop(-1)
            if item["type"] == "way":
                alinks.append(item)
            elif item["type"] == "node":
                n.append(item)
            self.__emit_all(["Value", tot_items - i])
        gc.collect()

        self.logger.info("Setting data structures for nodes")
        self.__emit_all(["text", "Setting data structures for nodes"])
        self.__emit_all(["maxValue", len(n)])

        self.node_df = []
        for i, node in enumerate(n):
            nid = node.pop("id")
            _ = node.pop("type")
            node["node_id"] = i + self.node_start
            self.nodes[nid] = node
            self.node_df.append([node["node_id"], nid, node["lon"], node["lat"]])
            self.__emit_all(["Value", i])
        del n
        self.node_df = (
            pd.DataFrame(self.node_df, columns=["A", "B", "C", "D"])
            .drop_duplicates(subset=["C", "D"])
            .to_records(index=False)
        )

        self.logger.info("Setting data structures for links")
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

        self.logger.info("Finalizing data structures")
        self.__emit_all(["text", "Finalizing data structures"])

        node_count = self.unique_count(np.array(all_nodes))

        return node_count

    def importing_links(self, node_count):
        node_ids = {}

        vars = {}
        vars["link_id"] = 1
        table = "links"
        fields = self.get_link_fields()
        self.__update_table_structure()
        field_names = ",".join(fields)

        self.logger.info("Adding network nodes")
        self.__emit_all(["text", "Adding network nodes"])
        sql = "insert into nodes(node_id, is_centroid, osm_id, geometry) Values(?, 0, ?, MakePoint(?,?, 4326))"
        self.conn.executemany(sql, self.node_df)
        self.conn.commit()
        del self.node_df

        self.logger.info("Adding network links")
        self.__emit_all(["text", "Adding network links"])
        L = len(list(self.links.keys()))
        self.__emit_all(["maxValue", L])

        counter = 0
        mode_codes, not_found_tags = self.modes_per_link_type()
        owf, twf = self.field_osm_source()
        all_attrs = []
        all_osm_ids = list(self.links.keys())
        for osm_id in all_osm_ids:
            link = self.links.pop(osm_id)
            self.__emit_all(["Value", counter])
            counter += 1
            if counter % 1000 == 0:
                self.logger.info(f"Creating segments from {counter:,} out of {L:,} OSM link objects")
            vars["osm_id"] = osm_id
            vars["link_type"] = "default"
            linknodes = link["nodes"]
            linktags = link["tags"]

            indices = np.searchsorted(node_count[:, 0], linknodes)
            nodedegree = node_count[indices, 1]

            # Makes sure that beginning and end are end nodes for a link
            nodedegree[0] = 2
            nodedegree[-1] = 2

            intersections = np.where(nodedegree > 1)[0]
            segments = intersections.shape[0] - 1

            # Attributes that are common to all individual links/segments
            vars["direction"] = (linktags.get("oneway") == "yes") * 1

            for k, v in owf.items():
                vars[k] = linktags.get(v)

            for k, v in twf.items():
                val = linktags.get(v["osm_source"])
                if vars["direction"] == 0:
                    for d1, d2 in [("ab", "forward"), ("ba", "backward")]:
                        vars[f"{k}_{d1}"] = self.__get_link_property(d2, val, linktags, v)
                elif vars["direction"] == -1:
                    vars[f"{k}_ba"] = linktags.get(f"{v['osm_source']}:{'backward'}", val)
                elif vars["direction"] == 1:
                    vars[f"{k}_ab"] = linktags.get(f"{v['osm_source']}:{'forward'}", val)

            vars["modes"] = mode_codes.get(linktags.get("highway"), not_found_tags)

            vars["link_type"] = self.__link_type_quick_reference.get(
                vars["link_type"].lower(), self.__repair_link_type(vars["link_type"])
            )

            if len(vars["modes"]) > 0:
                for i in range(segments):
                    attributes = self.__build_link_data(vars, intersections, i, linknodes, node_ids, fields)
                    all_attrs.append(attributes)
                    vars["link_id"] += 1

            self.__emit_all(["text", f"{counter:,} of {L:,} super links added"])
            self.links[osm_id] = []
        sql = self.insert_qry.format(table, field_names, ",".join(["?"] * (len(all_attrs[0]) - 1)))
        self.logger.info("Adding network links")
        self.__emit_all(["text", "Adding network links"])
        try:
            self.curr.executemany(sql, all_attrs)
        except Exception as e:
            self.logger.error("error when inserting link {}. Error {}".format(all_attrs[0], e.args))
            self.logger.error(sql)
            raise e

        self.conn.commit()
        self.curr.close()

    def __worksetup(self):
        self.__link_types = self.project.network.link_types
        lts = self.__link_types.all_types()
        for lt_id, lt in lts.items():
            self.__model_link_types.append(lt.link_type)
            self.__model_link_type_ids.append(lt_id)

    def __update_table_structure(self):
        curr = self.conn.cursor()
        curr.execute("pragma table_info(Links)")
        structure = curr.fetchall()
        has_fields = [x[1].lower() for x in structure]
        fields = [field.lower() for field in self.get_link_fields()] + ["osm_id"]
        for field in [f for f in fields if f not in has_fields]:
            ltype = self.get_link_field_type(field).upper()
            curr.execute(f"Alter table Links add column {field} {ltype}")
        self.conn.commit()

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
                haversine(self.nodes[x]["lon"], self.nodes[x]["lat"], self.nodes[y]["lon"], self.nodes[y]["lat"])
                for x, y in zip(all_nodes[1:], all_nodes[:-1])
            ]
        )

        geometry = ["{} {}".format(self.nodes[x]["lon"], self.nodes[x]["lat"]) for x in all_nodes]
        geometry = "LINESTRING ({})".format(", ".join(geometry))

        attributes = [vars.get(x) for x in fields]
        attributes.append(geometry)
        return attributes

    def __repair_link_type(self, link_type: str) -> str:
        original_link_type = link_type
        link_type = "".join([x for x in link_type if x in string.ascii_letters + "_"]).lower()

        split = link_type.split("_")
        for i, piece in enumerate(split[1:]):
            if piece in ["link", "segment", "stretch"]:
                link_type = "_".join(split[0 : i + 1])

        if len(link_type) == 0:
            link_type = "empty"

        if len(self.__model_link_type_ids) >= 51 and link_type not in self.__model_link_types:
            link_type = "aggregate_link_type"

        if link_type in self.__model_link_types:
            lt = self.__link_types.get_by_name(link_type)
            if original_link_type not in lt.description:
                lt.description += f", {original_link_type}"
                lt.save()
            self.__link_type_quick_reference[original_link_type.lower()] = link_type
            return link_type

        letter = link_type[0]
        if letter in self.__model_link_type_ids:
            letter = letter.upper()
            if letter in self.__model_link_type_ids:
                for letter in string.ascii_letters:
                    if letter not in self.__model_link_type_ids:
                        break
        lt = self.__link_types.new(letter)
        lt.link_type = link_type
        lt.description = f"Link types from Open Street Maps: {original_link_type}"
        lt.save()
        self.__model_link_types.append(link_type)
        self.__model_link_type_ids.append(letter)
        self.__link_type_quick_reference[original_link_type.lower()] = link_type
        return link_type

    def __get_link_property(self, d2, val, linktags, v):
        vald = linktags.get(f'{v["osm_source"]}:{d2}', val)
        if vald is None:
            return vald

        if vald.isdigit():
            if vald == val and v["osm_behaviour"] == "divide":
                vald = float(val) / 2
        return vald

    @staticmethod
    def unique_count(a):
        # From: https://stackoverflow.com/a/21124789/1480643
        unique, inverse = np.unique(a, return_inverse=True)
        count = np.zeros(len(unique), int)
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
    def get_link_field_type(field_name):
        p = Parameters()
        fields = p.parameters["network"]["links"]["fields"]

        if field_name[-3:].lower() in ["_ab", "_ba"]:
            field_name = field_name[:-3]
            for tp in fields["two-way"]:
                if field_name in tp:
                    return tp[field_name]["type"]
        else:
            for tp in fields["one-way"]:
                if field_name in tp:
                    return tp[field_name]["type"]

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

        type_list = {k: "".join(set(v)) for k, v in type_list.items()}

        return type_list, "{}".format(notfound)

    @staticmethod
    def get_node_fields():
        p = Parameters()
        fields = p.parameters["network"]["nodes"]["fields"]
        fields = [list(x.keys())[0] for x in fields]
        return fields + ["osm_id"]
