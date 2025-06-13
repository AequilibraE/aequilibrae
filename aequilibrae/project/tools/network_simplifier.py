import warnings
from copy import deepcopy
from math import ceil
from typing import List

import numpy as np
import pandas as pd
from shapely.geometry.linestring import LineString
from shapely.ops import linemerge
from shapely.ops import substring

from aequilibrae.context import get_active_project
from aequilibrae.paths.graph import Graph
from aequilibrae.utils.aeq_signal import SIGNAL
from aequilibrae.utils.db_utils import commit_and_close
from aequilibrae.utils.interface.worker_thread import WorkerThread


class NetworkSimplifier(WorkerThread):
    signal = SIGNAL(object)

    def __init__(self, project=None) -> None:
        super().__init__(None)

        self.project = project or get_active_project()
        self.network = self.project.network
        self.link_layer = self.network.links.data

        warnings.warn("This will alter your database in place. Make sure you have a backup.")

    def simplify(self, graph: Graph, max_speed_ratio: float = 1.1):
        """
        Simplifies the network by merging links that are shorter than a given threshold

        :Arguments:
            **graph** (:obj:`Graph`): AequilibraE graph

            **max_speed_ratio** (:obj:`float`, *Optional*): Maximum ratio between the fastest
            and slowest speed for a link to be considered for simplification.
        """
        self.graph = graph

        # Creates the sequence of compressed link IDs so we don't have to rebuild through path computation
        compressed_idx, compressed_link_data, _ = self.graph.create_compressed_link_network_mapping()

        link_set_df = self.graph.network.merge(self.graph.graph[["link_id", "__compressed_id__"]], on="link_id")

        # compressed_ids that appear for more than one link
        relevant_compressed_ids = pd.DataFrame(link_set_df.value_counts("__compressed_id__")).query("count>1").index
        link_set_df = link_set_df[link_set_df.__compressed_id__.isin(relevant_compressed_ids)]

        # Makes sure we always get the compressed Id for the same (arbitrary) direction
        link_set_df = link_set_df.sort_values("__compressed_id__").drop_duplicates(subset="link_id", keep="first")

        # We only need one link ID per "super link"
        link_set_df = link_set_df.drop_duplicates(subset="__compressed_id__")
        centroid_connectors = self.graph.network.query("link_id not in @self.link_layer.link_id").link_id.to_numpy()

        links_to_delete, new_links = [], []
        max_link_id = self.link_layer.link_id.max() + 1
        self.signal.emit(["start", link_set_df.shape[0], "Simplifying links"])

        counter = 0
        for _, rec in link_set_df.iterrows():
            counter += 1
            self.signal.emit(["update", counter, "Simplifying links"])
            compressed_id = rec.__compressed_id__

            # We got to the group of links where AequilibraE can no longer compress
            if compressed_id + 1 == compressed_idx.shape[0]:
                continue

            link_sequence = compressed_link_data[compressed_idx[compressed_id] : compressed_idx[compressed_id + 1]]
            if len(link_sequence) < 2:
                continue

            link_sequence = [abs(x) for x in link_sequence]
            self.link_sequence = [x for x in link_sequence if x not in centroid_connectors]
            self.candidates = self.link_layer.query("link_id in @link_sequence").set_index("link_id")

            if self.candidates.shape[0] <= 1:
                continue

            # To merge, all links have to have the same number of lanes and link type
            breaker = self.candidates["link_type"].nunique() > 1
            if breaker:
                continue

            # We build the geometry sequence, speeds, and capacities
            candidates, geos, long_dir, long_lnk = self.__process_link_fields(
                self.candidates, self.link_sequence, max_speed_ratio
            )

            if candidates.empty:
                continue

            new_geo = linemerge(geos)
            if not isinstance(new_geo, LineString):
                warnings.warn(f"Failed to merge geometry for superlink around link {rec.link_id}")
                continue

            break_into = ceil(new_geo.length)

            # Now we build the template for the links we will build
            main_data = long_lnk.to_dict()

            # Some values we will bring from the weighted average
            for field in ["speed_ab", "speed_ba", "capacity_ab", "capacity_ba"]:
                metric = (candidates[field] * candidates["distance"]).sum() / candidates["distance"].sum()
                if long_dir == 1:
                    main_data[field] = metric
                else:
                    field2 = field.replace("ab", "ba") if "ab" in field else field.replace("ba", "ab")
                    main_data[field2] = metric

            # If that link is in the opposite direction, we need to swap lanes,
            # as we would have swapped the geometry as well
            if long_dir == -1:
                main_data["lanes_ab"], main_data["lanes_ba"] = main_data["lanes_ba"], main_data["lanes_ab"]

            for i in range(break_into):
                data = deepcopy(main_data)
                data["link_id"] = max_link_id

                sub_geo = substring(new_geo, i / break_into, (i + 1.0) / break_into, normalized=True)
                if sub_geo.length < 0.000001:
                    raise ValueError("Link with zero length")

                data["geo"] = sub_geo.wkb
                max_link_id += 1
                new_links.append(data)
            links_to_delete.extend(candidates.index.tolist())

        self.signal.emit(["finished"])

        self.project.logger.info(f"{len(links_to_delete):,} links will be removed")
        self.project.logger.info(f"{len(new_links):,} links will be added")
        if new_links:
            self.__execute_link_deletion_and_addition(new_links, links_to_delete)

        self.project.logger.warning("Network has been rebuilt. You should run this tool's rebuild network method")

    def __process_link_fields(self, candidates, link_sequence, max_speed_ratio):
        start_node = candidates.loc[link_sequence[0]]["a_node"]
        longest_link_id = candidates.sort_values("distance", ascending=False).index[0]
        speed_ab, speed_ba, geos, lanes_ab, lanes_ba = [], [], [], [], []
        longest_link, longest_direction = None, None
        for link_id in link_sequence:
            link = candidates.loc[link_id]
            direction = "AB" if start_node == link.a_node else "BA"
            geos.append(link.geometry if direction == "AB" else link.geometry.reverse())
            speed_ab.append(link.speed_ab if direction == "AB" else link.speed_ba)
            speed_ba.append(link.speed_ba if direction == "AB" else link.speed_ab)
            lanes_ab.append(link.lanes_ab if direction == "AB" else link.lanes_ba)
            lanes_ba.append(link.lanes_ba if direction == "AB" else link.lanes_ab)
            if link_id == longest_link_id:
                # We use the longest link as a template
                longest_direction = 1 if direction == "AB" else -1
                longest_link = link
            start_node = link.b_node if direction == "AB" else link.a_node

        constraints_broken = int(np.unique(lanes_ab).shape[0] > 1)
        constraints_broken += np.unique(lanes_ba).shape[0] > 1

        # Speeds cannot diverge by more than 10%
        constraints_broken += max(speed_ab) / max(min(speed_ab), 0.00001) > max_speed_ratio
        constraints_broken += max(speed_ba) / max(min(speed_ba), 0.00001) > max_speed_ratio
        if constraints_broken > 0:
            return pd.DataFrame([]), None, None, None

        return candidates, geos, longest_direction, longest_link

    def __execute_link_deletion_and_addition(self, new_links, links_to_delete):
        df = pd.DataFrame(new_links)
        df = df.drop(columns=["a_node", "b_node", "geometry", "ogc_fid"]).rename({"geo": "geometry"}, axis=1)
        cols = list(df.columns)
        df = df[cols]
        data = df.assign(srid=self.link_layer.crs.to_epsg()).to_records(index=False)

        sql = f"INSERT INTO links({','.join(df.columns)}) VALUES ({','.join(['?'] * (len(df.columns) - 1))},GeomFromWKB(?, ?))"
        with commit_and_close(self.project.path_to_file, spatial=True) as conn:
            conn.executemany(sql, data)
            conn.executemany("DELETE FROM links WHERE link_id=?", [[x] for x in links_to_delete])
            conn.commit()

        # Validate that we kept distances the same
        old_dist = self.link_layer.geometry.length.sum()
        new_layer = self.network.links
        new_layer.refresh()
        new_dist = new_layer.data.geometry.length.sum()

        self.project.logger.warning(
            f"Old distance: {old_dist}, new distance: {new_dist}. Difference: {old_dist - new_dist}"
        )
        self.link_layer = new_layer.data

    def collapse_links_into_nodes(self, links: List[int]):
        """
        Collapses links into nodes, adjusting the network in the neighborhood.

        :Arguments:
            **links** (:obj:`List[int]`): List containing link IDs to be collapsed.
        """
        srid = self.link_layer.crs.to_epsg()
        target_links = self.link_layer.query("link_id in @links")
        with commit_and_close(self.project.path_to_file, spatial=True) as conn:
            for _, link in target_links.iterrows():
                wkb = link.geometry.interpolate(0.5, normalized=True).wkb
                conn.execute("DELETE FROM links WHERE link_id=?", [link.link_id])
                conn.commit()
                conn.execute("UPDATE nodes SET geometry=GeomFromWKB(?, ?) WHERE node_id=?", [wkb, srid, link.a_node])
                conn.execute("UPDATE nodes SET geometry=GeomFromWKB(?, ?) WHERE node_id=?", [wkb, srid, link.b_node])
                conn.commit()

        self.link_layer = self.network.links.data
        self.project.logger.warning(f"{len(links)} links collapsed into nodes")

    def rebuild_network(self):
        """Rebuilds the network elements that would have to be rebuilt after massive network simplification"""

        self.network.links.refresh()
        self.network.nodes.refresh()

        with commit_and_close(self.project.path_to_file, spatial=True) as conn:
            conn.execute("VACUUM")
