import pandas as pd
from os.path import join
from ...utils import WorkerThread

from aequilibrae.parameters import Parameters


class GMNSExporter(WorkerThread):
    def __init__(self, net, path) -> None:
        WorkerThread.__init__(self, None)
        self.p = Parameters()
        self.links_df = net.links.data
        self.nodes_df = net.nodes.data
        self.source = net.source
        self.conn = net.conn
        self.output_path = path

        self.gmns_parameters = self.p.parameters["network"]["gmns"]
        self.gmns_links = self.gmns_parameters["link"]
        self.gmns_nodes = self.gmns_parameters["node"]

        cur = self.conn.execute("select mode_name, mode_id, description, pce, vot, ppv from modes").fetchall()
        self.modes_df = pd.DataFrame(cur, columns=["mode_name", "mode_id", "description", "pce", "vot", "ppv"])

    def doWork(self):
        if "ogc_fid" in list(self.links_df.columns):
            self.links_df.drop("ogc_fid", axis=1, inplace=True)
        if "ogc_fid" in list(self.nodes_df.columns):
            self.nodes_df.drop("ogc_fid", axis=1, inplace=True)

        self.update_direction_field()

        # Converting from meters to kilometers
        self.links_df.distance /= 1000

        self.update_field_names()

        self.reorder_fields()

        # Exporting network (links and nodes)
        self.links_df.to_csv(join(self.output_path, "link.csv"), index=False)
        self.nodes_df.to_csv(join(self.output_path, "node.csv"), index=False)

        self.update_modes_fields()

        # Exporting use_definition table
        self.modes_df.to_csv(join(self.output_path, "use_definition.csv"), index=False)

    def update_direction_field(self):
        two_way_cols = list(set([col[:-3] for col in list(self.links_df.columns) if col[-3:] in ["_ab", "_ba"]]))

        ab_links = pd.DataFrame(self.links_df[self.links_df.direction > -1], copy=True)
        ba_links = pd.DataFrame(self.links_df[self.links_df.direction < 1], copy=True)

        # treats ab_links and bi-directionals
        if ab_links.shape[0]:
            ab_links.loc[:, "dir_flag"] = 1
            for col in two_way_cols:
                ab_links.loc[:, col] = ab_links.loc[:, col + "_ab"]

        # treats ba_links and bi-directionals
        if ba_links.shape[0]:
            ba_links.loc[:, "direction"] = 1
            ba_links.loc[:, "dir_flag"] = -1
            b = ba_links.b_node.to_numpy()
            ba_links.loc[:, "b_node"] = ba_links.a_node.to_numpy()[:]
            ba_links.loc[:, "a_node"] = b[:]

            for col in two_way_cols:
                ba_links.loc[:, col] = ba_links.loc[:, col + "_ba"]

        self.links_df = pd.concat([ab_links, ba_links])

    def update_field_names(self):
        """
        Updates field names according to equivalency between AequilibraE and GMNS fields.
        """

        links_equiv = self.gmns_links["equivalency"]
        nodes_equiv = self.gmns_nodes["equivalency"]

        for col in list(self.links_df.columns):
            if col in links_equiv:
                if links_equiv[col] not in list(self.links_df.columns):
                    self.links_df.rename(columns={f"{col}": f"{links_equiv[col]}"}, inplace=True)
                elif col not in ["lanes", "capacity", "link_id", "name", "geometry"]:
                    self.links_df.drop(col, axis=1, inplace=True)
            elif col[-3:] in ["_ab", "_ba"]:
                self.links_df.drop(col, axis=1, inplace=True)

        for idx, row in self.nodes_df.iterrows():
            self.nodes_df.loc[idx, "node_type"] = "centroid" if row.is_centroid == 1 else None

        for col in list(self.nodes_df.columns):
            if col in nodes_equiv:
                if nodes_equiv[col] not in list(self.nodes_df.columns):
                    self.nodes_df.rename(columns={f"{col}": f"{nodes_equiv[col]}"}, inplace=True)
                elif col != "node_id":
                    self.links_df.drop(col, axis=1, inplace=True)
            elif col == "geometry":
                self.nodes_df = self.nodes_df.assign(
                    x_coord=[self.nodes_df.geometry[idx].coords[0][0] for idx in list(self.nodes_df.index)]
                )
                self.nodes_df = self.nodes_df.assign(
                    y_coord=[self.nodes_df.geometry[idx].coords[0][1] for idx in list(self.nodes_df.index)]
                )
                self.nodes_df.drop("geometry", axis=1, inplace=True)

    def reorder_fields(self):
        link_cols = list(self.links_df.columns)
        link_req = [k for k in self.gmns_links["fields"] if self.gmns_links["fields"][k]["required"]]
        main_cols = ["link_id", "from_node_id", "to_node_id", "directed"]
        link_cols = (
            main_cols
            + [c for c in link_cols if c in link_req and c not in main_cols]
            + [c for c in link_cols if c not in link_req]
        )

        node_cols = list(self.nodes_df.columns)
        node_req = [k for k in self.gmns_nodes["fields"] if self.gmns_nodes["fields"][k]["required"]]
        main_cols = ["node_id", "x_coord", "y_coord"]
        node_cols = (
            main_cols
            + [c for c in node_cols if c in node_req and c not in main_cols]
            + [c for c in node_cols if c not in node_req]
        )

        self.links_df = self.links_df[link_cols]
        self.nodes_df = self.nodes_df[node_cols]

    def update_modes_fields(self):
        """
        Updates AequilibraE modes table so it can be exported as a GMNS use_definition table.
        """

        fields_dict = self.gmns_parameters["use_definition"]["equivalency"]
        self.modes_df = self.modes_df[["mode_name", "ppv", "pce", "description", "mode_id"]].rename(
            columns={"mode_name": fields_dict["mode_name"], "ppv": "persons_per_vehicle"}
        )
