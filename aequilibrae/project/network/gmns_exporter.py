import pandas as pd
from os.path import join
from ...utils import WorkerThread

from aequilibrae.parameters import Parameters


class GMNSExporter(WorkerThread):
    def __init__(self, net, path) -> None:
        WorkerThread.__init__(self, None)
        self.p = Parameters()
        self.links = net.links
        self.nodes = net.nodes
        self.source = net.source
        self.conn = net.conn
        self.output_path = path

        self.gmns_par = self.p.parameters["network"]["gmns"]
        self.gmns_l = self.gmns_par["link"]
        self.gmns_n = self.gmns_par["node"]

    def doWork(self):

        l_equiv = self.gmns_l["equivalency"]
        n_equiv = self.gmns_n["equivalency"]

        links_df = self.links.data
        nodes_df = self.nodes.data

        if "ogc_fid" in list(links_df.columns):
            links_df.drop("ogc_fid", axis=1, inplace=True)
        if "ogc_fid" in list(nodes_df.columns):
            nodes_df.drop("ogc_fid", axis=1, inplace=True)

        two_way_cols = list(set([col[:-3] for col in list(links_df.columns) if col[-3:] in ["_ab", "_ba"]]))
        for idx, row in links_df.iterrows():

            if row.direction == 0:
                links_df = pd.concat([links_df, links_df.loc[idx:idx, :]], axis=0)
                links_df.reset_index(drop=True, inplace=True)

                links_df.loc[links_df.index[-1], "link_id"] = max(list(links_df.link_id)) + 1
                links_df.loc[links_df.index[-1], "a_node"] = row.b_node
                links_df.loc[links_df.index[-1], "b_node"] = row.a_node

                links_df.loc[links_df.index[-1], "direction"] = 1
                links_df.loc[idx, "direction"] = 1

                links_df.loc[links_df.index[-1], "dir_flag"] = -1
                links_df.loc[idx, "dir_flag"] = 1

                for col in two_way_cols:
                    links_df.loc[idx, col] = links_df.loc[idx, col + "_ab"]
                    links_df.loc[links_df.index[-1], col] = links_df.loc[idx, col + "_ba"]

            elif row.direction == -1:
                for col in two_way_cols:
                    links_df.loc[idx, col] = links_df.loc[idx, col + "_ba"]

                links_df.loc[idx, "a_node"] = row.b_node
                links_df.loc[idx, "b_node"] = row.a_node
                links_df.loc[idx, "direction"] = 1
                links_df.loc[idx, "dir_flag"] = -1

            else:
                for col in two_way_cols:
                    links_df.loc[idx, col] = links_df.loc[idx, col + "_ab"]

                links_df.loc[idx, "dir_flag"] = 1

        links_df.distance = links_df.distance.apply(lambda x: x / 1000)

        for col in list(links_df.columns):
            if col in l_equiv:
                if l_equiv[col] not in list(links_df.columns):
                    links_df.rename(columns={f"{col}": f"{l_equiv[col]}"}, inplace=True)
                elif col not in ["lanes", "capacity", "link_id", "name", "geometry"]:
                    links_df.drop(col, axis=1, inplace=True)
            elif col[-3:] in ["_ab", "_ba"]:
                links_df.drop(col, axis=1, inplace=True)

        for idx, row in nodes_df.iterrows():
            nodes_df.loc[idx, "node_type"] = "centroid" if row.is_centroid == 1 else None

        for col in list(nodes_df.columns):
            if col in n_equiv:
                if n_equiv[col] not in list(nodes_df.columns):
                    nodes_df.rename(columns={f"{col}": f"{n_equiv[col]}"}, inplace=True)
                elif col != "node_id":
                    links_df.drop(col, axis=1, inplace=True)
            elif col == "geometry":
                nodes_df = nodes_df.assign(
                    x_coord=[nodes_df.geometry[idx].coords[0][0] for idx in list(nodes_df.index)]
                )
                nodes_df = nodes_df.assign(
                    y_coord=[nodes_df.geometry[idx].coords[0][1] for idx in list(nodes_df.index)]
                )
                nodes_df.drop("geometry", axis=1, inplace=True)

        link_cols = list(links_df.columns)
        link_req = [k for k in self.gmns_l["fields"] if self.gmns_l["fields"][k]["required"]]
        main_cols = ["link_id", "from_node_id", "to_node_id", "directed"]
        link_cols = (
            main_cols
            + [c for c in link_cols if c in link_req and c not in main_cols]
            + [c for c in link_cols if c not in link_req]
        )

        node_cols = list(nodes_df.columns)
        node_req = [k for k in self.gmns_n["fields"] if self.gmns_n["fields"][k]["required"]]
        main_cols = ["node_id", "x_coord", "y_coord"]
        node_cols = (
            main_cols
            + [c for c in node_cols if c in node_req and c not in main_cols]
            + [c for c in node_cols if c not in node_req]
        )

        links_df = links_df[link_cols]
        nodes_df = nodes_df[node_cols]

        links_df.to_csv(join(self.output_path, "link.csv"), index=False)
        nodes_df.to_csv(join(self.output_path, "node.csv"), index=False)

        # Getting use definition table

        fields_dict = self.gmns_par["use_definition"]["equivalency"]
        cur = self.conn.execute(
            "select mode_name, mode_id, description, pce, vot, persons_per_vehicle from modes"
        ).fetchall()
        modes_df = pd.DataFrame(
            cur, columns=["mode_name", "mode_id", "description", "pce", "vot", "persons_per_vehicle"]
        )

        modes_df = modes_df[["mode_name", "persons_per_vehicle", "pce", "description", "mode_id"]].rename(
            columns={"mode_name": fields_dict["mode_name"]}
        )
        modes_df.to_csv(join(self.output_path, "use_definition.csv"), index=False)
