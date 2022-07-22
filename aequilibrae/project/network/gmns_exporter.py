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
        self.gmns_l_fields = self.gmns_par["link_fields"]
        self.gmns_n_fields = self.gmns_par["node_fields"]

    def doWork(self):

        gmns_par = self.gmns_par
        l_fields = self.gmns_l_fields
        n_fields = self.gmns_n_fields

        links_df = self.links.data
        nodes_df = self.nodes.data

        if 'ogc_fid' in list(links_df.columns):
            links_df.drop('ogc_fid', axis=1, inplace=True)
        if 'ogc_fid' in list(nodes_df.columns):
            nodes_df.drop('ogc_fid', axis=1, inplace=True)

        two_way_cols = list(set([col[:-3] for col in list(links_df.columns) if col[-3:] in ['_ab', '_ba']]))
        for idx, row in links_df.iterrows():

            if row.direction == 0:
                links_df = pd.concat([links_df, links_df.loc[idx:idx, :]], axis=0)
                links_df.reset_index(drop=True, inplace=True)

                links_df.loc[links_df.index[-1], 'link_id'] = max(list(links_df.link_id)) + 1
                links_df.loc[links_df.index[-1], 'a_node'] = row.b_node
                links_df.loc[links_df.index[-1], 'b_node'] = row.a_node

                links_df.loc[links_df.index[-1], 'direction'] = 1
                links_df.loc[idx, 'direction'] = 1

                links_df.loc[links_df.index[-1], 'dir_flag'] = -1
                links_df.loc[idx, 'dir_flag'] = 1

                for col in two_way_cols:
                    links_df.loc[idx, col] = links_df.loc[idx, col + '_ab']
                    links_df.loc[links_df.index[-1], col] = links_df.loc[idx, col + '_ba']

            elif row.direction == -1:
                for col in two_way_cols:
                    links_df.loc[idx, col] = links_df.loc[idx, col + '_ba']
                links_df.loc[idx, 'a_node'] = row.b_node
                links_df.loc[idx, 'b_node'] = row.a_node
                links_df.loc[idx, 'direction'] = 1
                links_df.loc[idx, 'dir_flag'] = -1

            else:
                for col in two_way_cols:
                    links_df.loc[idx, col] = links_df.loc[idx, col + '_ab']
                links_df.loc[idx, 'dir_flag'] = 1

        links_df.distance = links_df.distance.apply(lambda x: x / 1000)
        if 'length' in list(links_df.columns):
            links_df.rename(columns={'length': 'length_aeq'}, inplace=True)

        for col in list(links_df.columns):
            if col in l_fields:
                links_df.rename(columns={f'{col}': f'{l_fields[col]}'}, inplace=True)
            elif col[-3:] in ['_ab', '_ba']:
                links_df.drop(col, axis=1, inplace=True)

        for idx, row in nodes_df.iterrows():
            nodes_df.loc[idx, 'node_type'] = 'centroid' if row.is_centroid == 1 else None

        for col in list(nodes_df.columns):
            if col in n_fields:
                nodes_df.rename(columns={f'{col}': f'{n_fields[col]}'}, inplace=True)
            elif col == 'geometry':
                nodes_df = nodes_df.assign(x_coord=[nodes_df.geometry[idx].coords[0][0] for idx in list(nodes_df.index)])
                nodes_df = nodes_df.assign(y_coord=[nodes_df.geometry[idx].coords[0][1] for idx in list(nodes_df.index)])
                nodes_df.drop('geometry', axis=1, inplace=True)

        link_cols = list(links_df.columns)
        link_cols = gmns_par['required_link_fields'] + [c for c in link_cols if c not in gmns_par['required_link_fields']]

        node_cols = list(nodes_df.columns)
        node_cols = gmns_par['required_node_fields'] + [c for c in node_cols if c not in gmns_par['required_node_fields']]

        links_df = links_df[link_cols]
        nodes_df = nodes_df[node_cols]

        links_df.to_csv(join(self.output_path, 'link.csv'), index=False)
        nodes_df.to_csv(join(self.output_path, 'node.csv'), index=False)
