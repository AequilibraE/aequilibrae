from aequilibrae.paths.graph import GraphBase


class TransitGraph(GraphBase):
    def __init__(self, config: dict = None, od_node_mapping: pd.DataFrame = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = config
        self.od_node_mapping = od_node_mapping
        self.mode = "t"

    @property
    def config(self):
        return self._config
