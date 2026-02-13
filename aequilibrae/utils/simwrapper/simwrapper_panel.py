import sqlite3

import numpy as np

from aequilibrae.utils.simwrapper.simwrapper_utils import pretty_round


class SimwrapperPanel:
    """Base class for all simwrapper panels

    :Arguments:
        **type** (:obj:`str`): panel type
        **title** (:obj:`str`): title to show in the dashboard
        **height** (:obj:`int`, *Optional*): panel height
        **width** (:obj:`int`, *Optional*): panel width

    :Example:
        panel = SimwrapperPanel("text", "My Panel", height=3, width=6)
    """

    def __init__(self, type, title, height=None, width=None):
        self.type = type
        self.title = title
        self.height = height
        self.width = width

    def to_dict(self):
        """Returns dictionary representation of the panel"""
        panel = {
            "type": self.type,
            "title": self.title,
        }

        if self.height:
            panel["height"] = self.height

        if self.width:
            panel["width"] = self.width

        return panel

    def _add_if_set(self, panel: dict, **kwargs):
        """Add key/value pairs to `panel` only when the value is truthy.

        Keeps callers concise and preserves original behaviour that
        omitted empty/falsey values (empty dict/list/None/0/"").
        """
        for k, v in kwargs.items():
            if v:
                panel[k] = v
        return panel


class ConvergencePanel(SimwrapperPanel):
    def __init__(self, title, config, height=None, width=None):
        super().__init__("vega", title, height=height, width=width)
        self.config = config

    def to_dict(self):
        """Returns dictionary representation of the panel."""
        panel = super().to_dict()
        self._add_if_set(panel, config=self.config)
        return panel


class TilePanel(SimwrapperPanel):
    """
    Panel used to display tabular summary statistics.

    :Arguments:
        **title** (:obj:`str`): title
        **dataset** (:obj:`str`): path to csv dataset used by tile
        **height** (:obj:`int`, *Optional*): panel height
        **width** (:obj:`int`, *Optional*): panel width

    :Example:
        panel = TilePanel("Summary Statistics", "data/summary.csv", height=3)
    """

    def __init__(self, title, dataset, height=None, width=None, colors=None):
        super().__init__("tile", title, height=height, width=width)
        self.dataset = dataset
        self.colors = colors

    def to_dict(self):
        """Returns dictionary representation of the panel."""
        panel = super().to_dict()
        panel["dataset"] = self.dataset
        self._add_if_set(panel, colors=self.colors)
        return panel


class TextPanel(SimwrapperPanel):
    """
    Panel for displaying text content.

    :Arguments:
        **title** (:obj:`str`): title
        **data** (:obj:`str`): text content or file path
        **is_file** (:obj:`bool`, *Optional*): if data is a file reference
        **height** (:obj:`int`, *Optional*): panel height
        **width** (:obj:`int`, *Optional*): panel width

    :Example:
        panel = TextPanel("Overview", "text/overview.md", is_file=True)
    """

    def __init__(self, title, data, is_file=False, height=None, width=None):
        super().__init__("text", title, height=height, width=width)
        self.data = data
        self.is_file = is_file

    def to_dict(self):
        """Returns dictionary representation of the panel."""
        panel = super().to_dict()
        if self.is_file:
            self._add_if_set(panel, file=self.data)
        else:
            self._add_if_set(panel, content=self.data)
        return panel


class AequilibraEMapPanel(SimwrapperPanel):
    """
    Panel for rendering interactive AequilibraE network maps.

    :Arguments:
        **title** (:obj:`str`): title
        **database** (:obj:`str`, *Optional*): project database
        **height** (:obj:`int`, *Optional*): panel height
        **width** (:obj:`int`, *Optional*): panel width
        **center** (:obj:`list`, *Optional*): map center coordinates
        **zoom** (:obj:`int`, *Optional*): initial zoom level
        **projection** (:obj:`str`, *Optional*): coordinate reference system

    :Example:
        panel = AequilibraEMapPanel(title, database, view, height, width, center, zoom, projection)
    """

    def __init__(
        self,
        title,
        database="project_database.sqlite",
        height=None,
        width=None,
        center=None,
        zoom=None,
        projection=None,
    ):
        super().__init__("aequilibrae", title, height=height, width=width)

        self.database = database
        self.center = center
        self.zoom = zoom
        self.projection = projection

        self.defaults = None

        self.extra_databases = None
        self.layers = {}
        self.legend = None

    def set_defaults(self, defaults_dict=None):
        """Sets default visuals for map layers"""
        if defaults_dict:
            available_keys = {"fillColor", "lineColor", "lineWidth", "pointRadius"}
            assert not available_keys ^ defaults_dict.keys(), (
                "Defaults dictionary can only contain the following keys: " + ", ".join(available_keys)
            )

            self.defaults = defaults_dict
        else:
            self.defaults = {
                "fillColor": "#00ffef",
                "lineColor": "#ffff00",
                "lineWidth": 500,
                "pointRadius": 20,
            }

    def add_layer(self, name, layer_dict):
        """Adds a layer definition under the given name"""
        self.layers[name] = layer_dict

    def set_legend(self, legend_list):
        """Sets legend configuration for the map"""
        self.legend = legend_list

    def set_extra_databases(self, database_dict):
        """Registers extra databases used by map"""
        self.extra_databases = database_dict

    def to_dict(self):
        """Returns dictionary representation of the panel"""
        panel = super().to_dict()

        panel["database"] = self.database

        self._add_if_set(
            panel,
            center=self.center,
            zoom=self.zoom,
            projection=self.projection,
            defaults=self.defaults,
            legend=self.legend,
            extraDatabases=self.extra_databases,
            layers=self.layers,
        )

        return panel


class AequilibraEResultsMapPanel(AequilibraEMapPanel):
    """
    Panel for rendering interactive AequilibraE network maps with results layers.

    :Arguments:
        **title** (:obj:`str`): title
        **project** (:obj:`Project`, *Optional*): AequilibraE project, used to compute data ranges from percentiles.
            If not provided, data ranges default to [0, 1].
    """

    def __init__(
        self,
        title,
        project=None,
        project_database="project_database.sqlite",
        results_database="results_database.sqlite",
        colour_metric=None,
        width_metric=None,
        height=None,
        width=None,
        center=None,
        zoom=None,
        projection=None,
        palette="Temps",
        results_table=None,
    ):
        super().__init__(
            title, project_database, height=height, width=width, center=center, zoom=zoom, projection=projection
        )

        self.results_database = results_database
        self.colour_metric = colour_metric
        self.width_metric = width_metric
        self.palette = palette
        self.results_table = results_table

        super().set_extra_databases({"results": self.results_database})

        colour_range = self._compute_data_range(project, self.colour_metric)
        width_range = self._compute_data_range(project, self.width_metric)

        super().set_legend(self.build_legend(colour_range, width_range))

        self.set_colour_styling(colour_range)
        self.set_width_styling(width_range)

        self.add_layer()

    def _compute_data_range(self, project, metric, lower_pct=5, upper_pct=95):
        """Compute a pretty-rounded data range from the 5th and 95th percentiles.

        :Arguments:
            **project**: AequilibraE project with results_connection
            **metric** (:obj:`str`): column name to compute range for
            **lower_pct** (:obj:`int`): lower percentile (default 5)
            **upper_pct** (:obj:`int`): upper percentile (default 95)

        :Returns:
            **list**: [lower_bound, upper_bound] rounded to pretty numbers
        """
        if not metric or not project or not self.results_table:
            return [0, 1]

        try:
            with project.results_connection as conn:
                cursor = conn.execute(f"SELECT [{metric}] FROM [{self.results_table}] WHERE [{metric}] IS NOT NULL")
                values = np.array([row[0] for row in cursor.fetchall()], dtype=float)
        except (sqlite3.OperationalError, Exception):
            return [0, 1]

        if len(values) == 0:
            return [0, 1]

        p_low = float(np.percentile(values, lower_pct))
        p_high = float(np.percentile(values, upper_pct))

        # If the range is essentially zero, return a trivial range
        if p_high - p_low < 1e-12:
            return [0, max(1, pretty_round(p_high, "up"))]

        lower = pretty_round(p_low, "down")
        upper = pretty_round(p_high, "up")

        # Snap small values near zero to zero
        span = upper - lower
        if span > 0 and abs(lower) / span < 0.05:
            lower = 0

        return [lower, upper]

    def build_legend(self, colour_range, width_range):
        legend = []

        if self.width_metric:
            w_min, w_max = width_range
            w_mean = (w_min + w_max) / 2
            legend.append({"subtitle": self.width_metric})
            legend.append({"label": str(w_min), "color": "#444444", "size": 1, "shape": "line"})
            legend.append({"label": str(w_mean), "color": "#444444", "size": 5, "shape": "line"})
            legend.append({"label": str(w_max), "color": "#444444", "size": 10, "shape": "line"})

        if self.colour_metric:
            c_min, c_max = colour_range
            c_mean = (c_min + c_max) / 2
            legend.append({"subtitle": self.colour_metric})
            legend.append({"label": str(c_min), "color": "#009392", "size": 5, "shape": "line"})
            legend.append({"label": str(c_mean), "color": "#e9e29c", "size": 5, "shape": "line"})
            legend.append({"label": str(c_max), "color": "#cf597e", "size": 5, "shape": "line"})

        return legend

    def set_colour_styling(self, data_range):

        if self.colour_metric:
            self.colour_style = {
                "lineColor": {
                    "column": self.colour_metric,
                    "palette": self.palette,
                    "dataRange": data_range,
                }
            }
        else:
            self.colour_style = {"lineColor": "#000000"}

    def set_width_styling(self, data_range):
        if self.width_metric:
            self.width_style = {
                "lineWidth": {
                    "column": self.width_metric,
                    "dataRange": data_range,
                    "widthRange": [10, 250],
                }
            }
        else:
            self.width_style = {"lineWidth": 10}

    def add_layer(self):

        super().add_layer(
            "links",
            {
                "table": "links",
                "geometry": "line",
                "join": {
                    "database": "results",
                    "table": self.results_table,
                    "leftKey": "link_id",
                    "rightKey": "link_id",
                    "type": "left",
                },
                "style": self.colour_style | self.width_style,
            },
        )
