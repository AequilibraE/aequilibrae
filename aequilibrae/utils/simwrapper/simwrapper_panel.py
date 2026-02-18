import sqlite3

import numpy as np

from aequilibrae.utils.simwrapper.simwrapper_utils import pretty_round


class SimwrapperPanel:
    """Base class for all SimWrapper panels.

    Arguments:
        **type** (:obj:`str`): Panel type
        **title** (:obj:`str`): Title to show in the dashboard
        **height** (:obj:`int`, optional): Panel height
        **width** (:obj:`int`, optional): Panel width
    """

    def __init__(self, type, title, height=None, width=None):
        self.type = type
        self.title = title
        self.height = height
        self.width = width

    def to_dict(self):
        """Returns dictionary representation of the panel."""
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
        """Adds key/value pairs to ``panel`` only when the value is truthy."""
        for k, v in kwargs.items():
            if v:
                panel[k] = v
        return panel


class ConvergencePanel(SimwrapperPanel):
    """Panel wrapper for Vega/Vega-Lite specifications."""

    def __init__(self, title, config, height=None, width=None):
        super().__init__("vega", title, height=height, width=width)
        self.config = config

    def to_dict(self):
        """Returns dictionary representation of the panel."""
        panel = super().to_dict()
        self._add_if_set(panel, config=self.config)
        return panel


class TilePanel(SimwrapperPanel):
    """Panel used to display tabular summary statistics.

    Arguments:
        **title** (:obj:`str`): Title
        **dataset** (:obj:`str` or :obj:`list`): CSV path or inline dataset
        **height** (:obj:`int`, optional): Panel height
        **width** (:obj:`int`, optional): Panel width
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
    """Panel for displaying text content.

    Arguments:
        **title** (:obj:`str`): Title
        **data** (:obj:`str`): Text content or file path
        **is_file** (:obj:`bool`, optional): Whether ``data`` is a file reference
        **height** (:obj:`int`, optional): Panel height
        **width** (:obj:`int`, optional): Panel width
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
    """Panel for rendering interactive AequilibraE network maps.

    Arguments:
        **title** (:obj:`str`): Title
        **database** (:obj:`str`, optional): Project database
        **height** (:obj:`int`, optional): Panel height
        **width** (:obj:`int`, optional): Panel width
        **center** (:obj:`list`, optional): Map center coordinates
        **zoom** (:obj:`int`, optional): Initial zoom level
    """

    def __init__(
        self,
        title,
        database="project_database.sqlite",
        height=None,
        width=None,
        center=None,
        zoom=None,
    ):
        super().__init__("aequilibrae", title, height=height, width=width)

        self.database = database
        self.center = center
        self.zoom = zoom

        self.defaults = None

        self.extra_databases = None
        self.layers = {}
        self.legend = None

    def set_defaults(self, defaults_dict=None):
        """Sets default visuals for map layers"""
        if defaults_dict:
            available_keys = {"fillColor", "lineColor", "lineWidth", "pointRadius"}
            # ensure provided keys are a subset of the allowed keys (previous code required an exact match)
            assert set(defaults_dict.keys()).issubset(available_keys), (
                "Defaults dictionary can only contain the following keys: " + ", ".join(sorted(available_keys))
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
            defaults=self.defaults,
            legend=self.legend,
            extraDatabases=self.extra_databases,
            layers=self.layers,
        )

        return panel


class AequilibraEResultsMapPanel(AequilibraEMapPanel):
    """
    Panel for rendering interactive AequilibraE network maps with optional results styling.

    Behavior
    - When `results_table` is provided the panel LEFT JOINs the map `links` layer with that
      table from an extra `results` database and reads metric values from the joined table.
    - When `results_table` is omitted the panel reads metrics directly from the project's
      `links` table in `project_database.sqlite` (no separate results DB/table required).
    - Data ranges used for styling are computed from the lower/upper percentiles (default
      5th/95th) and pretty-rounded with `pretty_round`.
    - If a metric or the corresponding table/column is missing the panel falls back to the
      default data range `[0, 1]` (graceful fallback).

    :Arguments:
        **title** (:obj:`str`): panel title shown in the dashboard
        **project** (:obj:`Project`, *Optional*): open Project instance used to read metric values
            and compute percentile ranges; if not provided ranges default to [0, 1]
        **project_database** (:obj:`str`): main project database filename (default: ``project_database.sqlite``)
        **results_database** (:obj:`str`): results database filename (default: ``results_database.sqlite``);
            used only when ``results_table`` is provided
        **colour_metric** (:obj:`str`, *Optional*): column name used to style line colour. Look-up order:
            results table (if provided) → project `links` table
        **width_metric** (:obj:`str`, *Optional*): column name used to style line width (same lookup rules)
        **results_table** (:obj:`str`, *Optional*): name of the table in the results DB to join to `links`.
            When omitted, metrics are read from the project `links` table instead.
        **palette** (:obj:`str`): colour palette for `colour_metric` scaling (default: ``Temps``)
        **height**, **width**, **center**, **zoom**: visual/layout parameters (see
            :class:`AequilibraEMapPanel`)
        **sql_filter** (:obj:`str`, *Optional*): SQL filter expression applied to the ``links`` layer

    Notes
        - The panel will register an `extraDatabases` entry named ``results`` only when
          ``results_table`` is specified.
        - Ranges are computed from percentiles (5th/95th) and rounded via
          :func:`aequilibrae.utils.simwrapper.simwrapper_utils.pretty_round`.

    Examples:
        - Read a metric from the project ``links`` table
        - Read a metric from a results table and join it to ``links``
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
        palette="Temps",
        results_table=None,
        sql_filter=None,
    ):
        super().__init__(title, project_database, height=height, width=width, center=center, zoom=zoom)

        # inputs
        self.results_database = results_database
        self.colour_metric = colour_metric
        self.width_metric = width_metric
        self.palette = palette
        self.results_table = results_table
        self.sql_filter = sql_filter

        # only register an extra `results` database when a results table is being used
        if self.results_table:
            self.set_extra_databases({"results": self.results_database})

        # compute ranges (supports reading from results DB *or* from project `links` table)
        colour_range = self._compute_data_range(project, self.colour_metric)
        width_range = self._compute_data_range(project, self.width_metric)

        self.set_legend(self.build_legend(colour_range, width_range))

        self.set_colour_styling(colour_range)
        self.set_width_styling(width_range)

        # build the links layer; only add a `join` if results_table is provided
        layer = {
            "table": "links",
            "geometry": "line",
            "style": self.colour_style | self.width_style,
        }

        if self.sql_filter:
            layer["sqlFilter"] = self.sql_filter

        if self.results_table:
            layer["join"] = {
                "database": "results",
                "table": self.results_table,
                "leftKey": "link_id",
                "rightKey": "link_id",
                "type": "left",
            }

        self.add_layer("links", layer)

    def _compute_data_range(self, project, metric, lower_pct=5, upper_pct=95):
        """Compute a pretty-rounded data range from the 5th and 95th percentiles.

        This supports reading the metric either from a results table (when
        ``self.results_table`` is set) or directly from the project `links` table
        (when no results table is provided).

        :Arguments:
            **project**: AequilibraE project with connections
            **metric** (:obj:`str`): column name to compute range for
            **lower_pct** (:obj:`int`): lower percentile (default 5)
            **upper_pct** (:obj:`int`): upper percentile (default 95)

        :Returns:
            **list**: [lower_bound, upper_bound] rounded to pretty numbers
        """
        if not metric or not project:
            return [0, 1]

        try:
            if self.results_table:
                # metric lives in a results DB/table
                with project.results_connection as conn:
                    cursor = conn.execute(f"SELECT [{metric}] FROM [{self.results_table}] WHERE [{metric}] IS NOT NULL")
                    values = np.array([row[0] for row in cursor.fetchall()], dtype=float)
            else:
                # metric comes from the main project `links` table
                with project.db_connection as conn:
                    cursor = conn.execute(f"SELECT [{metric}] FROM links WHERE [{metric}] IS NOT NULL")
                    values = np.array([row[0] for row in cursor.fetchall()], dtype=float)
        except (sqlite3.OperationalError, sqlite3.DatabaseError, ValueError, TypeError):
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
