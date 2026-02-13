from pathlib import Path
import yaml
import geopandas as gpd
import pandas as pd
import json
import csv

from aequilibrae.utils.simwrapper.simwrapper_panel import (
    SimwrapperPanel,
    ConvergencePanel,
    TilePanel,
    TextPanel,
    AequilibraEMapPanel,
    AequilibraEResultsMapPanel,
)
from aequilibrae.utils.simwrapper.simwrapper_utils import get_project_center, get_project_zoom


class SimwrapperConfigGenerator:
    """
    Generate simwrapper ready .yaml file from an AequilibraE Project with
    minimal manual work.
    """

    def __init__(self, project, output_dir="simwrapper"):
        """Initialise the config generator and create output directories.

        :Arguments:
            **project** (:obj:`Project`): AequilibraE Project object
            **output_dir** (:obj:`str`, *Optional*): Root directory for SimWrapper outputs
        """
        self.project = project
        self.output_dir = Path(output_dir)
        self.generated_files = {}
        self._create_directories()
        self.center = get_project_center(self.project)
        self.zoom = get_project_zoom(self.project)

    def _create_directories(self):
        """
        Create output directory structure for simwrapper

        Structure:
        PROJECT-DIRECTORY/
            simwrapper_data/    # Data files referenced by configs
                linkstats.csv   # CSV of link properties/metrics
                other_stats.csv # Additional CSV outputs
                ...
            dashboard-*.yaml    # Dashboard configuration file(s)
        """
        self.data_dir = self.output_dir / "simwrapper_data"  # make subcategories

        self.output_dir.mkdir(exist_ok=True)  # base
        self.data_dir.mkdir(exist_ok=True)  # data

    def _find_project_title(self):
        """Generate  project title from the project folder name, otherwise returns "AequilibraE Project" """

        try:
            folder_name = self.project.project_base_path.name
            title = folder_name.replace("_", " ").title()

            if not title.strip():
                raise ValueError

            return title

        except Exception:
            return "AequilibraE Project"

    def _add_to_generated_files(self, key, path):
        """Add file to self.generated_files"""
        self.generated_files[key] = Path(path)

    def _dashboard_skeleton(self):
        """Defines header and layout structure for yaml and returns the basic config skeleton"""

        config = {"header": {"title": self._find_project_title(), "description": "insert description"}, "layout": {}}

        return config

    def _intro_row(self):
        """Returns project details text panel."""

        return [TextPanel(title="title", data="intro")]

    def _get_link_types(self):
        """returns list of link types in network"""
        return self.project.network.link_types.all_types()

    def _categorical_palette(self, n):
        """Returns n visually distinct colors"""

        base = [
            "#4C78A8",
            "#F58518",
            "#E45756",
            "#72B7B2",
            "#54A24B",
            "#EECA3B",
            "#B279A2",
            "#FF9DA6",
            "#9D755D",
        ]
        return base[:n]

    def _truncate_results_tables(self, results_tables, max_tables=3):
        """Return a truncated results list and a flag indicating whether truncation occurred."""

        if len(results_tables) <= max_tables:
            return results_tables, False

        return results_tables[:max_tables], True

    def _results_truncation_notice(self, shown, total):
        return TextPanel(
            title="Please Note",
            data=(
                f"Showing {shown} of {total} result scenarios.\n\n"
                "Additional scenarios were omitted to keep the dashboard readable."
            ),
            height=2,
            width=6,
        )

    def _stats_rows(self):
        """returns stats rows panels"""
        dataset = [
            {
                "key": "Link Count",
                "value": {"database": "project_database.sqlite", "query": "SELECT printf('%,d', COUNT(*)) FROM links"},
            },
            {
                "key": "Node Count",
                "value": {"database": "project_database.sqlite", "query": "SELECT printf('%,d', COUNT(*)) FROM nodes"},
            },
        ]

        panel = TilePanel("Network Size", dataset, height=1, colors="monochrome")

        return [panel]

    def _entire_network_row(self):
        """Builds yaml config for map of entire network"""

        # aequilibrae panel with center and zoom
        panel = AequilibraEMapPanel(
            "Entire Network",
            height=10,
            width=6,
            center=self.center,
            zoom=self.zoom,
            projection="EPSG:32719",
        )

        # set legend
        panel.set_legend(
            [
                {"label": "Regular Links", "color": "#4c72b0", "shape": "line"},
                {"label": "Centroid Connectors", "color": "#9c72b0", "shape": "line"},
                {"label": "Centroid Node", "color": "#FF6600", "shape": "circle"},
                {"label": "Regular Node", "color": "#cacaca", "shape": "circle"},
            ]
        )

        # non-centroid connector links
        panel.add_layer(
            "links_regular",
            {
                "table": "links",
                "geometry": "line",
                "sqlFilter": "link_type != 3",
                "style": {"lineColor": "#4C78A8", "lineWidth": 2},
            },
        )

        # centroid connector links
        panel.add_layer(
            "links_centroid_connectors",
            {
                "table": "links",
                "geometry": "line",
                "sqlFilter": "link_type = 3",
                "style": {
                    "lineColor": "#9c72b0",
                    "lineWidth": 20,
                },
            },
        )

        # add centroid nodes layer
        centroid_node_style = {"fillColor": "#FF6600", "pointRadius": 300}
        panel.add_layer(
            "nodes_centroids",
            {"table": "nodes", "geometry": "point", "sqlFilter": "is_centroid=1", "style": centroid_node_style},
        )

        # add regular nodes layer
        regular_node_style = {"fillColor": "#cacaca", "pointRadius": 100}
        panel.add_layer(
            "nodes_regular",
            {"table": "nodes", "geometry": "point", "sqlFilter": "is_centroid=0", "style": regular_node_style},
        )

        # return panel inside a list
        return [panel]

    def _links_info_row(self):
        """Builds yaml config for panel to show attributes of selected link"""

        link_types = self._get_link_types()

        # if there are no links types map nothing
        if not link_types:
            links = self.project.network.links.data
            link_types = links["link_type"].unique()

        else:
            link_type_names = self.project.network.link_types
            link_types = [link_type_names.get(x).link_type for x in link_types]

        colours = self._categorical_palette(len(link_types))
        colour_map = dict(zip(link_types, colours, strict=True))

        # map panel
        panel = AequilibraEMapPanel("Link Types", height=10, width=6, center=self.center, zoom=self.zoom)

        # build and set legend
        legend = [{"subtitle": "Link Types"}]
        for i, lt in enumerate(link_types):
            legend.append({"label": f"{lt}", "color": f"{colours[i]}", "shape": "line"})

        panel.set_legend(legend)

        # add links layer styled by link type
        panel.add_layer(
            "links",
            {
                "table": "links",
                "geometry": "line",
                "style": {
                    "lineColor": {
                        "column": "link_type",
                        "colors": colour_map,
                    },
                    "lineWidth": 10,
                },
            },
        )

        return [panel]

    def _capacity_map_row(self):
        """Map showing links styled by capacity"""
        panel = AequilibraEMapPanel(title="Link Capacity", height=10, width=6, center=self.center, zoom=self.zoom)

        panel.set_legend(
            [
                {"subtitle": "Link Capacity"},
                {"label": "0 - 1,000", "color": "#2C115F", "size": 2, "shape": "line"},
                {"label": "1,000 - 3,000", "color": "#721F81", "size": 4, "shape": "line"},
                {"label": "3,000 - 6,000", "color": "#B73779", "size": 6, "shape": "line"},
                {"label": "6,000 - 10,000", "color": "#F1605D", "size": 8, "shape": "line"},
            ]
        )

        # add links layer styled by capacity
        capacity_styling = {
            "lineColor": {
                "column": "capacity_ab",
                "palette": "SunsetDark",
                "dataRange": [0, 1000],
            },
            "lineWidth": {
                "column": "capacity_ab",
                "dataRange": [0, 1000],
                "widthRange": [1, 200],
            },
        }

        panel.add_layer(
            "links",
            {
                "table": "links",
                "geometry": "line",
                # "sqlFilter": "link_type != 3",
                "style": capacity_styling,
            },
        )

        return [panel]

    def _scenario_metric_map(
        self,
        title,
        results_table,
        metric_column,
        legend,
        data_range,
        palette="Temps",
        width_by_link_type=True,
        width_by_metric=None,
    ):
        """makes scenario comparison map for a network's performance metric'

        :Arguments:
            **title** (:obj:`str`): panel title
            **results_table** (:obj:`str`): results table to join to links
            **metric_column** (:obj:`str`): metric to use for link colouring
            **legend** (:obj:`list`): legend def for the map
            **data_range** (:obj:`list`): value range used for colour scale
            **palette** (:obj:`str`, *Optional*): colour palette to use
            **width_by_link_type** (:obj:`bool`, *Optional*): whether to vary line width by link type (optional)
        """

        panel = AequilibraEMapPanel(
            title=title,
            height=10,
            width=3,
            center=self.center,
            zoom=self.zoom,
            projection="EPSG:32719",
        )

        # add extra db
        panel.set_extra_databases({"results": "results_database.sqlite"})

        panel.set_legend(legend)

        # links layer styling
        style = {
            "lineColor": {
                "column": metric_column,
                "palette": palette,
                "dataRange": data_range,
            }
        }

        # link type by line width: optional (kept for compatibility with example YAML)
        if width_by_link_type:
            style["lineWidth"] = {
                "column": "link_type",
                "widths": {
                    3: 20,
                    2: 40,
                    1: 20,
                },
            }

        if width_by_metric:
            style["lineWidth"] = {
                "column": width_by_metric,
                "dataRange": [0, 500],  # project-dependent default; adjust to your data's scale
                "widthRange": [10, 250],
            }

        # add links layer
        panel.add_layer(
            "links",
            {
                "table": "links",
                "geometry": "line",
                "join": {
                    "database": "results",
                    "table": results_table,
                    "leftKey": "link_id",
                    "rightKey": "link_id",
                    "type": "left",
                },
                "style": style,
            },
        )

        return panel

    def _delay_factor_row(self, results_tables):
        """Builds delay factor comparison panels"""

        row = []

        for table in results_tables:
            panel = AequilibraEResultsMapPanel(
                title=f"{table} Delay Factor",
                project=self.project,
                results_table=table,
                colour_metric="Delay_factor_Max",
                width_metric="capacity_ab",
            )
            row.append(panel)

        return row

    def _voc_comp_row(self, results_tables):
        """builds side by side comparison of Vehicles / Capacity maps for all scenarios"""

        row = []

        for table in results_tables:
            panel = AequilibraEResultsMapPanel(
                title=f"{table} vehicles / capacity", project=self.project, results_table=table, colour_metric="VOC_max"
            )
            row.append(panel)

        return row

    def _parse_convergence_json(self, json_string):
        """Parse procedure_report json and extract iteration and rgap arrays"""

        # if no procedure report
        if not json_string:
            return [], []

        # If the JSON is stored with escaped characters (double-encoded), unescape it first
        if json_string.startswith('{\\"'):
            json_string = json_string.encode().decode("unicode_escape")

        data: dict = json.loads(json_string)  # parsing json

        # double encoded json case
        if isinstance(data, str):
            data = json.loads(data)

        # if still a string
        if not isinstance(data, dict):
            return [], []

        convergence = data.get("convergence", {})  # get convergence block

        iteration = convergence.get("iteration", [])
        rgap = convergence.get("rgap", [])

        # return iteration and rgap arrays
        return (
            iteration,
            rgap,
        )

    def _export_convergence_csv(self, results_dataframe):
        """
        Export assignment convergence data for all results tables into a single CSV.

        Outputs: iteration, rgap, series
        """
        rows = []

        for _, row in results_dataframe.iterrows():
            table_name = row["table_name"]
            procedure_report = row.get("procedure_report")

            # extract convergence arrays
            iteration, rgap = self._parse_convergence_json(procedure_report)

            if not iteration or not rgap:
                continue  # skip tables with no convergence data

            # all same lengths
            if len(iteration) != len(rgap):
                raise ValueError(f"Iteration/RGAP length mismatch for {table_name}")

            # append rows
            for it, rg in zip(iteration, rgap, strict=True):
                rows.append(
                    {
                        "iteration": it,
                        "rgap": rg,
                        "series": table_name,
                    }
                )

        if not rows:
            return None

        output_path = self.data_dir / "assignment_convergence.csv"

        # write csv
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["iteration", "rgap", "series"])
            writer.writeheader()
            writer.writerows(rows)

        self._add_to_generated_files("assignment_convergence", output_path)
        return output_path

    def _write_convergence_vega_spec(self, csv_path):
        """Write a Vega-Lite spec for assignment convergence and return the filename."""

        # where to save it
        path = self.output_dir / "simwrapper_data" / "assignment_convergence.vega.json"

        spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {
                "url": str(csv_path),
                "format": {"type": "csv"},
            },
            "mark": {"type": "line", "point": False},
            "encoding": {
                "x": {
                    "field": "iteration",
                    "type": "quantitative",
                    "title": "Iteration",
                },
                "y": {"field": "rgap", "type": "quantitative", "title": "Relative Gap", "scale": {"type": "log"}},
                "color": {
                    "field": "series",
                    "type": "nominal",
                    "title": "Scenario",
                },
            },
        }

        # write vega json
        with open(path, "w") as f:
            json.dump(spec, f, indent=2)

        return path.name

    def _assignment_convergence_plot(self, results_dataframe):
        """Return a Vega-Lite convergence plot panel."""

        #  export convergence csv
        csv_path = self._export_convergence_csv(results_dataframe)

        # skip if no convergence data
        if csv_path is None:
            return None

        vega_spec = self._write_convergence_vega_spec(csv_path)

        # panel wrapper
        panel = ConvergencePanel(
            title="Assignment Convergence",
            config="simwrapper/simwrapper_data/" + vega_spec,
            height=6,
            width=6,
        )

        return [panel]

    def _flow_map_row(self, results_tables):
        """Map of links styled by assigned flows (PCE_tot)"""

        row = []

        for table in results_tables:
            panel = AequilibraEResultsMapPanel(
                title=f"{table} flow",
                project=self.project,
                results_table=table,
                colour_metric="VOC_max",
                width_metric="PCE_tot",
            )
            row.append(panel)

        return row

    def _build_dashboard_config(self):
        """Builds and returns full dashboard configuration for simwrapper"""

        config = self._dashboard_skeleton()  # base config

        # dashboard rows
        rows = {
            "introRow": self._intro_row(),
            "statsRow": self._stats_rows(),
            "entireNetworkRow": self._entire_network_row(),
            "linkTypeAndCapacityRow": self._links_info_row() + self._capacity_map_row(),
        }

        res_df = self.project.results.list()
        results_tables = res_df["table_name"].tolist()

        results_tables, truncated = self._truncate_results_tables(results_tables)

        if truncated:
            rows["resultsNoticeRow"] = [self._results_truncation_notice(len(results_tables), len(res_df))]

        # if we have results table, add relevant panels to dashboard
        if len(results_tables) > 0:
            rows["flowMapRow"] = self._flow_map_row(results_tables)
            rows["delayFactorComparisonRow"] = self._delay_factor_row(results_tables)
            rows["vocComparisonRow"] = self._voc_comp_row(results_tables)
            rows["assignmentConvergencePlot"] = self._assignment_convergence_plot(res_df)

        # convert panels to dicts and add to config
        for name, panels in rows.items():
            if panels:
                panel_dicts = []
                for p in panels:
                    panel_dicts.append(p.to_dict())
                config["layout"][name] = panel_dicts

        return config

    def write_yamls(self):
        """Write yamls"""
        config = self._build_dashboard_config()
        output_file = self.output_dir / "dashboard.yaml"

        # write it
        with output_file.open("w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        self._add_to_generated_files("dashboard", output_file)
