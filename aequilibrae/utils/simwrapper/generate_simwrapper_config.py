from pathlib import Path
import yaml
import geopandas as gpd
import pandas as pd
import json
import csv

from aequilibrae.utils.simwrapper.simwrapper_panel import SimwrapperPanel, ConvergencePanel, TilePanel, TextPanel, AequilibraEMapPanel
from aequilibrae.utils.simwrapper.simwrapper_utils import get_project_center, get_project_zoom

class SimwrapperConfigGenerator:
    """
    Generate simwrapper ready .yaml file from an AequilibraE Project with
    minimal manual work.
    """

    def __init__(self, project, output_dir="simwrapper"):
        """Initialise the config generator and create output directories.

        :Arguments:
            **output_dir** (:obj:`Project`, *Optional*): Aequilibrae Project being transferred
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

    def _add_to_generated_files(self, key, path):
        """Add file to self.generated_files"""
        self.generated_files[key] = Path(path)

    def _dashboard_skeleton(self):
        """Defines header and layout structure for yaml and returns the basic config skeleton"""

        config = {"header": {"title": "insert title", "description": "insert description"}, "layout": {}}

        return config

    def _intro_row(self):
        """resturns project details text panel"""

        return [TextPanel(title="title", data="intro")]

    def _stats_rows(self):
        """returns stats rows panels"""
        dataset = [{"key": "Link Count",
                    "value":
                        {"database": "project_database.sqlite",
                         "query": "SELECT COUNT(*) FROM links"}},
                    {"key": "Node Count",
                    "value":
                        {"database": "project_database.sqlite",
                         "query": "SELECT COUNT(*) FROM nodes"}}]

        panel = TilePanel("Network Size", dataset)

        return [panel]

    def _entire_network_row(self):
        """Builds yaml config for map of entire network"""

        # aequilibrae panel with center and zoom
        panel = AequilibraEMapPanel(
            "Entire Network", height=10, width=6, center=self.center, zoom=self.zoom, projection="EPSG:32719"
        )

        # set default styling
        default_style = {
            "fillColor": "#6f6f6f",
            "lineColor": "#FF6600",
            "lineWidth": 2,
            "pointRadius": 4,
        }
        panel.set_defaults(default_style)

        # add centroid nodes layer
        centroid_node_style = {"fillColor": "#FF6600", "pointRadius": 120}
        panel.add_layer(
            "nodes_centroids",
            {"table": "nodes", "geometry": "point", "sqlFilter": "is_centroid=1", "style": centroid_node_style},
        )

        # add regular nodes layer
        regular_node_style = {"fillColor": "#cacaca", "pointRadius": 35}
        panel.add_layer(
            "nodes_regular",
            {"table": "nodes", "geometry": "point", "sqlFilter": "is_centroid=0", "style": regular_node_style},
        )

        # retun panel inside a list
        return [panel]

    def _links_info_row(self):
        """Builds yaml config for panel to show attributes of selected link"""

        # map panel
        panel = AequilibraEMapPanel("Link Types", height=10, width=6, center=self.center, zoom=self.zoom)

        # set legend
        panel.set_legend(
            [
                {"subtitle": "Link Types"},
                {"label": "Freeway", "color": "#C3A34B", "shape": "line"},
                {"label": "Road", "color": "#74BBCD", "shape": "line"},
                {
                    "label": "Centroid Connector",
                    "color": "#99637f",
                    "shape": "line",
                },
            ]
        )

        # add links layer styled by link type
        link_type_by_colour = {
            3: "#99637f",
            2: "#C3A34B",
            1: "#74BBCD",
        }
        panel.add_layer(
            "links",
            {
                "table": "links",
                "geometry": "line",
                "style": {
                    "lineColor": {
                        "column": "link_type",
                        "colors": link_type_by_colour,
                    }
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
        self, title, results_table, metric_column, legend, data_range, palette="Temps", width_by_link_type=True
    ):
        """makes scenario comparison map for a network's performance metric'

        :Arguments:
            **title** (:obj:`str`): panel title
            **results_table** (:obj:`str`): results table to join to links
            **metric_column** (:obj:`str`): metric to use for link colouring
            **legend** (:obj:`list`): legend def for the map
            **data_range** (:obj:`list`): value range used for colour scale
            **palette** (:obj:`str`, *Optional*): colour palette to use
            **width_by_link_type** (:obj:`bool`, *Optional*): vary line width by link type????????? weird
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

        # defaultstyling
        panel.set_defaults(
            {
                "fillColor": "#6f6f6f",
                "lineColor": "#FF6600",
                "lineWidth": 2,
                "pointRadius": 4,
            }
        )

        panel.set_legend(legend)

        # links layer styling
        style = {
            "lineColor": {
                "column": metric_column,
                "palette": palette,
                "dataRange": data_range,
            }
        }

        # link type by line width?? made optional bc weird, but example yaml does this
        if width_by_link_type:
            style["lineWidth"] = {
                "column": "link_type",
                "widths": {
                    3: 20,
                    2: 40,
                    1: 20,
                },
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

    def _metric_comp_row(self, title, metric, tables):
        """Builds side by side comparison of base case vs active/transit metric map panels"""

        legend = [
            {"subtitle": metric},
            {"label": "Low", "color": "#009392", "size": 4, "shape": "line"},
            {"label": "Medium", "color": "#e9e29c", "size": 4, "shape": "line"},
            {"label": "High", "color": "#cf597e", "size": 4, "shape": "line"},
        ]

        row = []

        for table in tables:
            panel = self._scenario_metric_map(
                title=f"{table} {title}",
                results_table=table,
                metric_column=metric,
                legend=legend,
                data_range=[1, 3],
            )
            row.append(panel)

        return row

    def _delay_factor_comp_row(self):
        """Builds side by side comparison of base case vs active/transit delay factor map panels"""

        legend = [
            {"subtitle": "Delay Factor"},
            {"label": "1", "color": "#009392", "size": 4, "shape": "line"},
            {"label": "2", "color": "#e9e29c", "size": 4, "shape": "line"},
            {"label": ">3", "color": "#cf597e", "size": 4, "shape": "line"},
        ]

        # base case delay factor map
        base = self._scenario_metric_map(
            title="base case delay factor",
            results_table="base_case",
            metric_column="Delay_factor_Max",
            legend=legend,
            data_range=[1, 3],
        )

        # transit/active friendly delay factor map
        tat = self._scenario_metric_map(
            title="transit/active friendly delay factor",
            results_table="transit_and_active_friendly",
            metric_column="Delay_factor_Max",
            legend=legend,
            data_range=[1, 3],
        )

        return [base, tat]

    def _parse_convergence_json(self, json_string):
        """ Parse procedure_report json and extract iteration and rgap arrays"""

        # if no procedure report
        if not json_string:
            return [], []

        # if stored as excaped string unescape
        if json_string.startswith('{\\"'):
            json_string = json_string.encode().decode("unicode_escape")

        data: dict = json.loads(json_string) # parsing json

        # double encoded json case
        if isinstance(data, str):
            data = json.loads(data)

        # if still a string
        if not isinstance(data, dict):
            return [], []

        convergence = data.get("convergence", {}) # get convergence block

        iteration = convergence.get("iteration", [])
        rgap = convergence.get("rgap", [])

        # return iteration and rgap arrays
        return (iteration, rgap,)

    def _export_convergence_csv(self, results_dfataframe):
        """
        export assignment convergence data for all result tables into a single CSV.

        outputs: iteration, rgap, series
        """
        rows = []

        for _, row in results_dfataframe.iterrows():
            table_name = row["table_name"]
            procedure_report = row.get("procedure_report")

            # extract cinveregnce arrays
            iteration, rgap = self._parse_convergence_json(procedure_report)

            if not iteration or not rgap:
                continue  # skip tables with no convergence data

            # all same lengths
            if len(iteration) != len(rgap):
                raise ValueError(f"Iteration/RGAP length mismatch for {table_name}")

            # append rows
            for it, rg in zip(iteration, rgap, strict=True):
                rows.append({
                    "iteration": it,
                    "rgap": rg,
                    "series": table_name,
                })

        if not rows:
            return None

        output_path = self.data_dir / "assignment_convergence.csv"

        # write csv
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["iteration", "rgap", "series"]
            )
            writer.writeheader()
            writer.writerows(rows)

        self._add_to_generated_files("assignment_convergence", output_path)
        return output_path

    def _write_convergence_vega_spec(self, csv_path):
        """writes vegalite spec for assignment convergence, returns path to this"""

        # where to save it
        path = self.output_dir/"assignment_convergence.vega.json"

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
                "y": {
                    "field": "rgap",
                    "type": "quantitative",
                    "title": "Relative Gap",
                },
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
        """ returns vegalite convergence plot panel """

        #  export convergence csv
        csv_path = self._export_convergence_csv(results_dataframe)

        # skip if no convergence data
        if csv_path is None:
            return None

        vega_spec = self._write_convergence_vega_spec(csv_path)

        # panel wrapper
        panel = ConvergencePanel(
            type="vega",
            title="Assignment Convergence",
            config=vega_spec,
            height=6,
            width=6,
        )

        return [panel]

    def _build_dashboard_config(self):
        """Builds and returns full dashboard configuration for simwrapper"""

        config = self._dashboard_skeleton()  # based config

        # dashboard rows
        rows = {
            "introRow": self._intro_row(),
            "statsRow": self._stats_rows(),
            "entireNetworkRow": self._entire_network_row(),
            "linkTypeAndCapasityRow": self._links_info_row() + self._capacity_map_row(),
        }

        res_df = self.project.results.list()
        results_tables = res_df["table_name"].tolist()
        # if we have results table, add a delay factor comparison
        if len(results_tables) > 0:
            rows["delayFactorComparisonRow"] = self._metric_comp_row("delay factor", "Delay_factor_Max", results_tables)
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

    def _has_links(self):
        """Checks if project has a network with links"""
        return True

    def _has_nodes(self):
        """Checks if project has a network with nodes"""
        return True

    def _has_zones(self):
        """Checks if project has a network with nodes"""
        return True

    def _has_matrices(self):
        """Checks if project has a network with skims"""
        return True