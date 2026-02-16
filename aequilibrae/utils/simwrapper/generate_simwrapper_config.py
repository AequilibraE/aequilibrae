from pathlib import Path
import yaml
import geopandas as gpd
import pandas as pd
import json
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
            **output_dir** (:obj:`str`, *Optional*): Root directory for SimWrapper outputs (created inside project)
        """
        self.project = project
        od = Path(output_dir)
        # keep simwrapper outputs inside the opened project folder; treat any provided
        # name as a project-relative subdirectory unless the caller explicitly
        # supplies a path below the project.
        if not od.is_absolute():
            od = Path(self.project.project_base_path) / od
        else:
            if not str(od).startswith(str(self.project.project_base_path)):
                od = Path(self.project.project_base_path) / od.name
        self.output_dir = od
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
        """Generate project title using the project's model name when available;
        otherwise derive a readable title from the project folder name. Guaranteed
        to return a non-empty string.
        """
        model_name = getattr(self.project.about, "model_name", None)
        if model_name:
            return model_name

        folder_name = Path(self.project.project_base_path).name
        title = folder_name.replace("_", " ").title()
        if title.strip():
            return title

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
        """Return a list of link-type *names* present in the project's network (empty list if none)."""
        try:
            lts = self.project.network.link_types.all_types()
            if lts:
                return [lt.link_type for lt in lts.values()]
        except AttributeError:
            # safe fallback to links table below when link_types API is not available
            pass
        return []

    def _categorical_palette(self, n):
        """Returns n visually distinct hex colour strings using matplotlib colormaps."""
        if n <= 0:
            return []
        cmap = plt.get_cmap("tab20")
        try:
            if n <= getattr(cmap, "N", 20):
                return [mcolors.to_hex(cmap(i)) for i in range(n)]
        except (AttributeError, TypeError):
            # fall back to requesting a resized colormap below
            pass
        cmap = plt.get_cmap("tab20", n)
        return [mcolors.to_hex(cmap(i)) for i in range(n)]

    def _truncate_results_tables(self, results_tables, max_tables=3):
        """Return a truncated results list and a flag indicating whether truncation occurred."""
        # TODO: get 3 most recent results
        # TODO: optionally take user input on which tables to show
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
            {
                "key": "Zone Count",
                "value": {
                    "database": "project_database.sqlite",
                    "query": "SELECT printf('%,d', COUNT(*)) FROM nodes WHERE is_centroid=1",
                },
            },
        ]

        panel = TilePanel("Network Size", dataset, height=1, colors="monochrome")

        return [panel]

    def _entire_network_row(self):
        """Builds yaml config for map of entire network"""

        # aequilibrae panel with center and zoom
        # prefer project projection when available; fall back to None
        proj = getattr(self.project.about, "projection", None)
        panel = AequilibraEMapPanel(
            "Entire Network",
            height=10,
            center=self.center,
            zoom=self.zoom,
            projection=proj,
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

        # determine centroid-link identification; prefer explicit link_types, otherwise infer
        centroid_names = [
            name for name in (self._get_link_types() or []) if "centroid" in name.lower() or "connector" in name.lower()
        ]
        if centroid_names:
            centroid_filter = " OR ".join([f"link_type = '{n}'" for n in centroid_names])
            non_centroid_filter = " AND ".join([f"link_type != '{n}'" for n in centroid_names])
        else:
            centroid_filter = "a_node IN (SELECT node_id FROM nodes WHERE is_centroid=1) OR b_node IN (SELECT node_id FROM nodes WHERE is_centroid=1)"
            non_centroid_filter = f"NOT ({centroid_filter})"

        # non-centroid connector links
        panel.add_layer(
            "links_regular",
            {
                "table": "links",
                "geometry": "line",
                "sqlFilter": non_centroid_filter,
                "style": {"lineColor": "#4C78A8", "lineWidth": 2},
            },
        )

        # centroid connector links
        panel.add_layer(
            "links_centroid_connectors",
            {
                "table": "links",
                "geometry": "line",
                "sqlFilter": centroid_filter,
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

        link_type_names = self._get_link_types()

        # fallback: read unique values directly from the links table
        if not link_type_names:
            links = self.project.network.links.data
            link_type_names = sorted(links["link_type"].unique().tolist())

        colours = self._categorical_palette(len(link_type_names))
        colour_map = dict(zip(link_type_names, colours))

        # map panel
        panel = AequilibraEMapPanel("Link Types", height=10, width=1, center=self.center, zoom=self.zoom)

        # build and set legend
        legend = [{"subtitle": "Link Types"}]
        for i, lt_name in enumerate(link_type_names):
            legend.append({"label": f"{lt_name}", "color": colours[i], "shape": "line"})

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
        panel = AequilibraEMapPanel(title="Link Capacity", height=10, width=1, center=self.center, zoom=self.zoom)

        panel.set_legend(
            [
                {"subtitle": "Link Capacity"},  # TODO: colour map
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
                "dataRange": [
                    0,
                    1000,
                ],  # TODO: we have the capability to dynamically set this based on data, this method is probably superseeded by the results maps (using "capacity" as results)
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
                # "sqlFilter": "link_type != 3", # TODO: idea here was to not show centroid connectors, since they have huge capacities. If link_types table is defined, we can use that, otherwise we'll have to guess for it. Better of (a) defaulting to show everything and (b) allowing user to specify what the centroid connector is
                "style": capacity_styling,
            },
        )

        return [panel]

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
            config=f"{self.output_dir.name}/simwrapper_data/{vega_spec}",
            height=6,
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
