"""Utilities to generate SimWrapper dashboard configuration for an AequilibraE project.

Usage
-----
.. code-block:: python
    :caption: Generate SimWrapper dashboard for an open project

    >>> from aequilibrae.project import Project
    >>> from aequilibrae.utils.simwrapper.generate_simwrapper_config import SimwrapperConfigGenerator
    >>> prj = Project()
    >>> prj.open('/path/to/project')
    >>> gen = SimwrapperConfigGenerator(prj, output_dir='simwrapper')
    >>> gen.write_yamls()

Notes
-----
- `output_dir` must be inside the project directory
"""

from pathlib import Path
import json

import pandas as pd
import yaml

from aequilibrae.utils.simwrapper.simwrapper_panel import (
    ConvergencePanel,
    TilePanel,
    TextPanel,
    AequilibraEMapPanel,
    AequilibraEResultsMapPanel,
)
from aequilibrae.utils.simwrapper.simwrapper_utils import (
    get_project_center,
    get_project_zoom,
    export_convergence_csv,
)


class SimwrapperConfigGenerator:
    """Generates SimWrapper dashboard configuration for an AequilibraE project."""

    def __init__(
        self,
        project,
        output_dir="simwrapper",
        max_results_tables=3,
        results_tables=None,
        centroid_link_types=None,
    ):
        """Initialise the configuration generator.

        Arguments:
            **project** (:obj:`Project`): AequilibraE Project object
            **output_dir** (:obj:`str`, optional): Path where SimWrapper output files will be written.
                Relative paths are created under the project's base directory. Absolute paths are
                accepted only when they reside inside the project; absolute paths outside the
                project will raise a ValueError.
            **max_results_tables** (:obj:`int`, optional): Maximum number of results scenarios to include
                when ``results_tables`` is not specified (default: 3)
            **results_tables** (:obj:`list[str]`, optional): Explicit list of results table names to include.
                When set, no automatic truncation is applied
            **centroid_link_types** (:obj:`list[str]`, optional): Link type names considered to be centroid
                connectors. When not provided, the generator attempts to infer them
        """
        self.project = project
        self.max_results_tables = int(max_results_tables) if max_results_tables is not None else 3
        self.results_tables = results_tables
        self.centroid_link_types = centroid_link_types

        od = Path(output_dir)
        project_root = Path(self.project.project_base_path).resolve()
        # Treat relative paths as project-relative subdirectories; accept absolute paths
        # only when they are located inside the project directory. Absolute paths
        # outside the project are rejected to guarantee all SimWrapper outputs remain
        # under the project directory.
        if not od.is_absolute():
            od = (project_root / od).resolve()
        else:
            od = od.resolve()
            if not od.is_relative_to(project_root):
                raise ValueError(f"output_dir must be inside the project directory ({project_root}); got '{od}'")
        self.output_dir = od
        self.generated_files = {}
        self._create_directories()
        self.center = get_project_center(self.project)
        self.zoom = get_project_zoom(self.project)

    def _create_directories(self):
        """Create output directory structure for SimWrapper.

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
        """Generate a project title.

        Uses the project's model name when available;
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
        """Add a generated file reference for inspection."""
        self.generated_files[key] = Path(path)

    def _dashboard_skeleton(self):
        """Creates the base dashboard configuration."""

        title = self._find_project_title()
        desc = f"Dashboard generated from '{title}'"
        return {"header": {"title": title, "description": desc}, "layout": {}}

    def _intro_row(self):
        """Returns a project introduction text panel."""

        title = self._find_project_title()
        content = f"# {title}\n\nThis dashboard was generated by AequilibraE to help explore the network and results."
        return [TextPanel(title="Overview", data=content)]

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
        """Returns n visually distinct hex colour strings.

        Notes:
            This uses a fixed palette to avoid optional heavyweight dependencies.
        """
        if n <= 0:
            return []

        # Matplotlib tab20 palette (approx), stored as hex values.
        base = [
            "#1f77b4",
            "#aec7e8",
            "#ff7f0e",
            "#ffbb78",
            "#2ca02c",
            "#98df8a",
            "#d62728",
            "#ff9896",
            "#9467bd",
            "#c5b0d5",
            "#8c564b",
            "#c49c94",
            "#e377c2",
            "#f7b6d2",
            "#7f7f7f",
            "#c7c7c7",
            "#bcbd22",
            "#dbdb8d",
            "#17becf",
            "#9edae5",
        ]

        if n <= len(base):
            return base[:n]

        # For more categories than the base palette, cycle values.
        return [base[i % len(base)] for i in range(n)]

    def _select_results_tables(self, results_dataframe):
        """Selects which results tables to include in the dashboard.

        If ``self.results_tables`` is provided, returns that list (filtered to existing).
        Otherwise, selects up to ``self.max_results_tables`` most recent results based on the
        ``timestamp`` field when available.

        Returns:
            **tables** (:obj:`list[str]`): Selected results table names
            **truncated** (:obj:`bool`): Whether some scenarios were omitted
            **total** (:obj:`int`): Total number of available scenarios
        """
        res_df = results_dataframe
        total = int(len(res_df))

        if total == 0:
            return [], False, 0

        available = res_df["table_name"].tolist()

        if self.results_tables is not None:
            chosen = [t for t in self.results_tables if t in available]
            return chosen, False, total

        df = res_df.copy()
        if "timestamp" in df.columns:
            df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values(by=["_ts", "table_name"], ascending=[False, True])

        max_tables = max(0, int(self.max_results_tables))
        if total <= max_tables or max_tables == 0:
            # max_tables == 0 means "do not include any results" in the auto mode
            return ([] if max_tables == 0 else df["table_name"].tolist()), (max_tables == 0), total

        chosen = df["table_name"].head(max_tables).tolist()
        return chosen, True, total

    def _results_truncation_notice(self, shown, total):
        """Builds a short notice panel when not all scenarios are shown."""
        return TextPanel(
            title="Please Note",
            data=(
                f"Showing {shown} of {total} result scenarios (most recent first).\n\n"
                "Additional scenarios were omitted to keep the dashboard readable."
            ),
        )

    def _stats_rows(self):
        """Returns a row of basic project statistics."""
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

    def _centroid_link_filters(self):
        """Builds SQL filters for centroid connectors.

        Uses explicit ``self.centroid_link_types`` when provided; otherwise attempts to infer
        centroid link types by name.

        Returns:
            **centroid_filter** (:obj:`str`): SQL boolean expression matching centroid connectors
            **non_centroid_filter** (:obj:`str`): SQL boolean expression matching non-centroid links
        """
        centroid_names = self.centroid_link_types
        if centroid_names is None:
            centroid_names = [
                name
                for name in (self._get_link_types() or [])
                if "centroid" in name.lower() or "connector" in name.lower()
            ]

        if centroid_names:
            # escape single-quotes in link-type names for safe SQL interpolation
            safe_names = [n.replace("'", "''") for n in centroid_names]
            centroid_filter = " OR ".join([f"link_type = '{s}'" for s in safe_names])
            non_centroid_filter = " AND ".join([f"link_type != '{s}'" for s in safe_names])
        else:
            centroid_filter = (
                "a_node IN (SELECT node_id FROM nodes WHERE is_centroid=1) "
                "OR b_node IN (SELECT node_id FROM nodes WHERE is_centroid=1)"
            )
            non_centroid_filter = f"NOT ({centroid_filter})"

        return centroid_filter, non_centroid_filter

    def _entire_network_row(self):
        """Builds a map of the entire network."""

        # aequilibrae panel with center and zoom
        panel = AequilibraEMapPanel(
            "Entire Network",
            height=10,
            center=self.center,
            zoom=self.zoom,
        )

        centroid_filter, non_centroid_filter = self._centroid_link_filters()

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

        return [panel]

    def _links_info_row(self):
        """Builds a map styled by link type."""

        link_type_names = self._get_link_types()

        # fallback: read unique values directly from the links table
        if not link_type_names:
            with self.project.db_connection as conn:
                rows = conn.execute(
                    "SELECT DISTINCT link_type FROM links WHERE link_type IS NOT NULL ORDER BY link_type"
                ).fetchall()
            link_type_names = [r[0] for r in rows]

        colours = self._categorical_palette(len(link_type_names))
        colour_map = dict(zip(link_type_names, colours, strict=False))

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
        """Builds a map styled by link capacity.

        This reads ``capacity_ab`` directly from the project ``links`` table.
        """
        _, non_centroid_filter = self._centroid_link_filters()
        panel = AequilibraEResultsMapPanel(
            title="Link Capacity",
            project=self.project,
            colour_metric="capacity_ab",
            width_metric="capacity_ab",
            palette="SunsetDark",
            height=10,
            width=1,
            center=self.center,
            zoom=self.zoom,
            sql_filter=non_centroid_filter,
        )
        return [panel]

    def _delay_factor_row(self, results_tables):
        """Builds delay factor comparison panels."""

        return [
            AequilibraEResultsMapPanel(
                title=f"{table} Delay Factor",
                project=self.project,
                results_table=table,
                colour_metric="Delay_factor_Max",
                width_metric="capacity_ab",
            )
            for table in results_tables
        ]

    def _voc_comp_row(self, results_tables):
        """Builds vehicles/capacity comparison panels."""

        return [
            AequilibraEResultsMapPanel(
                title=f"{table} vehicles / capacity",
                project=self.project,
                results_table=table,
                colour_metric="VOC_max",
            )
            for table in results_tables
        ]

    def _export_convergence_csv(self, results_dataframe):
        """Delegate convergence CSV creation to utils and register the generated file.

        Keeps the public behaviour unchanged (returns Path or None; registers file).
        """
        output_path = export_convergence_csv(results_dataframe, self.data_dir)
        if output_path is None:
            return None
        self._add_to_generated_files("assignment_convergence", output_path)
        return output_path

    def _write_convergence_vega_spec(self, csv_path):
        """Write a Vega-Lite spec for assignment convergence.

        The Vega spec references the CSV using a relative URL to keep the output folder portable.

        Returns:
            **spec_name** (:obj:`str`): Vega spec filename
        """

        # where to save it
        path = self.data_dir / "assignment_convergence.vega.json"

        spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {
                "url": Path(csv_path).name,
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
        with path.open("w", encoding="utf-8") as f:
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
            config=(Path("simwrapper_data") / vega_spec).as_posix(),
            height=6,
        )

        return [panel]

    def _flow_map_row(self, results_tables):
        """Builds maps styled by assigned flows."""

        return [
            AequilibraEResultsMapPanel(
                title=f"{table} flow",
                project=self.project,
                results_table=table,
                colour_metric="VOC_max",
                width_metric="PCE_tot",
            )
            for table in results_tables
        ]

    def _build_dashboard_config(self):
        """Builds and returns the full dashboard configuration."""

        config = self._dashboard_skeleton()  # base config

        # dashboard rows
        rows = {
            "introRow": self._intro_row(),
            "statsRow": self._stats_rows(),
            "entireNetworkRow": self._entire_network_row(),
            "linkTypeAndCapacityRow": self._links_info_row() + self._capacity_map_row(),
        }

        res_df = self.project.results.list()
        results_tables, truncated, total = self._select_results_tables(res_df)

        if truncated and total:
            rows["resultsNoticeRow"] = [self._results_truncation_notice(len(results_tables), total)]

        # if we have results table, add relevant panels to dashboard
        if len(results_tables) > 0:
            rows["flowMapRow"] = self._flow_map_row(results_tables)
            rows["delayFactorComparisonRow"] = self._delay_factor_row(results_tables)
            rows["vocComparisonRow"] = self._voc_comp_row(results_tables)
            rows["assignmentConvergencePlot"] = self._assignment_convergence_plot(res_df)

        # convert panels to dicts and add to config
        for name, panels in rows.items():
            if panels:
                config["layout"][name] = [p.to_dict() for p in panels]

        return config

    def write_yamls(self):
        """Writes the SimWrapper dashboard YAML file."""
        config = self._build_dashboard_config()
        output_file = self.output_dir / "dashboard.yaml"

        # write it
        with output_file.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        self._add_to_generated_files("dashboard", output_file)


def main(argv=None):
    """Command-line entry point for generating SimWrapper configs.

    Example:
        aequilibrae-simwrapper --project /path/to/project --output-dir simwrapper
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="aequilibrae-simwrapper",
        description="Generate SimWrapper dashboard YAML for an AequilibraE project",
    )
    parser.add_argument("-p", "--project", default=".", help="Project path (folder containing project_database.sqlite)")
    parser.add_argument("-o", "--output-dir", default="simwrapper", help="Output directory (inside project)")
    parser.add_argument("--max-results-tables", type=int, default=None, help="Maximum number of results tables")
    parser.add_argument(
        "--results-tables", nargs="+", default=None, help="Explicit results table names (space-separated)"
    )
    parser.add_argument(
        "--centroid-link-types", nargs="+", default=None, help="Centroid link type names (space-separated)"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress informational output")
    args = parser.parse_args(argv)

    # lazy import to avoid side-effects at module import time
    import sys
    from aequilibrae.project import Project

    prj = Project()
    try:
        prj.open(args.project)
    except Exception as e:
        print(f"Error opening project at '{args.project}': {e}", file=sys.stderr)
        return 2

    try:
        gen = SimwrapperConfigGenerator(
            prj,
            output_dir=args.output_dir,
            max_results_tables=args.max_results_tables,
            results_tables=args.results_tables,
            centroid_link_types=args.centroid_link_types,
        )
        gen.write_yamls()
        if not args.quiet:
            print(f"Written {len(gen.generated_files)} files to {gen.output_dir}")
            for k, v in gen.generated_files.items():
                print(f" - {k}: {v}")
        return 0
    except Exception as e:
        print(f"Error generating simwrapper config: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
