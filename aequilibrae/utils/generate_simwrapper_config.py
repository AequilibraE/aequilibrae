from pathlib import Path
import yaml
import geopandas as gpd
import pandas as pd
import math

from aequilibrae.utils.simwrapper_panel import SimwrapperPanel, TilePanel, TextPanel, AequilibraEMapPanel


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
        self.center = self._get_project_center()
        self.zoom = self._get_project_zoom()

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

    def _get_links_bounds_box(self):
        """ 
        Compute box around all coordinates in links table of project.
        Queries spatial database to find max and min x and y coords across all link geomerties
        to return overall network links' reach.

        Returns bounding box values (xmin, ymin, xmax, ymax)

        """
        with self.project.db_connection_spatial as conn:

            cursor = conn.cursor() # database cursor to make sql query

            # compute box around all coordinates in links table of project
            cursor.execute(
            """
            SELECT
                MIN(MBRMinX(geometry)) AS xmin,
                MIN(MBRMinY(geometry)) AS ymin,
                MAX(MBRMaxX(geometry)) AS xmax,
                MAX(MBRMaxY(geometry)) AS ymax
            FROM links
            """
            )

            row = cursor.fetchone() # fetch the single row returned by query (ie bounding box values)
        return row

    def _get_project_center(self):
        """ Finds center coordinates of project """
        row = self._get_links_bounds_box() 

        if row is None or any(value is None for value in row):
            return [0,0] # if cant find coordinates bc of missing link vals, will make this better though but works for now

        xmin, ymin, xmax, ymax = row

        # find center on each axis 
        center = [(xmin + xmax)/2, (ymin + ymax)/2] # [horizontal center, vertical center] == [longitude ,latitude]

        return center

    def _get_project_zoom(self):
        """ Finds a reasonable zoom level based on project links' reach"""

        # just to keep things reasonable
        max_zoom = 15
        min_zoom = 5

        row = self._get_links_bounds_box()

        if row is None or any(value is None for value in row):
            return 10 # if cant find coordinates bc of missing link vals, will make this better though but works for now

        xmin, ymin, xmax, ymax = row

        x_span = abs(xmax - xmin)
        y_span = abs(ymax - ymin)

        max_span = max(x_span, y_span) # use larger of two so we see everything

        if max_span <= 0:
            return 10 # if invalid values, clearly not a negative distance we want

        # calculate ~ zoom:
        # at zoom of 0 the world is ~360degrees wide
        # each increment doubles the resolution
        zoom = int(round(math.log2(360/max_span)))

        # fix this within the allowed range
        zoom = max(min_zoom, min(max_zoom, zoom))

        return zoom

    def _export_simple_stats(self, csv_name, stats_dict):
        """
        Export a one row csv from stats dictionary and add file to generated files.

        :Arguments:
            **name** (:obj:`str`): name of export
            **stats_dict** (:obj:`dict`): key:value stats to write
        """
        rows = list(stats_dict.items())

        output_file = self.data_dir/ f"{csv_name}.csv"  #output file path

        # manually bc needs to be saved as columns of metrics and column of values for simwrapper to read it right 
        with open(output_file, "w", newline="") as f:
            for key, value in rows:
                f.write(f"{key},{value}\n")

        self._add_to_generated_files(csv_name, output_file)

    def export_link_stats(self):
        """ Export simple about network's links."""
        links_obj = self.project.network.links
        links_df = links_obj.data

        stats = {
            "Link count": len(links_df), 
            "Link type count": links_df["link_type"].nunique()
        }

        self._export_simple_stats("link_stats", stats)


    def export_node_stats(self):
        """ Export simple stats about network's nodes."""
        nodes_obj = self.project.network.nodes
        nodes_df = nodes_obj.data

        stats = {
            "Node count": len(nodes_df)
        }

        self._export_simple_stats("node_stats", stats)


    def export_project_stats(self):
        """ Export basic network stats to csv. currently links and nodes because 
        that is all that is in Chicago model. More helpers can be made as needed. """

        if self._has_links():
            self.export_link_stats()

        if self._has_zones():
            self.export_node_stats()

    def generate_config(self):
        """Create the SimWrapper .yaml dashboard configuration."""

        # export all project data
        self.export_project_stats()

        # build yaml 
        self._write_yamls()

    def _dashboard_skeleton(self):
        """ Defines header and layout structure for yaml and returns the basic config skeleton """

        config = {
            "header": {
                "title": "insert title",
                "description": "insert description"
            },

            "layout": {}
        }

        return config

    def _intro_row(self):
        """ resturns project details text panel"""

        return [TextPanel(title="title", data="intro")]

    def _stats_rows(self):
        """ returns stats rows panels"""
        panels = []

        ## add links stats tile if available
        if "link_stats" in self.generated_files:
            panels.append(TilePanel("Link Statistics", str(self.generated_files["link_stats"]))) #output_dir/simwrapper_data/link_stats.csv))

        # add nodes stats tile if available
        if "node_stats" in self.generated_files:
            panels.append(TilePanel("Node Statistics", str(self.generated_files["node_stats"])))

        return panels

    def _entire_network_row(self):
        """ Builds yaml config for map of entire network """

        # aequilibrae panel with center and zoom
        panel = AequilibraEMapPanel("Entire Network", height=10, width=6, center=self.center, 
                                    zoom=self.zoom, projection="EPSG:32719")

        # set default styling
        default_style = {
            "fillColor": "#6f6f6f",
            "lineColor": "#FF6600",
            "lineWidth": 2,
            "pointRadius": 4,
        }
        panel.set_defaults( default_style)

        # add centroid nodes layer
        centroid_node_style = {
                            "fillColor": "#FF6600",
                            "pointRadius": 120
                            }
        panel.add_layer("nodes_centroids",
                        {
                        "table": "nodes",
                        "geometry": "point",
                        "sqlFilter": "is_centroid=1",
                        "style": centroid_node_style
                        })

        # add regular nodes layer
        regular_node_style = {
                            "fillColor": "#cacaca",
                            "pointRadius": 35
                            }
        panel.add_layer("nodes_regular", 
                        {
                        "table": "nodes",
                        "geometry": "point",
                        "sqlFilter": "is_centroid=0",
                        "style": regular_node_style
                        })

        # retun panel inside a list
        return [panel]

    def _links_info_row(self):
        """ Builds yaml config for panel to show attributes of selected link """

        # map panel
        panel = AequilibraEMapPanel("Link Types", height=10, width=6, center=self.center, 
                                    zoom=self.zoom)

        # set legend 
        panel.set_legend([
            {"subtitle": "Link Types"},
            {"label": "Freeway", "color": "#C3A34B", "shape": "line"},
            {"label": "Road", "color": "#74BBCD", "shape": "line"},
            {
                "label": "Centroid Connector",
                "color": "#99637f",
                "shape": "line",
            }
        ])

        # add links layer styled by link type
        link_type_by_colour = {
                        3: "#99637f",
                        2: "#C3A34B",
                        1: "#74BBCD",
                    }
        panel.add_layer("links",
            {"table": "links",
            "geometry": "line",
            "style": {
                "lineColor": {
                    "column": "link_type",
                    "colors": link_type_by_colour,
                }
            }
        })

        return [panel]

    def _capacity_map_row(self):
        """ Map showing links styled by capacity"""
        panel = AequilibraEMapPanel(
            title="Link Capacity",
            height=10,
            width=6,
            center=self.center,
            zoom=self.zoom
        )

        panel.set_defaults({"lineWidth": 3})

        # add links layer styled by capacity
        capacity_styling = {
                "lineColor": {
                    "column": "capacity",
                    "palette": "Viridis",
                    "dataRange": [0, 10000]
                }
            }

        panel.add_layer("links", {
            "table": "links",
            "geometry": "line",
            "style": capacity_styling
        })

        return [panel]

    def _build_dashboard_config(self):
        """ Builds and returns full dashboard configuration for simwrapper"""

        config = self._dashboard_skeleton() # based config

        # dashboard rows
        rows = {
            "introRow": self._intro_row(),
            "statsRow": self._stats_rows(),
            "entireNetworkRow": self._entire_network_row(),
            "linksInfoRow": self._links_info_row(),
            "capacityMapRow": self._capacity_map_row(),
        }

        # convert panels to dicts and add to config
        for name, panels in rows.items():
            if panels:
                panel_dicts = []
                for p in panels:
                    panel_dicts.append(p.to_dict())
                config["layout"][name] = panel_dicts


        return config

    def _write_yamls(self):
        """Write yamls """
        self.export_project_stats()

        config = self._build_dashboard_config()
        output_file = self.output_dir / "dashboard.yaml"

        # write it
        with output_file.open("w") as f:
            yaml.safe_dump(config, f, sort_keys = False)

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

    def _has_skims(self):
        """Checks if project has a network with skims"""
        return True
