from pathlib import Path
import yaml
import geopandas as gpd
import pandas as pd


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

    def _get_project_center(self):
        """ Find center coordinates of project """

        with self.project.db_connection_spatial as conn:

            cursor = conn.cursor() # database cursor to make sql query

            # compute box of all coordinates in links table of project
            cursor.execute(
            """
            SELECT
                MIN(MBRMinX(geometry)) AS xmin,
                MIN(MBRMinY(geometry)) AS ymin,
                MAX(MBRMaxX(geometry)) AS xmax,
                MAX(MBRMaxY(geometry)) AS ymin
            FROM links
            """
            )

            row = cursor.fetchone() # fetch the single row returned by query

        if row is None or any(value is None for value in row):
            return [0,0] # if cant find coordinates bc of missing link vals, will make this better though but works for now
        
        xmin, ymin, xmax, ymax = row

        # find center on each axis 
        center = [(xmin + xmax)/2, (ymin + ymax)/2] # [horizontal center, vertical center] == [longitude ,latitude]

        return center


    def _export_simple_stats(self, csv_name, stats_dict):
        """
        Export a one row csv from stats dictionairy and add file to generated files.
        
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
        # export data
        self.export_project_stats()

        # build yaml
        self._write_yamls()

        # write dashboard.yaml

    def _dashboard_skeleton(self):
        """ What is always in yaml """
        config = {
            "header": {
                "title": "insert title",
                "description": "insert description"
            },
            "layout": {}
        }

        return config

    def _intro_row(self):
        """ Project details """
        config = {
            "introRow": [
                {
                    "type": "text",
                    "title": "insert title",
                    "content": (
                        " bla bla bla"
                    )
                }
            ]
        }

        return config

    def _stats_rows(self):
        """ stats rows """
        stats_rows = []

        # links
        if "link_stats" in self.generated_files:
            stats_rows.append({
                "title": "Link Statistics", 
                "type": "tile",
                "dataset": str(self.generated_files["link_stats"]) #output_dir/simwrapper_data/link_stats.csv
            })

        # nodes
        if "node_stats" in self.generated_files:
            stats_rows.append({
                "title": "Node Statistics", 
                "type": "tile",
                "dataset": str(self.generated_files["node_stats"]) #output_dir/simwrapper_data/link_stats.csv
            })

        return stats_rows
    
    def _entire_network_row(self):
        """ Builds yaml config for map of entire network """
        config = [
            {
                "type": "aequilibrae",
                "title": "Entire Network",
                "database": "project_database.sqlite",
                "view": "map",
                "height": 10,
                "width": 6,
                "center": self.center,
                "zoom": 10,
                "projection": "EPSG:32719", # coordinate system?

                # default colours etc for now
                "defaults": {
                    "fillColor": "#6f6f6f",
                    "lineColor": "#FF6600",
                    "lineWidth": 2,
                    "pointRadius": 4
                },

                "layers": {
                    "nodes_centroids": {
                        "table": "nodes",
                        "geometry": "point",
                        "sqlFilter": "is_centroid=1",
                        "style": {
                            "fillColor": "#FF6600",
                            "pointRadius": 120
                        },
                    },
                    "nodes_regular": {
                        "table": "nodes",
                        "geometry": "point",
                        "sqlFilter": "is_centroid=0",
                        "style": {
                            "fillColor": "#cacaca",
                            "pointRadius": 35
                        },
                    },
                },
            }
        ]

        return config
    
    def _links_info_row(self):
        """ Builds yaml config for panel to show attributes of selected link """
        config = [
            {
                "type": "aequilibrae",
                "title": "Link Types",
                "database": "project_database.sqlite",
                "view": "map",
                "height": 10,
                "width": 3,
                "center": [-87.6298, 41.8781],
                "zoom": 10,

                "defaults": {
                    "lineWidth": 4,
                },

                "legend": [
                    {"subtitle": "Link Types"},
                    {"label": "Freeway", "color": "#C3A34B", "shape": "line"},
                    {"label": "Road", "color": "#74BBCD", "shape": "line"},
                    {
                        "label": "Centroid Connector",
                        "color": "#99637f",
                        "shape": "line",
                    },
                ],

                # Layer definitions for link-type styling
                "layers": {
                    "links": {
                        "table": "links",
                        "geometry": "line",

                        # Style links based on link_type column
                        "style": {
                            "lineColor": {
                                "column": "link_type",
                                "colors": {
                                    3: "#99637f",   # centroid connector
                                    2: "#C3A34B",   # freeway
                                    1: "#74BBCD",   # road
                                },
                            },
                            "lineWidth": {
                                "column": "link_type",
                                "widths": {
                                    3: 20,
                                    2: 80,
                                    1: 20,
                                },
                            },
                        },
                    }
                },
            }
        ]

        return config

    def _build_dashboard_config(self):
        """ Build full dashboard configuration for simwrapper"""

        config = self._dashboard_skeleton() 

        config["layout"]["introRow"] = self._intro_row()

        # add available stats
        if self._stats_rows():
            config["layout"]["statsRow"] = self._stats_rows()

        if self._entire_network_row():
            config["layout"]["entireNetworkRow"] = self._entire_network_row()

        if self._links_info_row():
            config["layout"]["linkInfoRow"] = self._links_info_row()

        return config

    def _write_yamls(self):
        """Write yamls """

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



    

    
