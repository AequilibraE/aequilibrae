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

    def _export_simple_stats(self, csv_name, stats_dict):
        """
        Export a one row csv from stats dictionairy and add file to generated files.
        
        :Arguments:
        **name** (:obj:`str`): name of export
        **stats_dict** (:obj:`dict`): key:value stats to write
        """
        df = pd.DataFrame([stats_dict]) 
        output_file = self.data_dir/ f"{csv_name}.csv"  #output file path
        df.to_csv(output_file, index=False)

        self._add_to_generated_files(csv_name, output_file)

    def export_link_stats(self):
        """ Export simple about network's links."""
        links_obj = self.project.network.links
        links_df = links_obj.data

        stats = {
            "link_count": len(links_df), 
            "link_type_count": links_df["link_type"].nunique()            
        }

        self._export_simple_stats("link_stats", stats)


    def export_node_stats(self):
        """ Export simple stats about network's nodes."""
        nodes_obj = self.project.network.nodes
        nodes_df = nodes_obj.data

        stats = {
            "node_count": len(nodes_df)           
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

    def _build_dashboard_config(self):
        """ Build dashboard configuration for simwrapper"""

        config = {
            "header": {
                "title": "insert title",
                "description": "insert description"
            },
            "layout": []
        }

        # add available stats
        # links
        if "link_stats" in self.generated_files:
            config["layout"].append({
                "title": "Link Statistics", 
                "type": "table",
                "file": str(self.generated_files["link_stats"].relative_to(self.output_dir))
            })

        # nodes
        if "node_stats" in self.generated_files:
            config["layout"].append({
                "title": "Node Statistics", 
                "type": "table",
                "file": str(self.generated_files["node_stats"].relative_to(self.output_dir))
            })

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



    

    
