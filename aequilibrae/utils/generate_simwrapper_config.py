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
            data/               # Data files referenced by configs
                linkstats.csv   # CSV of link properties/metrics
                other_stats.csv # Additional CSV outputs
                ...             
            dashboard-*.yaml    # Dashboard configuration file(s)
        """
        self.data_dir = self.output_dir / "data"  # make subcategories

        self.output_dir.mkdir(exist_ok=True)  # base
        self.data_dir.mkdir(exist_ok=True)  # data

    def export_network_stats(self):
        """Export basic network stats to csv. currently links and nodes."""
        # access links table
        links_obj = self.project.network.links
        links_df = links_obj.data

        # access notes table
        nodes_obj = self.project.network.nodes
        nodes_df = nodes_obj.data

        # build stats into a dataframe
        stats = {
            "link_count": len(links_df), 
            "node_count": len(nodes_df)
                 }
        
        df = pd.DataFrame([stats])

        # write output file
        output_file = self.data_dir / "linkstats.csv"
        df.to_csv(output_file, index=False)

        # save stats

    def export_project_stats(self):
        """Export project level statistics"""
        self.export_network_stats()

    def generate_config(self):
        """Create the SimWrapper .yaml dashboard configuration."""
        # export data
        self.export_project_stats()

        # build yaml
        # write dashboard.yaml

    def _build_dashboard_config(self):
        pass

    def _write_yamls(self):
        pass

    def _has_links(self):
        """Checks if project has a network with links"""
        return (
            self.project is not None
            and hasattr(self.project, "network")
            and self.project.network.links is not None
            and len(self.project.network.links.data) > 0
        )

    def _has_zones(self):
        """Checks if project has a network with nodes"""
        return (
            self.project is not None
            and hasattr(self.project, "network")
            and self.project.network.nodes is not None
            and len(self.project.network.nodes.data) > 0
        )

    def _has_assignments(self):
        """Checks if project has a network with assignments"""
        return (
            self.project is not None
            and hasattr(self.project, "network")
            and self.project.network.assignments is not None
            and len(self.project.network.assignments.data) > 0
        )

    def _has_skims(self):
        """Checks if project has a network with skims"""
        return (
            self.project is not None
            and hasattr(self.project, "network")
            and self.project.network.skims is not None
            and len(self.project.network.skims.data) > 0
        )
    
    def _add_to_generated_files(self, key, path):
        """Add file to self.generated_files"""
        self.generated_files[key] = Path(path)

    
