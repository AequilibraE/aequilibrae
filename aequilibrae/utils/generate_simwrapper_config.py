<<<<<<< HEAD
from pathlib import Path
import yaml
import geopandas as gpd
import pandas as pd
=======
import os
import yaml
import geopandas as gpd
>>>>>>> 1ae1a49b57f6fab710149424d79b81015e197627

class SimwrapperConfigGenerator:
    """
    Generate simwrapper ready .yaml file from an AequilibraE Project with
    minimal manual work.
    """
    def __init__(self, output_dir = "simwrapper"):
<<<<<<< HEAD
        """ Initialise the config generator and create output directories.

        :Arguments:
            **output_dir** (:obj:`str`, *Optional*): Root directory for SimWrapper outputs
        """
        self.project = None
        self.output_dir = Path(output_dir)
        self._create_directories()


    def _create_directories(self):
        """
        Create output directory structure for simwrapper
        
        Structure:
        simwrapper/            
        ├── data/               # Data files referenced by configs
        │   ├── links.geojson   # Spatial layer for network
        │   ├── linkstats.csv   # CSV of link properties/metrics
        │   ├── other_stats.csv # Additional CSV outputs
        │   └── ...             # Other CSV/GeoJSON as needed
        ├── dashboard-*.yaml    # Dashboard configuration file(s)
        """
        self.data_dir = self.output_dir / "data" # make subcategories

        self.output_dir.mkdir(exist_ok=True) # base
        self.data_dir.mkdir(exist_ok=True) # data

=======
        self.project = None
        self.data_dir = os.path.join(output_dir, "data")
        self.output_dir = output_dir
>>>>>>> 1ae1a49b57f6fab710149424d79b81015e197627

    def set_project(self, project):
        """Set project the .yaml is describing.

        :Arguments:
            project (Project): AequilibraE project instance
        """
        self.project = project

    def export_network_stats(self):
<<<<<<< HEAD
        """ Export basic network stats to csv. currently links and nodes.
        """
        #access links table
        links_obj = self.project.network.links
        links_df = links_obj.data

        #access notes table
        nodes_obj = self.project.network.nodes
        nodes_df = nodes_obj.data

        #build stats into a dataframe
        stats = {
            "link_count" : len(links_df),
            "node_count" : len(nodes_df)
        }
        df = pd.DataFrame([stats])

        # write output file
        output_file = self.data_dir / "linkstats.csv" 
        df.to_csv(output_file, index = False)


    def export_project_stats(self):
        """ Export project level statistics"""
=======
        links_obj = self.project.network.links # get link networks
        links_df = links_obj.data # geodataframe w link geometries + attributes

        link_count = len(links_df)
        with open(self.output_dir+"linkstats.csv", "w") as f:
            f.write("Link Count,"+str(link_count))



    def export_project_stats(self):
>>>>>>> 1ae1a49b57f6fab710149424d79b81015e197627
        self.export_network_stats()

    def generate_config(self):
        """Create the SimWrapper .yaml dashboard configuration.

<<<<<<< HEAD
=======
            - Validates the project
            - Exports required data files
            - Builds and writes the dashboard .yaml
        
        Directory structure for files generated:
            simwrapper/
            ├── data/
            │   ├── links.geojson
            │   ├── link_type_summary.csv
            │   └── other_stats.csv
            └── dashboard.yaml

>>>>>>> 1ae1a49b57f6fab710149424d79b81015e197627
        """
        # if no project stop
        if self.project == None:
            raise RuntimeError("You need to set the project via set_project(project) first")
        
<<<<<<< HEAD
        self._ensure_directories()
        
        # export data
        self.export_project_stats()

        # build yaml
        # write dashboard.yaml

    def _ensure_directories(self):

        pass

    def _build_dashboard_config(self):
        pass

    def _write_yamls(self):
        pass

    def _has_links(self):
        """Checks if project has a network with links """
        return (
            self.project is None
            and hasattr(self.project, "network")
            and self.project.network.links is not None
            and len(self.project.network.links.data) > 0
        )

    def _has_zones(self):
        """Checks if project has a network with nodes """
        return (
            self.project is None
            and hasattr(self.project, "network")
            and self.project.network.nodes is not None
            and len(self.project.network.nodes.data) > 0
        )

    def _has_assignments(self):
        pass



=======
        self.export_project_stats()
        
>>>>>>> 1ae1a49b57f6fab710149424d79b81015e197627




