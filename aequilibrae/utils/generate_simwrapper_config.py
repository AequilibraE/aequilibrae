import os
import yaml
import geopandas as gpd

class SimwrapperConfigGenerator:
    """
    Generate simwrapper ready .yaml file from an AequilibraE Project with
    minimal manual work.
    """
    def __init__(self, output_dir = "simwrapper"):
        self.project = None
        self.data_dir = os.path.join(output_dir, "data")
        self.output_dir = output_dir

    def set_project(self, project):
        """Set project the .yaml is describing.

        :Arguments:
            project (Project): AequilibraE project instance
        """
        self.project = project

    def export_network_stats(self):
        links_obj = self.project.network.links # get link networks
        links_df = links_obj.data # geodataframe w link geometries + attributes

        link_count = len(links_df)
        with open(self.output_dir+"linkstats.csv", "w") as f:
            f.write("Link Count,"+str(link_count))



    def export_project_stats(self):
        self.export_network_stats()

    def generate_config(self):
        """Create the SimWrapper .yaml dashboard configuration.

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

        """
        # if no project stop
        if self.project == None:
            raise RuntimeError("You need to set the project via set_project(project) first")
        
        self.export_project_stats()
        




