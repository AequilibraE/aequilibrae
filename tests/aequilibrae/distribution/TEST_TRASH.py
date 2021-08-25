from unittest import TestCase
from uuid import uuid4
from tempfile import gettempdir
from os.path import join
from aequilibrae import Project
import geopandas as gpd
import urllib.request
from string import ascii_lowercase


class TestGravityApplication(TestCase):
    def test_apply(self):
        fldr = join(gettempdir(), uuid4().hex)
        project = Project()
        project.new(fldr)

        dest_path = join(fldr, "queluz.gpkg")
        urllib.request.urlretrieve('https://aequilibrae.com/data/queluz.gpkg', dest_path)

        gdf = gpd.read_file(dest_path)

        link_types = gdf.link_type.unique()

        lt = project.network.link_types
        lt_dict = lt.all_types()
        existing_types = [ltype.link_type for ltype in lt_dict.values()]

        types_to_add = [ltype for ltype in link_types if ltype not in existing_types]
        for i, ltype in enumerate(types_to_add):
            new_type = lt.new(ascii_lowercase[i])
            new_type.link_type = ltype
            # new_type.description = 'Your custom description here if you have one'
            new_type.save()

        md = project.network.modes
        md_dict = md.all_modes()
        existing_modes = {k: v.mode_name for k, v in md_dict.items()}

        # We get all the unique mode combinations and merge into a single string
        all_variations_string = ''.join(gdf.modes.unique())

        # We then get all the unique modes in that string above
        all_modes = set(all_variations_string)

        modes_to_add = [mode for mode in all_modes if mode not in existing_modes]
        for i, mode_id in enumerate(modes_to_add):
            new_mode = md.new(mode_id)
            # You would need to figure out the right name for each one, but this will do
            new_mode.mode_name = f'Mode_from_original_data_{mode_id}'
            # new_type.description = 'Your custom description here if you have one'

            # It is a little different, because you need to add it to the project
            project.network.modes.add(new_mode)
            new_mode.save()

        links = project.network.links
        link_data = links.fields()
        # Create the field and add a good description for it
        link_data.add('source_id', 'link_id from the data source')

        # We need to refresh the fields so the adding method can see it
        links.refresh_fields()

        print('start adding')
        for idx, record in gdf.iterrows():
            new_link = links.new()

            # Now let's add all the fields we had
            new_link.source_id = record.link_id
            new_link.direction = record.direction
            new_link.modes = record.modes
            new_link.link_type = record.link_type
            new_link.name = record.name
            new_link.geometry = record.geometry
            new_link.save()

            # We only do this to clear memory
            links.refresh()