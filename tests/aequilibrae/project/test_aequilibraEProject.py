# from unittest import TestCase
# from aequilibrae.project import Project as AequilibraEProject
# from ...data import project_file
#
#
# class TestAequilibraEProject(TestCase):
#     def load_model(self):
#         self.model = AequilibraEProject()
#         self.model.load(project_file)
#         self.model.load_model(project_file)
#
#     # def test_load_model(self):
#     #     model = AequilibraEProject()
#     #     model.load_model(project_file)
#
#     def test_metadata(self):
#         self.load_model()
#         assert self.model.metadata("model_name") == "My model test project"
#         assert self.model.metadata("author") == "Pedro Camargo"
#         assert self.model.metadata("contact_info") == "c@margo.co"
#         assert self.model.metadata("description") == "Project generic description"
#         assert self.model.metadata("origin_of_network") == "OSM"
#         assert self.model.metadata("origin_of_demand") == "Custom"
#         assert self.model.metadata("license") == "Same as AequilibraE"
#
#         # try:
#         #     x = self.model.metadata("xxxx")
#         #     self.fail("Failed to return an error on {} querying metadata  for AequilibraE project")
#         # except ValueError:
#         #     pass
#
#     def test_write_metadata(self):
#         self.load_model()
#         self.model.write_metadata("model_name", "Did it work?")
#         assert self.model.metadata("model_name") == "Did it work?"
#
#         self.model.write_metadata("model_name", "My model test project")
