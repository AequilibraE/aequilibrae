from os.path import dirname, abspath, join

spatialite_database = join(dirname(dirname(abspath(__file__))), "reference_files", "spatialite.sqlite")
