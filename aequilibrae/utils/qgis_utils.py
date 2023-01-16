import importlib.util as iutil

# If we can find the qgis module to import ... we are running inside qgis
inside_qgis = iutil.find_spec("qgis") is not None
