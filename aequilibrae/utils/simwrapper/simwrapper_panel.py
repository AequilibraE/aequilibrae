class SimwrapperPanel():
    """ Base class for all simwrapper panels

    :Arguments:
        **type** (:obj:`str`): panel type
        **title** (:obj:`str`): title to show in the dashboard
        **height** (:obj:`int`, *Optional*): panel height
        **width** (:obj:`int`, *Optional*): panel width

    :Example:
        panel = SimwrapperPanel("text", "My Panel", height=3, width=6)
    """
    def __init__(self, type, title, height=None, width=None):
        self.type = type
        self.title = title
        self.height = height
        self.width = width

    def to_dict(self):
        """Returns dictionary representation of the panel"""
        panel = {
            "type": self.type,
            "title": self.title,
        }

        if self.height:
            panel["height"] = self.height

        if self.width:
            panel["width"] = self.width

        return panel

class ConvergencePanel(SimwrapperPanel):
    def __init__(self, title, config, height=None, width=None):
        super().__init__("vega", title, height=height, width=width)
        self.config = config

    def to_dict(self):
        """Returns dictionary representation of the panel"""
        panel = super().to_dict()

        panel["config"] = self.config

        return panel


class TilePanel(SimwrapperPanel):
    """
    Panel used to display tabular summary statistics.

    :Arguments:
        **title** (:obj:`str`): title
        **dataset** (:obj:`str`): path to csv dataset used by tile
        **height** (:obj:`int`, *Optional*): panel height
        **width** (:obj:`int`, *Optional*): p[anel width

    :Example:
        panel = TilePanel("Summary Statistics", "data/summary.csv", height=3)
    """
    def __init__(self, title, dataset, height=None, width=None):
        super().__init__("tile", title, height=height, width=width)
        self.dataset = dataset

    def to_dict(self):
        """Returns dictionary representation of the panel"""
        panel = super().to_dict()

        panel["dataset"] = self.dataset

        return panel

class TextPanel(SimwrapperPanel):
    """
    Panel for displaying text content.

    :Arguments:
        **title** (:obj:`str`): title
        **data** (:obj:`str`): text content or file path
        **is_file** (:obj:`bool`, *Optional*): if data is a file reference
        **height** (:obj:`int`, *Optional*): panel height
        **width** (:obj:`int`, *Optional*): panel width

    :Example:
        panel = TextPanel("Overview", "text/overview.md", is_file=True)
    """
    def __init__(self, title, data, is_file=False, height=None, width=None):
        super().__init__("text", title, height=height, width=width)
        self.data = data
        self.is_file = is_file

    def to_dict(self):
        """Returns dictionary representation of the panel"""
        panel = super().to_dict()

        if self.is_file:
            panel["file"] = self.data
        else:
            panel["content"] = self.data

        return panel

class AequilibraEMapPanel(SimwrapperPanel):
    """
    Panel for rendering interactive AequilibraE network maps.

    :Arguments:
        **title** (:obj:`str`): title
        **database** (:obj:`str`, *Optional*): project database
        **view** (:obj:`str`, *Optional*): panel view type
        **height** (:obj:`int`, *Optional*): panel height
        **width** (:obj:`int`, *Optional*): panel width
        **center** (:obj:`list`, *Optional*): map center coordinates
        **zoom** (:obj:`int`, *Optional*): initial zoom level
        **projection** (:obj:`str`, *Optional*): coordinate reference system

    :Example:
        panel = AequilibraEMapPanel(title, database, view, height, width, center, zoom, projection)
    """
    def __init__(self, title, database="project_database.sqlite", view="map", height=None, width=None, 
                         center=None, zoom=None, projection=None):
        super().__init__("aequilibrae", title, height=height, width=width)

        self.database = database
        self.view = view
        self.center = center
        self.zoom = zoom
        self.projection = projection

        self.defaults = None
        self.extra_databases = None
        self.layers = {}
        self.legend = None

    def set_defaults(self, defaults_dict):
        """ Sets default visuals for map layers"""
        self.defaults = defaults_dict

    def add_layer(self, name, layer_dict):
        """ Adds a layer definition under the given name"""
        self.layers[name] = layer_dict

    def set_legend(self, legend_list):
        """ Sets legend configuration for the map"""
        self.legend = legend_list

    def set_extra_databases(self, database_dict):
        """ Registers extra databases used by map """
        self.extra_databases = database_dict

    def to_dict(self):
        """Returns dictionary representation of the panel"""
        panel = super().to_dict()

        panel["database"] = self.database
        panel["view"] = self.view

        if self.center:
            panel["center"] = self.center

        if self.zoom:
            panel["zoom"] = self.zoom

        if self.projection:
            panel["projection"] = self.projection

        if self.defaults:
            panel["defaults"] = self.defaults

        if self.legend:
            panel["legend"] = self.legend

        if self.extra_databases:
            panel["extraDatabases"] = self.extra_databases

        if self.layers:
            panel["layers"] = self.layers

        return panel