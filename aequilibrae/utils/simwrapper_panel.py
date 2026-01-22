class SimwrapperPanel():
    """ Base class for all simwrapper panels"""
    def __init__(self, type, title, height=None, width=None):
        self.type = type
        self.title = title
        self.height = height
        self.width = width

    def to_dict(self):
        """Returns dictionairy representation of the panel"""
        panel = {
            "type": self.type,
            "title": self.title,
        }

        if self.height:
            panel["height"] = self.height

        if self.width:
            panel["width"] = self.width

        return panel

class TilePanel(SimwrapperPanel):
    def __init__(self, title, dataset, height=None, width=None):
        super().__init__("tile", title, height=height, width=width)
        self.dataset = dataset

    def to_dict(self):
        """Returns dictionairy representation of the panel"""
        panel = super().to_dict()

        panel["dataset"] = self.dataset

        return panel

# my = TilePanel(title, dataset)

class TextPanel(SimwrapperPanel):
    def __init__(self, title, data, is_file=False, height=None, width=None):
        super().__init__("text", title, height=height, width=width)
        self.data = data
        self.is_file = is_file

    def to_dict(self):
        """Returns dictionairy representation of the panel"""
        panel = super().to_dict()

        if self.is_file:
            panel["file"] = self.data
        else:
            panel["content"] = self.data

        return panel

# my = TextPanel(title, data=text, is_file=False, height=None, width=None)    

# my = AequilibraEMapPanel(title, database, view, height, width, center, zoom, projection) 
class AequilibraEMapPanel(SimwrapperPanel):
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
        """Returns dictionairy representation of the panel"""
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