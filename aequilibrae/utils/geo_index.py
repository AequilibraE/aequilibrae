import importlib.util as iutil
import warnings
from typing import Union, List

from shapely.geometry import Point, Polygon, LineString, MultiPoint, MultiPolygon, MultiLineString
from shapely.wkb import loads

complete = iutil.find_spec("rtree") is not None
qgis = iutil.find_spec("qgis") is not None
if complete:
    from rtree.index import Index as Index

    env = "Python"
elif qgis:
    from qgis.core import QgsSpatialIndex as Index
    from qgis.core import QgsGeometry, QgsFeature

    env = "QGIS"
else:
    env = "NOT AVAILABLE"


class GeoIndex:
    """Implements a generic GeoIndex class that uses the QGIS index when using the GUI and RTree otherwise"""

    def __init__(self):
        self.idx = Index()
        self.built = False

    def build_from_layer(self, layer) -> dict:
        if env != "QGIS":
            warnings.warn("This method works inside QGIS only")
        self.built = True
        self.idx = Index(layer.getFeatures())
        return {f.id(): loads(f.geometry().asWkb().data()) for f in layer.getFeatures()}

    def insert(
            self, feature_id: int,
            geometry: Union[Point, Polygon, LineString, MultiPoint, MultiPolygon, MultiLineString]
    ) -> None:
        """Inserts a valid shapely geometry in the index

        Args:
            *feature_id* (:obj:`int`): ID of the geometry being inserted
            *geo* (:obj:`Shapely geometry`): Any valid shapely geometry
        """
        self.built = True
        if env == "QGIS":
            g = QgsGeometry()
            g.fromWkb(geometry.wkb)
            feature = QgsFeature()
            feature.setGeometry(g)
            feature.setId(feature_id)
            self.idx.addFeature(feature)
        elif env == "Python":
            self.idx.insert(feature_id, geometry.bounds)
        else:
            warnings.warn("You need RTREE to build a spatial index")

    def nearest(self, geo: Union[Point, Polygon, LineString, MultiPoint, MultiPolygon], num_results) -> List[int]:
        """Finds nearest neighbor for a given geometry

        Args:
            *geo* (:obj:`Shapely geometry`): Any valid shapely geometry
            *num_results* (:obj:`int`): A positive integer for the number of neighbors to return
        Return:
            *neighbors* (:obj:`List[int]`): List of IDs of the closest neighbors in the index
        """
        if env == "QGIS":
            g = QgsGeometry()
            g.fromWkb(geo.wkb)
            return self.idx.nearestNeighbor(g, num_results)
        elif env == "Python":
            return self.idx.nearest(geo.bounds, num_results)
        else:
            warnings.warn("You need RTREE to build a spatial index")

    def delete(self, feature_id, geometry: Union[Point, Polygon, LineString, MultiPoint, MultiPolygon]):
        if env not in ["QGIS", "Python"]:
            warnings.warn("You need RTREE to build a spatial index")
        self.idx.delete(feature_id, geometry.bounds)

    def reset(self):
        self.idx = Index()
        self.built = False
