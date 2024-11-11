import warnings
from typing import Union, List

from shapely.geometry import Point, Polygon, LineString, MultiPoint, MultiPolygon, MultiLineString
from shapely.wkb import loads

from aequilibrae.utils.qgis_utils import inside_qgis, rtree_avail

if inside_qgis:
    from qgis.core import QgsSpatialIndex as Index
    from qgis.core import QgsGeometry, QgsFeature
else:
    from rtree import Index


class GeoIndex:
    """Implements a generic GeoIndex class that uses the QGIS index when using the GUI and RTree otherwise"""

    def __init__(self):
        self.idx = Index()
        self.built = False

    def build_from_layer(self, layer) -> dict:
        if inside_qgis:
            warnings.warn("This method works inside QGIS only")
        self.built = True
        self.idx = Index(layer.getFeatures())
        return {f.id(): loads(f.geometry().asWkb().data()) for f in layer.getFeatures()}

    def insert(
        self,
        feature_id: int,
        geometry: Union[Point, Polygon, LineString, MultiPoint, MultiPolygon, MultiLineString],
    ) -> None:
        """Inserts a valid shapely geometry in the index

        :Arguments:
            **feature_id** (:obj:`int`): ID of the geometry being inserted
            **geo** (:obj:`Shapely.geometry`): Any valid shapely geometry
        """
        self.built = True
        if inside_qgis:
            g = QgsGeometry()
            g.fromWkb(geometry.wkb)
            feature = QgsFeature()
            feature.setGeometry(g)
            feature.setId(feature_id)
            self.idx.addFeature(feature)
        elif rtree_avail:
            self.idx.insert(feature_id, geometry.bounds)
        else:
            warnings.warn("You need RTREE to build a spatial index")

    def nearest(self, geo: Union[Point, Polygon, LineString, MultiPoint, MultiPolygon], num_results) -> List[int]:
        """Finds nearest neighbor for a given geometry

        :Arguments:
            **geo** (:obj:`Shapely geometry`): Any valid shapely geometry

            **num_results** (:obj:`int`): A positive integer for the number of neighbors to return

        :Returns:
            **neighbors** (:obj:`List[int]`): List of IDs of the closest neighbors in the index
        """
        if inside_qgis:
            g = QgsGeometry()
            g.fromWkb(geo.wkb)
            return self.idx.nearestNeighbor(g, num_results)
        elif rtree_avail:
            return self.idx.nearest(geo.bounds, num_results)
        else:
            warnings.warn("You need RTREE to build a spatial index")

    def delete(self, feature_id, geometry: Union[Point, Polygon, LineString, MultiPoint, MultiPolygon]):
        if inside_qgis:
            g = QgsGeometry()
            g.fromWkb(geometry.wkb)
            feature = QgsFeature()
            feature.setGeometry(g)
            feature.setId(feature_id)
            self.idx.deleteFeature(feature)
        elif rtree_avail:
            self.idx.delete(feature_id, geometry.bounds)
        else:
            warnings.warn("You need RTREE to build a spatial index")

    def reset(self):
        self.idx = Index()
        self.built = False
