"""OpenStreetMap sources (Overpass + .osm.pbf).

XML / .osm / .osm.bz2 is not supported: convert with
``osmium cat in.osm -o out.osm.pbf`` then use the PBF source.
"""

from .overpass import OSMOverpassSource
from .pbf import OSMPbfSource

__all__ = ["OSMOverpassSource", "OSMPbfSource"]
