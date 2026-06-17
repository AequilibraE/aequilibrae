"""Legacy package retained for back-compat import paths.

The OSM importer has moved to
``aequilibrae.project.network.importer.sources.osm`` and is invoked via
``Network.import_from_osm(...)``. The old ``placegetter``, ``OSMDownloader``,
and ``OSMBuilder`` symbols are no longer available.
"""

__all__: list[str] = []
