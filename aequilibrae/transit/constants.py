from typing import Dict, Any

AGENCY_MULTIPLIER = 10_000_000_000
ROUTE_ID_MULTIPLIER = 1000000
PATTERN_ID_MULTIPLIER = 1000
TRIP_ID_MULTIPLIER = 1

TRANSIT_STOP_RANGE = 1000000
WALK_LINK_RANGE = 30000000
TRANSIT_LINK_RANGE = 20000000
WALK_AGENCY_ID = 1

# 1 for right, -1 for wrong (left)
DRIVING_SIDE = 1


class Constants:
    agencies: Dict[str, Any] = {}
    srid: Dict[int, int] = {}
    routes: Dict[int, int] = {}
    trips: Dict[int, int] = {}
    patterns: Dict[int, int] = {}
    pattern_lookup: Dict[int, int] = {}
    stops: Dict[int, int] = {}
    fares: Dict[int, int] = {}
    links: Dict[int, int] = {}
    transit_links: Dict[int, int] = {}
