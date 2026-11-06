# First and last extended GTFS route type of each family, with the basic route type it maps onto
EXTENDED_ROUTE_TYPES = [
    (100, 117, 2),  # Railway
    (200, 209, 3),  # Coach
    (400, 404, 1),  # Urban railway and metro
    (405, 405, 12),  # Monorail
    (700, 716, 3),  # Bus
    (800, 800, 11),  # Trolleybus
    (900, 906, 0),  # Tram
    (1000, 1000, 4),  # Water transport
    (1200, 1200, 4),  # Ferry
    (1300, 1307, 6),  # Aerial lift
    (1400, 1400, 7),  # Funicular
]


def base_route_type(route_type: int) -> int:
    """Returns the basic GTFS route type of an extended one, and any other route type unchanged"""
    route_type = int(route_type)
    return next((basic for first, last, basic in EXTENDED_ROUTE_TYPES if first <= route_type <= last), route_type)


def lookup_by_route_type(table: dict, route_type: int, default=None):
    """Returns the value for a route type, falling back to that of its basic type and then to ``default``"""
    return table.get(route_type, table.get(base_route_type(route_type), default))
