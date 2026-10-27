"""Mapping of extended GTFS route types onto the basic ones.

Feeds may use the extended route types of the Hierarchical Vehicle Type (HVT) list, such as ``109`` for suburban
rail or ``700`` for bus service, instead of the basic types ``0`` to ``12`` of the GTFS reference. Default capacities,
PCEs, maximum speeds and map-matching eligibility are defined for the basic types, so extended types are looked up
through their basic type.
"""

# First and last extended code of each family, with the basic type it maps to
_EXTENDED_FAMILIES = [
    (100, 117, 2),  # Railway services
    (200, 209, 3),  # Coach services
    (400, 404, 1),  # Urban railway and metro services
    (405, 405, 12),  # Monorail
    (700, 716, 3),  # Bus services
    (800, 800, 11),  # Trolleybus service
    (900, 906, 0),  # Tram services
    (1000, 1000, 4),  # Water transport service
    (1200, 1200, 4),  # Ferry service
    (1300, 1307, 6),  # Aerial lift services
    (1400, 1400, 7),  # Funicular service
]


def base_route_type(route_type: int) -> int:
    """Returns the basic GTFS route type for a basic or extended route type.

    Basic types (``0`` to ``12``) and extended types without a basic equivalent (such as ``1100`` for air
    services or ``1500`` for taxis) are returned unchanged.

    :Arguments:
        **route_type** (:obj:`int`): Route type as read from ``routes.txt``

    :Returns:
        **route_type** (:obj:`int`): Corresponding basic route type

    .. code-block:: python

        >>> from aequilibrae.transit.route_types import base_route_type

        >>> base_route_type(109), base_route_type(700), base_route_type(3), base_route_type(1500)
        (2, 3, 3, 1500)
    """
    route_type = int(route_type)
    for first, last, basic in _EXTENDED_FAMILIES:
        if first <= route_type <= last:
            return basic
    return route_type


def lookup_by_route_type(table: dict, route_type: int, default=None):
    """Returns the value for a route type, falling back to its basic type and then to ``default``.

    :Arguments:
        **table** (:obj:`dict`): Values keyed by route type

        **route_type** (:obj:`int`): Route type as read from ``routes.txt``

        **default** (*Optional*): Value returned when neither the route type nor its basic type is in ``table``
    """
    if route_type in table:
        return table[route_type]
    return table.get(base_route_type(route_type), default)
