"""Which scheduled trips belong to a period.

Times count seconds from the start of the GTFS service day, so they may exceed ``86400``: a trip that leaves at
``25:10:00`` is stored as ``90600``. Periods use the same axis and never wrap around midnight.
"""

import sqlite3

# The trip-level rules a trip is assigned to a period by
TRIP_INCLUSION_RULES = ("start", "end", "midpoint", "any")

# Seconds in a service day. A period longer than this still holds at most one service day of trips.
SERVICE_DAY = 86400


def period_window(conn: sqlite3.Connection, period_id: int) -> tuple[int, int]:
    """Return the start and end of a period, in seconds from the start of the service day."""
    row = conn.execute("SELECT period_start, period_end FROM periods WHERE period_id = ?;", [period_id]).fetchone()
    if row is None:
        raise ValueError(f"period_id {period_id} does not exist")
    return int(row[0]), int(row[1])


def period_duration(start: float, end: float) -> float:
    """Return the length over which a period's trips are averaged into frequencies.

    The whole-day default period runs to ``48:00:00`` so that it keeps every trip of the service day, but its trips
    still describe a single day. An empty period holds no trips, so its length is never divided by.
    """
    if end < start:
        raise ValueError("a period cannot end before it starts")
    return float(min(end - start, SERVICE_DAY))


def trips_in_period_sql(start: float, end: float, inclusion_cond: str = "start") -> str:
    """Return a query selecting the ``trip_id`` of every trip assigned to the half-open window ``[start, end)``.

    ``start`` and ``end`` use the trip's first departure and last arrival, ``midpoint`` their mean, and ``any``
    includes a trip with any arrival or departure inside the window. Each of ``start``, ``end`` and ``midpoint``
    assigns every trip to exactly one of a set of consecutive periods.
    """
    if inclusion_cond not in TRIP_INCLUSION_RULES:
        raise ValueError(f"inclusion_cond must be one of {TRIP_INCLUSION_RULES}")

    start, end = float(start), float(end)
    if inclusion_cond == "any":
        return f"""
            SELECT DISTINCT trip_id FROM trips_schedule
            WHERE (arrival >= {start} AND arrival < {end}) OR (departure >= {start} AND departure < {end})
        """

    probe = {
        "start": "MIN(departure)",
        "end": "MAX(arrival)",
        "midpoint": "(MIN(departure) + MAX(arrival)) / 2.0",
    }[inclusion_cond]
    return f"""
        SELECT trip_id FROM trips_schedule GROUP BY trip_id
        HAVING {probe} >= {start} AND {probe} < {end}
    """
