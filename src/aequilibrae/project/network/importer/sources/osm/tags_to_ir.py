"""OSM tag → staged-network conversion (typed fields + mode rules).

There is no ``highway`` allow-list filter. Every value of ``highway`` is
preserved as ``link_type``. Mode assignment uses access semantics only
(``access``, ``motor_vehicle``, ``bicycle``, ``foot``, ``vehicle``,
``service``, ``junction``, …).
"""

import re
from typing import Mapping

from aequilibrae.project.network.importer.schema.modes import ModeRule


# AequilibraE default mode codes (matches parameters.yml)
MODE_CODE = {
    "car": "c",
    "transit": "t",
    "bicycle": "b",
    "walk": "w",
}


def _has(tags: Mapping, key: str, *values: str) -> bool:
    v = tags.get(key)
    if v is None:
        return False
    return str(v).lower() in values


def _denied(tags: Mapping, key: str) -> bool:
    """A key is 'denied' if its value is in the deny-list."""
    return _has(tags, key, "no", "private", "destination", "customers", "forestry", "agricultural", "delivery")


def _explicit_allowed(tags: Mapping, key: str) -> bool:
    return _has(tags, key, "yes", "designated", "permissive", "official")


# --- Mode predicates ---

def _allow_car(tags: Mapping) -> bool:
    highway = str(tags.get("highway", "")).lower()
    if not highway:
        return False
    # Hard exclusions for pedestrian/cycle-only ways unless explicitly allowed
    pedestrian_only = {"footway", "pedestrian", "steps", "path", "cycleway", "bridleway",
                       "corridor", "elevator", "escalator", "via_ferrata"}
    if highway in pedestrian_only:
        return _explicit_allowed(tags, "motor_vehicle") or _explicit_allowed(tags, "vehicle")
    # Generic access denials
    if _denied(tags, "access") and not _explicit_allowed(tags, "motor_vehicle"):
        return False
    if _denied(tags, "motor_vehicle"):
        return False
    if _denied(tags, "vehicle") and not _explicit_allowed(tags, "motor_vehicle"):
        return False
    # service=parking_aisle/driveway/private gets dropped
    if highway == "service" and _has(tags, "service",
                                     "parking_aisle", "driveway", "private", "emergency_access"):
        return False
    return True


def _allow_walk(tags: Mapping) -> bool:
    highway = str(tags.get("highway", "")).lower()
    if not highway:
        return False
    motor_only = {"motorway", "motorway_link", "trunk", "trunk_link"}
    if highway in motor_only:
        return _explicit_allowed(tags, "foot")
    if _denied(tags, "access") and not _explicit_allowed(tags, "foot"):
        return False
    if _denied(tags, "foot"):
        return False
    return True


def _allow_bicycle(tags: Mapping) -> bool:
    highway = str(tags.get("highway", "")).lower()
    if not highway:
        return False
    motor_only = {"motorway", "motorway_link"}
    if highway in motor_only:
        return _explicit_allowed(tags, "bicycle")
    pedestrian_blocked = {"footway", "steps", "corridor", "elevator", "escalator"}
    if highway in pedestrian_blocked:
        return _explicit_allowed(tags, "bicycle")
    if _denied(tags, "access") and not _explicit_allowed(tags, "bicycle"):
        return False
    if _denied(tags, "bicycle"):
        return False
    return True


def _allow_transit(tags: Mapping) -> bool:
    """Bus-capable highways (proxy for 'transit')."""
    highway = str(tags.get("highway", "")).lower()
    if not highway:
        return False
    if highway in {"bus_guideway", "busway"}:
        return True
    # Most road types are bus-capable; mimic the legacy parameters.yml transit set.
    bus_capable = {
        "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
        "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
        "residential", "living_street", "service", "road",
    }
    if highway not in bus_capable:
        return False
    if _denied(tags, "access") and not _explicit_allowed(tags, "bus"):
        return False
    if _has(tags, "psv", "no") and not _explicit_allowed(tags, "bus"):
        return False
    if highway == "service" and _has(tags, "service",
                                     "parking_aisle", "driveway", "private", "emergency_access"):
        return False
    return True


MODE_RULES: list[ModeRule] = [
    ModeRule("car", MODE_CODE["car"], _allow_car),
    ModeRule("transit", MODE_CODE["transit"], _allow_transit),
    ModeRule("bicycle", MODE_CODE["bicycle"], _allow_bicycle),
    ModeRule("walk", MODE_CODE["walk"], _allow_walk),
]


# --- Tag normalisation ---

_SQL_KEY_RE = re.compile(r"[^a-zA-Z0-9_]+")


def normalise_tag_key(key: str) -> str:
    """Make an OSM tag key safe to use as a SQL column name.

    Replaces ``:`` and any other non-alphanumeric character with ``_``.
    """
    if not key:
        return ""
    cleaned = _SQL_KEY_RE.sub("_", str(key))
    # Strip leading underscores (so user-facing tags don't look like IR scratch)
    return cleaned.lstrip("_") or "_unnamed"


# --- Direction / lanes / speed parsers ---

def parse_direction(tags: Mapping) -> int:
    """OSM oneway / junction=roundabout → AequilibraE direction (-1/0/1)."""
    oneway = str(tags.get("oneway", "")).lower()
    junction = str(tags.get("junction", "")).lower()
    if oneway in ("yes", "true", "1"):
        return 1
    if oneway in ("-1", "reverse"):
        return -1
    if oneway in ("no", "false", "0"):
        return 0
    if junction == "roundabout":
        return 1
    return 0


_SPEED_RE = re.compile(r"\s*([0-9.]+)\s*(km/h|kmh|kph|mph|knots)?\s*", re.IGNORECASE)


def parse_speed(value) -> float | None:
    """Parse an OSM maxspeed tag (e.g. ``"50"``, ``"30 mph"``) into km/h."""
    if value is None:
        return None
    s = str(value)
    m = _SPEED_RE.match(s)
    if not m:
        return None
    try:
        magnitude = float(m.group(1))
    except (TypeError, ValueError):
        return None
    unit = (m.group(2) or "").lower().replace(" ", "")
    if unit == "mph":
        return magnitude * 1.609344
    if unit == "knots":
        return magnitude * 1.852
    # km/h, kmh, kph, or unitless → assume km/h
    return magnitude


def directional_speeds(tags: Mapping) -> tuple[float | None, float | None]:
    """Return ``(speed_ab, speed_ba)`` from OSM tags.

    Uses ``maxspeed:forward`` / ``maxspeed:backward`` when present and falls
    back to ``maxspeed`` for both directions.
    """
    speed = parse_speed(tags.get("maxspeed"))
    fwd = parse_speed(tags.get("maxspeed:forward")) or speed
    bwd = parse_speed(tags.get("maxspeed:backward")) or speed
    direction = parse_direction(tags)
    if direction == 1:
        return fwd, None
    if direction == -1:
        return None, bwd
    return fwd, bwd


def directional_lanes(tags: Mapping) -> tuple[int | None, int | None]:
    """Return ``(lanes_ab, lanes_ba)`` from OSM tags."""

    def _as_int(value):
        if value is None:
            return None
        try:
            return int(float(str(value).split(";")[0]))
        except (TypeError, ValueError):
            return None

    total = _as_int(tags.get("lanes"))
    fwd = _as_int(tags.get("lanes:forward"))
    bwd = _as_int(tags.get("lanes:backward"))
    direction = parse_direction(tags)
    if direction == 1:
        return (fwd if fwd is not None else total), None
    if direction == -1:
        return None, (bwd if bwd is not None else total)
    # Bidirectional: use explicit forward/backward when available;
    # otherwise propagate the total to both directions (the legacy "divide by 2"
    # was a bug; OSM lanes is typically the total for the carriageway).
    return (fwd if fwd is not None else total), (bwd if bwd is not None else total)
