"""OSM tag parsing for staged networks."""

import re
from typing import Mapping

from aequilibrae.project.network.importer.schema.modes import MODE_CODE


def _has(tags: Mapping, key: str, *values: str) -> bool:
    v = tags.get(key)
    if v is None:
        return False
    return any(part.strip() in values for part in str(v).lower().split(";"))


def _denied(tags: Mapping, key: str, include_private: bool = False) -> bool:
    return _has(tags, key, "no") or (
        not include_private
        and _has(tags, key, "private", "destination", "customers", "forestry", "agricultural", "delivery", "permit")
    )


def _explicit_allowed(tags: Mapping, key: str) -> bool:
    return _has(tags, key, "yes", "designated", "permissive", "official")


def _allow_car(tags: Mapping, include_private: bool) -> bool:
    highway = str(tags.get("highway", "")).lower()
    if not highway:
        return False
    pedestrian_only = {
        "footway",
        "pedestrian",
        "steps",
        "path",
        "cycleway",
        "bridleway",
        "corridor",
        "elevator",
        "escalator",
        "via_ferrata",
    }
    if not include_private and _denied(tags, "access") and not _explicit_allowed(tags, "motor_vehicle"):
        return False
    if _denied(tags, "motor_vehicle", include_private) or _denied(tags, "motorcar", include_private):
        return False
    if _denied(tags, "vehicle", include_private) and not _explicit_allowed(tags, "motor_vehicle"):
        return False
    if highway in pedestrian_only:
        return _explicit_allowed(tags, "motor_vehicle") or _explicit_allowed(tags, "vehicle")
    if (
        not include_private
        and highway == "service"
        and _has(tags, "service", "parking_aisle", "driveway", "private", "emergency_access")
    ):
        return False
    return True


def _allow_walk(tags: Mapping, include_private: bool) -> bool:
    highway = str(tags.get("highway", "")).lower()
    if not highway:
        return False
    motor_only = {"motorway", "motorway_link", "trunk", "trunk_link"}
    if highway in motor_only:
        return _explicit_allowed(tags, "foot")
    if not include_private and _denied(tags, "access") and not _explicit_allowed(tags, "foot"):
        return False
    if _denied(tags, "foot", include_private):
        return False
    return True


def _allow_bicycle(tags: Mapping, include_private: bool) -> bool:
    highway = str(tags.get("highway", "")).lower()
    if not highway:
        return False
    motor_only = {"motorway", "motorway_link"}
    if highway in motor_only:
        return _explicit_allowed(tags, "bicycle")
    pedestrian_blocked = {"footway", "steps", "corridor", "elevator", "escalator"}
    if highway in pedestrian_blocked:
        return _explicit_allowed(tags, "bicycle")
    if not include_private and _denied(tags, "access") and not _explicit_allowed(tags, "bicycle"):
        return False
    if _denied(tags, "bicycle", include_private):
        return False
    return True


def _allow_transit(tags: Mapping, include_private: bool) -> bool:
    """Bus-capable highways (proxy for 'transit')."""
    highway = str(tags.get("highway", "")).lower()
    if not highway:
        return False
    bus_capable = {
        "bus_guideway",
        "busway",
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
        "unclassified",
        "residential",
        "living_street",
        "service",
        "road",
    }
    if highway not in bus_capable:
        return False
    if not include_private and _denied(tags, "access") and not _explicit_allowed(tags, "bus"):
        return False
    if _denied(tags, "bus", include_private):
        return False
    if any(_denied(tags, key, include_private) for key in ("vehicle", "motor_vehicle", "psv")) and not (
        _explicit_allowed(tags, "bus") or _explicit_allowed(tags, "psv")
    ):
        return False
    if (
        not include_private
        and highway == "service"
        and _has(tags, "service", "parking_aisle", "driveway", "private", "emergency_access")
    ):
        return False
    return True


MODE_RULES = {
    MODE_CODE["car"]: _allow_car,
    MODE_CODE["transit"]: _allow_transit,
    MODE_CODE["bicycle"]: _allow_bicycle,
    MODE_CODE["walk"]: _allow_walk,
}


def modes_for_tags(tags: Mapping, include_private: bool = False) -> str:
    if not include_private and _has(tags, "service", "private"):
        return ""
    return "".join(sorted(code for code, predicate in MODE_RULES.items() if predicate(tags, include_private)))


_SQL_KEY_RE = re.compile(r"[^a-zA-Z0-9_]+")


def normalise_tag_key(key: str) -> str:
    """Replace punctuation with underscores in OSM attribute names."""
    if not key:
        return ""
    cleaned = _SQL_KEY_RE.sub("_", str(key))
    return cleaned.lstrip("_") or "_unnamed"


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


_SPEED_RE = re.compile(r"\s*([0-9]+(?:\.[0-9]+)?)\s*(km/h|kmh|kph|mph|knots)?\s*", re.IGNORECASE)


def parse_speed(value) -> float | None:
    """Parse a single speed and optional unit into km/h; otherwise return None."""
    m = _SPEED_RE.fullmatch(str(value))
    if not m:
        return None
    magnitude = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit == "mph":
        return magnitude * 1.609344
    if unit == "knots":
        return magnitude * 1.852
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
        try:
            return int(float(str(value).split(";")[0]))
        except ValueError:
            return None

    total = _as_int(tags.get("lanes"))
    fwd = _as_int(tags.get("lanes:forward"))
    bwd = _as_int(tags.get("lanes:backward"))
    direction = parse_direction(tags)
    if direction == 1:
        return (fwd if fwd is not None else total), None
    if direction == -1:
        return None, (bwd if bwd is not None else total)

    if fwd is not None and bwd is not None:
        return fwd, bwd
    if fwd is not None:
        other = (total - fwd) if (total is not None and total - fwd >= 1) else fwd
        return fwd, other
    if bwd is not None:
        other = (total - bwd) if (total is not None and total - bwd >= 1) else bwd
        return other, bwd

    return _split_total_lanes(total)


def _split_total_lanes(total: int | None) -> tuple[int | None, int | None]:
    """Split total lanes, assigning odd remainders to AB and sharing a single lane."""
    if total is None or total <= 1:
        return total, total
    return (total + 1) // 2, total // 2
