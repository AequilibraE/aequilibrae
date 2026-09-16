from collections.abc import Mapping
from typing import Any

import pandas as pd

from aequilibrae.project.project_table import SpatialProjectTable


class TurnRestrictions(SpatialProjectTable):
    """Manage turns defined by ``from_node -> via_node -> to_node``.

    Use ``insert`` and ``update`` for single records, or ``insert_from`` and
    ``update_from`` for DataFrames. Penalties use the same units as the graph
    cost. ``None``, ``NaN`` and positive infinity mean the turn is prohibited;
    negative penalties are not allowed.

    Supply modes on insert. Database triggers reject missing, empty, unknown
    or repeated mode IDs and overlapping restrictions for the same turn.
    Updates leave omitted fields unchanged.

    Records and ``data`` include the geometry built by the database from the
    three nodes. Use ``project.transaction()`` to group several writes.
    """

    name = "turn_restrictions"
    key = "restriction_id"
    record_name = "TurnRestrictionRecord"
    has_numeric_key = True

    def _prepare_insert(self, values: Mapping[str, Any]) -> dict[str, Any]:
        row = super()._prepare_insert(values)
        row["penalty"] = self.__normalise_penalty(values.get("penalty"))
        return row

    def update(self, key: int, **values: Any) -> None:
        """Update supplied fields. Set penalty to None to prohibit a turn."""
        if "penalty" in values:
            values["penalty"] = self.__normalise_penalty(values["penalty"])
        super().update(key, **values)

    def _prepare_rows(self, frame: pd.DataFrame, value_columns: tuple[str, ...]) -> list[tuple[Any, ...]]:
        if "penalty" in value_columns:
            try:
                # ``astype`` cannot convert pandas' scalar ``NA`` directly;
                # ``to_numeric`` consistently normalises it to NaN. The textual
                # NaN spellings are prohibitions too, while other non-numeric
                # strings still raise.
                raw_penalties = frame["penalty"].replace(["nan", "+nan", "-nan"], float("nan"))
                penalties = pd.to_numeric(raw_penalties, errors="raise").astype("float64")
            except (TypeError, ValueError) as exc:
                raise ValueError("penalty must be None/+inf for a prohibition or a non-negative finite cost") from exc

            if penalties.lt(0).any():
                raise ValueError("Negative turn penalties are not allowed. Use None or +inf for a prohibition")

            prohibited = penalties.isna() | penalties.eq(float("inf"))
            # Use object dtype so prohibitions reach SQLite as None, not pandas missing values.
            frame = frame.assign(penalty=penalties.astype(object).mask(prohibited, None))
        return super()._prepare_rows(frame, value_columns)

    def clear_restrictions(self) -> int:
        """Delete all restrictions and return the number removed."""
        with self._connection as conn:
            count = conn.execute("DELETE FROM turn_restrictions").rowcount
        self._invalidate()
        return count

    @staticmethod
    def __normalise_penalty(penalty: Any) -> float | None:
        if penalty is None or pd.isna(penalty):
            return None

        try:
            value = float(penalty)
        except (TypeError, ValueError) as exc:
            raise ValueError("penalty must be None/+inf for a prohibition or a non-negative finite cost") from exc

        if value < 0:
            raise ValueError("Negative turn penalties are not allowed. Use None or +inf for a prohibition")
        if pd.isna(value) or value == float("inf"):
            return None
        return value
