import logging
import sqlite3
from typing import Any, List, Optional

import geopandas as gpd
import pandas as pd

from aequilibrae.project.basic_table import BasicTable
from aequilibrae.project.data_loader import DataLoader

_UNSET = object()

logger = logging.getLogger(__name__)


class TurnRestrictions(BasicTable):
    """
    Provides an interface for managing turn restrictions in an AequilibraE project.

    A turn is defined as a node sequence ``from_node -> via_node -> to_node``.
    """

    def __init__(self, network):
        super().__init__(network.project)
        self.__table_type__ = "turn_restrictions"
        self._network = network

    def _canonicalize_modes(self, conn, modes: str) -> str:
        if modes is None:
            raise ValueError("modes cannot be None")

        mode_rows = conn.execute("SELECT mode_id FROM modes ORDER BY mode_id").fetchall()
        valid_modes = [row[0] for row in mode_rows]
        requested = set(str(modes))
        unknown = requested - set(valid_modes)
        if unknown:
            raise ValueError(f"Unknown mode IDs in modes: {sorted(unknown)}")

        canonical = "".join([mode for mode in valid_modes if mode in requested])
        if len(canonical) == 0:
            raise ValueError("modes must include at least one valid mode")
        return canonical

    def _default_modes(self, conn) -> str:
        mode_rows = conn.execute("SELECT mode_id FROM modes ORDER BY mode_id").fetchall()
        modes = "".join(row[0] for row in mode_rows)
        if len(modes) == 0:
            raise ValueError("No modes available in project")
        return modes

    @staticmethod
    def _normalise_penalty(penalty: Any) -> Optional[float]:
        if penalty is None or pd.isna(penalty):
            return None

        try:
            value = float(penalty)
        except (TypeError, ValueError) as exc:
            raise ValueError("penalty must be None/+inf for a prohibition or a non-negative finite cost") from exc

        if value < 0:
            raise ValueError("Negative turn penalties are not allowed. Use None or +inf for a prohibition")
        elif value == float("inf"):
            return None
        return value

    def add_restriction(
        self,
        from_node: int,
        via_node: int,
        to_node: int,
        penalty: Optional[float] = None,
        modes: Optional[str] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> int:
        """
        Adds a turn restriction to the network.

        :Arguments:
            **from_node** (:obj:`int`): Incoming movement origin node

            **via_node** (:obj:`int`): Turn node

            **to_node** (:obj:`int`): Outgoing movement destination node

            **penalty** (:obj:`float`, *Optional*): Turn penalty in the same time unit as the graph cost.
                ``None``, ``NaN`` or ``+inf`` mean the turn is prohibited. Negative values are not allowed.
                Defaults to ``None`` (prohibited).

            **modes** (:obj:`str`, *Optional*): Concatenated mode IDs this restriction applies to.
                If omitted, applies to all registered modes.

        :Returns:
            **restriction_id** (:obj:`int`): The ID of the newly created restriction
        """
        with conn or self.project.db_connection_spatial as conn:
            modes_txt = self._default_modes(conn) if modes is None else self._canonicalize_modes(conn, modes)
            penalty_value = self._normalise_penalty(penalty)

            cursor = conn.execute(
                """INSERT INTO turn_restrictions (from_node, via_node, to_node, penalty, modes)
                   VALUES (?, ?, ?, ?, ?)""",
                (from_node, via_node, to_node, penalty_value, modes_txt),
            )
            restriction_id = cursor.lastrowid
            assert isinstance(restriction_id, int)

            logger.info(
                f"Added turn restriction {restriction_id}: {from_node} -> {via_node} -> {to_node} [{modes_txt}]"
            )
        return restriction_id

    def add_restrictions_from_dataframe(self, df: pd.DataFrame) -> List[int]:
        """
        Adds multiple turn restrictions from a DataFrame.

        The DataFrame must have columns: 'from_node', 'via_node', 'to_node'.
        Optional columns: 'penalty', 'modes'.
        """
        required_cols = {"from_node", "via_node", "to_node"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"DataFrame missing required columns: {missing}")

        restriction_ids = []
        with self.project.db_connection_spatial as conn:
            for _, row in df.iterrows():
                penalty = row.get("penalty", None)
                restriction_ids.append(
                    self.add_restriction(
                        from_node=int(row["from_node"]),
                        via_node=int(row["via_node"]),
                        to_node=int(row["to_node"]),
                        penalty=penalty if pd.notna(penalty) else None,
                        modes=row.get("modes", None) if pd.notna(row.get("modes", None)) else None,
                        conn=conn,
                    )
                )
        return restriction_ids

    def remove_restriction(self, restriction_id: int, conn: Optional[sqlite3.Connection] = None) -> bool:
        with conn or self.project.db_connection_spatial as conn:
            cursor = conn.execute("DELETE FROM turn_restrictions WHERE restriction_id = ?", (restriction_id,))
            if cursor.rowcount > 0:
                logger.info(f"Removed turn restriction {restriction_id}")
                return True
            logger.warning(f"Turn restriction {restriction_id} not found")
            return False

    def clear_restrictions(self, conn: Optional[sqlite3.Connection] = None) -> int:
        with conn or self.project.db_connection_spatial as conn:
            cursor = conn.execute("DELETE FROM turn_restrictions")
            count = cursor.rowcount
            logger.info(f"Cleared {count} turn restrictions")
        return count

    @property
    def data(self) -> gpd.GeoDataFrame:
        """Returns all turn restrictions data as a GeoPandas DataFrame

        :Returns:
            **table** (:obj:`GeoDataFrame`): GeoPandas GeoDataFrame with all the nodes
        """
        dl = DataLoader(self.project.path_to_file, "turn_restrictions")
        result = dl.load_table()
        assert isinstance(result, gpd.GeoDataFrame), "Turn restrictions table must have geometry"
        return result

    def get_restriction(self, restriction_id: int, conn: Optional[sqlite3.Connection] = None) -> Optional[dict]:
        with conn or self.project.db_connection_spatial as conn:
            cursor = conn.execute("SELECT * FROM turn_restrictions WHERE restriction_id = ?", (restriction_id,))
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row, strict=False))
        return None

    def update_restriction(
        self,
        restriction_id: int,
        penalty: Any = _UNSET,
        modes: Optional[str] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> bool:
        with conn or self.project.db_connection_spatial as conn:
            updates = []
            values: List[Any] = []

            if penalty is not _UNSET:
                updates.append("penalty = ?")
                values.append(self._normalise_penalty(penalty))

            if modes is not None:
                canonical_modes = self._canonicalize_modes(conn, modes)
                updates.append("modes = ?")
                values.append(canonical_modes)

            if not updates:
                return False

            values.append(restriction_id)

            cursor = conn.execute(
                f"UPDATE turn_restrictions SET {', '.join(updates)} WHERE restriction_id = ?",
                values,
            )
            if cursor.rowcount > 0:
                logger.info(f"Updated turn restriction {restriction_id}")
                return True
        return False

    def count(self, conn: Optional[sqlite3.Connection] = None) -> int:
        with conn or self.project.db_connection_spatial as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM turn_restrictions")
            return cursor.fetchone()[0]

    def __len__(self) -> int:
        return self.count()
