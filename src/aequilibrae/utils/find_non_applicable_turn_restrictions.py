import pandas as pd
from typing import Any, Dict, Set, cast

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aequilibrae.project.project import Project


def _to_int(value: Any) -> int:
    return int(cast(Any, value))


def _to_float(value: Any) -> float:
    return float(cast(Any, value))


def _is_prohibited_penalty(value: Any) -> bool:
    if pd.isna(value):
        return True

    try:
        return _to_float(value) == float("inf")
    except (TypeError, ValueError):
        return False


def _directed_edges_from_links(links_df: pd.DataFrame) -> Dict[tuple[int, int], Set[str]]:
    directed: Dict[tuple[int, int], Set[str]] = {}

    for row_any in links_df.to_dict(orient="records"):
        row = cast(Dict[str, Any], row_any)
        a_node = _to_int(row["a_node"])
        b_node = _to_int(row["b_node"])
        direction = _to_int(row["direction"])
        modes = set(str(row.get("modes", "") or ""))

        if direction in (0, 1):
            directed.setdefault((a_node, b_node), set()).update(modes)
        if direction in (0, -1):
            directed.setdefault((b_node, a_node), set()).update(modes)

    return directed


def find_non_applicable_turn_restrictions(project: "Project") -> pd.DataFrame:
    """
    Returns turn restrictions/penalties that are not applicable to the current network.

    A restriction is flagged when one or more conditions hold:
    - from/via/to node does not exist
    - the directed movement legs (from_node->via_node and via_node->to_node) are missing
    - one or more restriction modes do not exist in modes table
    - restriction modes do not overlap with the available modes on one or both movement legs
    """
    columns = [
        "restriction_id",
        "from_node",
        "via_node",
        "to_node",
        "penalty",
        "modes",
        "turn_type",
        "issue",
    ]

    with project.db_connection as conn:
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='turn_restrictions'"
        ).fetchone()
        if table_exists is None:
            return pd.DataFrame(columns=columns)

        restrictions = pd.read_sql("SELECT * FROM turn_restrictions", conn)
        if restrictions.empty:
            return pd.DataFrame(columns=columns)

        links = pd.read_sql("SELECT a_node, b_node, direction, modes FROM links", conn)
        nodes_df = pd.read_sql("SELECT node_id FROM nodes", conn)
        modes_df = pd.read_sql("SELECT mode_id FROM modes", conn)

    valid_modes: Set[str] = {str(x) for x in modes_df["mode_id"].tolist()}
    valid_nodes: Set[int] = {_to_int(x) for x in nodes_df["node_id"].tolist()}
    directed_edges = _directed_edges_from_links(links)

    issues = []
    for row_any in restrictions.to_dict(orient="records"):
        row = cast(Dict[str, Any], row_any)
        restriction_modes_raw = str(row.get("modes", "") or "")
        restriction_modes: Set[str] = set(restriction_modes_raw)
        row_issues: list[str] = []

        from_node = _to_int(row["from_node"])
        via_node = _to_int(row["via_node"])
        to_node = _to_int(row["to_node"])

        if from_node not in valid_nodes:
            row_issues.append("missing_from_node")
        if via_node not in valid_nodes:
            row_issues.append("missing_via_node")
        if to_node not in valid_nodes:
            row_issues.append("missing_to_node")

        if from_node == via_node or via_node == to_node:
            row_issues.append("degenerate_turn_nodes")

        unknown_modes = restriction_modes - valid_modes
        if unknown_modes:
            row_issues.append("unknown_modes")

        if len(restriction_modes_raw) == 0:
            row_issues.append("empty_modes")

        from_leg_modes = directed_edges.get((from_node, via_node), set())
        to_leg_modes = directed_edges.get((via_node, to_node), set())

        if not from_leg_modes:
            row_issues.append("missing_from_leg")
        if not to_leg_modes:
            row_issues.append("missing_to_leg")

        if restriction_modes and from_leg_modes and not (restriction_modes & from_leg_modes):
            row_issues.append("no_mode_overlap_from_leg")
        if restriction_modes and to_leg_modes and not (restriction_modes & to_leg_modes):
            row_issues.append("no_mode_overlap_to_leg")

        penalty_val = row.get("penalty")
        if not _is_prohibited_penalty(penalty_val):
            try:
                if _to_float(penalty_val) < 0:
                    row_issues.append("negative_penalty")
            except (TypeError, ValueError):
                row_issues.append("invalid_penalty")

        if row_issues:
            issues.append(
                {
                    "restriction_id": _to_int(row["restriction_id"]),
                    "from_node": from_node,
                    "via_node": via_node,
                    "to_node": to_node,
                    "penalty": penalty_val,
                    "modes": restriction_modes_raw,
                    "turn_type": "ban" if _is_prohibited_penalty(penalty_val) else "penalty",
                    "issue": ";".join(row_issues),
                }
            )

    if not issues:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(issues, columns=columns)
