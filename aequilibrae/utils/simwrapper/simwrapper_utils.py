import math
import csv
import json
from pathlib import Path


def pretty_round(value, direction="up"):
    """Round a value to a 'pretty' number (1, 2, 5 multiples of powers of 10).

    :Arguments:
        **value** (:obj:`float`): value to round
        **direction** (:obj:`str`): 'up' to ceil, 'down' to floor

    :Returns:
        **float**: the rounded pretty number
    """
    if value == 0:
        return 0

    sign = 1 if value >= 0 else -1
    abs_val = abs(value)

    exponent = math.floor(math.log10(abs_val))
    mantissa = abs_val / (10**exponent)

    pretty_steps = [1, 2, 5, 10]

    if direction == "up":
        chosen = next((s for s in pretty_steps if s >= mantissa), 10)
    else:
        chosen = next((s for s in reversed(pretty_steps) if s <= mantissa), 1)

    result = sign * chosen * (10**exponent)

    # Snap to zero if the result is very small compared to a "normal" scale
    if abs(result) < 1e-10:
        return 0

    return result


def get_links_bounds_box(project):
    """
    Compute box around all coordinates in links table of project.
    Queries spatial database to find max and min x and y coords across all link geometries
    to return overall network links' reach.

    Returns bounding box values (xmin, ymin, xmax, ymax)

    """
    with project.db_connection_spatial as conn:
        cursor = conn.cursor()  # database cursor to make sql query

        # compute box around all coordinates in links table of project
        cursor.execute(
            """
        SELECT
            MIN(MBRMinX(geometry)) AS xmin,
            MIN(MBRMinY(geometry)) AS ymin,
            MAX(MBRMaxX(geometry)) AS xmax,
            MAX(MBRMaxY(geometry)) AS ymax
        FROM links
        """
        )

        row = cursor.fetchone()  # fetch the single row returned by query (ie bounding box values)
    return row


def get_project_center(project):
    """Finds center coordinates of project"""
    row = get_links_bounds_box(project)

    if row is None or any(value is None for value in row):
        return [
            0,
            0,
        ]  # If coordinates cannot be determined (missing link values), return a fallback [0, 0].

    xmin, ymin, xmax, ymax = row

    # find center on each axis
    center = [(xmin + xmax) / 2, (ymin + ymax) / 2]  # [horizontal center, vertical center] == [longitude ,latitude]

    return center


def get_project_zoom(project):
    """Finds a reasonable zoom level based on project links' reach"""

    # just to keep things reasonable
    max_zoom = 15
    min_zoom = 5

    row = get_links_bounds_box(project)

    if row is None or any(value is None for value in row):
        return 10  # If bounding box can't be determined, return a default zoom level.

    xmin, ymin, xmax, ymax = row

    x_span = abs(xmax - xmin)
    y_span = abs(ymax - ymin)

    max_span = max(x_span, y_span)  # use larger of two so we see everything

    if max_span <= 0:
        return 10  # If max_span is non-positive, return the default zoom level.

    # calculate ~ zoom:
    # at zoom of 0 the world is ~360degrees wide
    # each increment doubles the resolution
    zoom = int(round(math.log2(360 / max_span)))

    # fix this within the allowed range
    zoom = max(min_zoom, min(max_zoom, zoom))

    return zoom


def parse_convergence_json(json_string):
    """Parse a (possibly double-encoded) procedure-report JSON string.

    Returns (iteration_list, rgap_list). For invalid/missing input returns ([], []).
    """
    if not json_string:
        return [], []

    # If the JSON is stored with escaped characters (double-encoded), unescape it first
    if isinstance(json_string, str) and json_string.startswith('{\\"'):
        json_string = json_string.encode().decode("unicode_escape")

    try:
        data = json.loads(json_string)
    except (TypeError, json.JSONDecodeError, ValueError):
        return [], []

    # double encoded json case
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, ValueError):
            return [], []

    if not isinstance(data, dict):
        return [], []

    convergence = data.get("convergence", {})
    iteration = convergence.get("iteration", [])
    rgap = convergence.get("rgap", [])

    return iteration, rgap


def export_convergence_csv(results_dataframe, data_dir):
    """Write assignment convergence for all results tables into data_dir/assignment_convergence.csv.

    Returns Path to written CSV, or None if there was no convergence data. Raises ValueError
    when iteration/rgap lengths mismatch for a scenario.
    """
    rows = []

    for _, row in results_dataframe.iterrows():
        table_name = row["table_name"]
        procedure_report = row.get("procedure_report")

        iteration, rgap = parse_convergence_json(procedure_report)

        if not iteration or not rgap:
            continue

        if len(iteration) != len(rgap):
            raise ValueError(f"Iteration/RGAP length mismatch for {table_name}")

        for it, rg in zip(iteration, rgap, strict=True):
            rows.append({"iteration": it, "rgap": rg, "series": table_name})

    if not rows:
        return None

    output_path = Path(data_dir) / "assignment_convergence.csv"

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "rgap", "series"])
        writer.writeheader()
        writer.writerows(rows)

    return output_path
