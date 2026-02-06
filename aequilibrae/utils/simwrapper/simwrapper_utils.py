import math

def get_links_bounds_box(project):
    """
    Compute box around all coordinates in links table of project.
    Queries spatial database to find max and min x and y coords across all link geomerties
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
        ]  # if cant find coordinates bc of missing link vals, will make this better though but works for now

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
        return (
            10  # if cant find coordinates bc of missing link vals, will make this better though but works for now
        )

    xmin, ymin, xmax, ymax = row

    x_span = abs(xmax - xmin)
    y_span = abs(ymax - ymin)

    max_span = max(x_span, y_span)  # use larger of two so we see everything

    if max_span <= 0:
        return 10  # if invalid values, clearly not a negative distance we want

    # calculate ~ zoom:
    # at zoom of 0 the world is ~360degrees wide
    # each increment doubles the resolution
    zoom = int(round(math.log2(360 / max_span)))

    # fix this within the allowed range
    zoom = max(min_zoom, min(max_zoom, zoom))

    return zoom
