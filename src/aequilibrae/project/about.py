from aequilibrae.project.project_table import NonSpatialProjectTable


class About(NonSpatialProjectTable):
    """Project-table gateway for the project's key/value ``about`` metadata."""

    name = "about"
    key = "infoname"
    record_name = "AboutRecord"
