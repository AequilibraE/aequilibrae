from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aequilibrae.project.project import Project


# FIXME: project dependency should be narrowed to its required domain owner.
def clean(project: "Project") -> None:
    """Remove non-centroid nodes which are no longer linked."""
    # SQLite does not define trigger execution order, so remove extraneous
    # non-centroid nodes at project lifecycle boundaries.
    with project.db_connection as connection:
        connection.execute(
            """DELETE FROM nodes WHERE is_centroid=0 AND
            (SELECT count(*) FROM links WHERE a_node=node_id OR b_node=node_id)=0"""
        )
