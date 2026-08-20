import logging
from typing import Any

from aequilibrae.project.project_table import NonSpatialProjectTable

logger = logging.getLogger(__name__)


class Periods(NonSpatialProjectTable):
    """Access to the API resources to manipulate the periods table in the network

    .. code-block:: python

        >>> project = create_example(project_path, "coquimbo")

        >>> periods = project.network.periods

        # We can get a single period as an immutable record
        >>> period = periods.get(1)

        # Or create a new one
        >>> periods.new_period(2, 21600, 32400, "morning peak")
        2

        >>> project.close()
    """

    name = "periods"
    key = "period_id"
    record_name = "PeriodRecord"

    def new_period(self, period_id: int, start: int, end: int, description: str | None = None) -> int:
        """Creates a new period with a given ID

        :Arguments:
            **period_id** (:obj:`int`): Id of the period to be created

            **start** (:obj:`int`): Start time of the period to be created

            **end** (:obj:`int`): End time of the period to be created

            **description** (:obj:`str`): Optional human readable description of the time period e.g. '1pm - 5pm'
        """
        return self.insert(
            period_id=period_id,
            period_start=start,
            period_end=end,
            period_description=description if description is not None else "",
        )

    def renumber(self, period_id: int, new_id: int) -> None:
        """Renumbers a period in the network

        :Arguments:
            **period_id** (:obj:`int`): Current period_id

            **new_id** (:obj:`int`): New period_id
        """
        new_id = int(new_id)

        if new_id == 1 or period_id == 1:
            raise ValueError("You cannot renumber, or renumber another period to the default period.")

        if new_id == period_id:
            logger.warning("This is already the period number")
            return

        self._change_key(period_id, new_id)
        logger.info(f"Period {period_id} was renumbered to {new_id}")

    @property
    def default_period(self) -> Any:
        """The default period (period 1), which always exists"""
        return self.get(1)
