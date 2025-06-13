from copy import deepcopy

import pandas as pd

from aequilibrae.project.basic_table import BasicTable
from aequilibrae.project.data_loader import DataLoader
from aequilibrae.project.network.period import Period
from aequilibrae.project.table_loader import TableLoader


class Periods(BasicTable):
    """Access to the API resources to manipulate the periods table in the network

    .. code-block:: python

        >>> project = create_example(project_path, "coquimbo")

        >>> all_periods = project.network.periods

        # We can just get one link in specific
        >>> period = all_periods.get(1)

        # We can save changes for all periods we have edited so far
        >>> all_periods.save()
    """

    #: Query sql for retrieving periods
    sql = ""

    def __init__(self, net):
        super().__init__(net.project)
        self.__table_type__ = "periods"
        self.__items = {}
        self.__fields = []

        if self.sql == "":
            self.refresh_fields()

    def extent(self):
        # FIXME: This is not real subclassing, the extent function needs to be moved out of BasicTable
        # This hack will do for now
        raise NotImplementedError("Not applicable to Periods class")

    def get(self, period_id: int) -> Period:
        """Get a period from the network by its **period_id**

        It raises an error if period_id does not exist

        :Arguments:
            **period_id** (:obj:`int`): Id of a period to retrieve

        :Returns:
            **period** (:obj:`Period`): Period object for requested period_id
        """

        if period_id in self.__items:
            period = self.__items[period_id]

            # If this element has not been renumbered, we return it. Otherwise we
            # store the object under its new number and carry on
            if period.period_id == period_id:
                return period
            else:
                self.__items[period.period_id] = self.__items.pop(period_id)

        with self.project.db_connection as conn:
            data = conn.execute(f"{self.sql} where period_id=?", [period_id]).fetchone()
        if data:
            data = dict(zip(self.__fields, data))
            period = Period(data, self.project)
            self.__items[period.period_id] = period
            return period

        raise ValueError(f"Period {period_id} does not exist in the model")

    def refresh_fields(self) -> None:
        """After adding a field one needs to refresh all the fields recognized by the software"""
        tl = TableLoader()
        with self.project.db_connection as conn:
            tl.load_structure(conn, "periods")
        self.sql = tl.sql
        self.__fields = deepcopy(tl.fields)

    def refresh(self):
        """Refreshes all the periods in memory"""
        lst = list(self.__items.keys())
        for k in lst:
            del self.__items[k]

    def new_period(self, period_id: int, start: int, end: int, description: str = None) -> Period:
        """Creates a new period with a given ID

        :Arguments:
            **period_id** (:obj:`int`): Id of the centroid to be created

            **start** (:obj:`int`): Start time of the period to be created

            **end** (:obj:`int`): End time of the period to be created

            **description** (:obj:`str`): Optional human readable description of the time period e.g. '1pm - 5pm'
        """

        with self.project.db_connection as conn:
            dt = conn.execute("SELECT COUNT(*) FROM periods WHERE period_id=?", [period_id]).fetchone()[0]
        if dt > 0:
            raise Exception("period_id already exists. Failed to create it")

        data = {key: None for key in self.__fields}
        data["period_id"] = period_id
        data["period_start"] = start
        data["period_end"] = end
        data["period_description"] = description if description is not None else ""
        period = Period(data, self.project)
        self.__items[period_id] = period
        return period

    def save(self):
        for item in self.__items.values():
            item.save()

    @property
    def data(self) -> pd.DataFrame:
        """Returns all periods data as a Pandas DataFrame

        :Returns:
            **table** (:obj:`DataFrame`): Pandas DataFrame with all the periods
        """
        dl = DataLoader(self.project.path_to_file, "periods")
        return dl.load_table()

    def __del__(self):
        self.__items.clear()

    @property
    def default_period(self) -> Period:
        return self.get(1)
