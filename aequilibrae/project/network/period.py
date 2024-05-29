from .safe_class import SafeClass
from ...utils.db_utils import commit_and_close
from ...utils.spatialite_utils import connect_spatialite


class Period(SafeClass):
    """A Period object represents a single record in the *periods* table

    .. code-block:: python

        >>> from aequilibrae import Project

        >>> proj = Project.from_path("/tmp/test_project")

        >>> all_periods = proj.network.periods

        # We can just get one link in specific
        >>> period1 = all_periods.get(1)

        # We can find out which fields exist for the period
        >>> which_fields_do_we_have = period1.data_fields()

        # It succeeds if the period_id already does not exist
        >>> period1.renumber(998877)

        # We can just save the period
        >>> period1.save()
    """

    def __init__(self, dataset, project):
        """"""
        super().__init__(dataset, project)
        self.__fields = list(dataset.keys())
        self._table = "periods"

    def save(self):
        """Saves period to database"""

        with commit_and_close(connect_spatialite(self.project.path_to_file)) as conn:
            if self.period_id != self.__original__["period_id"]:
                raise ValueError("One cannot change the period_id")

            data, sql = self.__save_period()

            if data:
                conn.execute(sql, data)

    def data_fields(self) -> list:
        """Lists all data fields for the period, as available in the database

        :Returns:
            **data fields** (:obj:`list`): list of all fields available for editing
        """

        return list(self.__original__.keys())

    def renumber(self, new_id: int):
        """Renumbers the period in the network

        Logs a warning if another period already exists with this period_id

        :Arguments:
            **new_id** (:obj:`int`): New period_id
        """

        new_id = int(new_id)

        if new_id == 1 or self.period_id == 1:
            raise ValueError("You cannot renumber, or renumber another period to the default period.")

        if new_id == self.period_id:
            self._logger.warning("This is already the period number")
            return

        with commit_and_close(connect_spatialite(self.project.path_to_file)) as conn:
            try:
                conn.execute("Update periods set period_id=? where period_id=?", [new_id, self.period_id])
            finally:
                conn.commit()
        self._logger.info(f"Period {self.period_id} was renumbered to {new_id}")
        self.__dict__["period_id"] = new_id
        self.__original__["period_id"] = new_id

    def __save_period(self):
        data = []
        txts = []
        for key, val in self.__dict__.items():
            if key not in self.__original__:
                continue
            data.append(val)
            txts.append(f'"{key}"')

        if not data:
            self._logger.warning(f"Nothing to update for period {self.period_id}")
            return [], ""

        values = ",".join("?" * len(txts))
        txts = ",".join(txts)
        sql = f"INSERT OR REPLACE INTO periods ({txts}) VALUES ({values})"
        return data, sql

    def __setattr__(self, instance, value) -> None:
        if instance not in self.__dict__ and instance[:1] != "_":
            raise AttributeError(f'"{instance}" is not a valid attribute for a period')
        elif instance == "period_id":
            raise AttributeError("Setting period_id is not allowed")
        self.__dict__[instance] = value
