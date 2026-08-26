from typing import Any

from aequilibrae.project.project_table import _SELECT_ONE_SQL, NonSpatialProjectTable


class Modes(NonSpatialProjectTable):
    """
    Object to manipulate the modes table in the database.

    .. code-block:: python

        >>> project = create_example(project_path)

        >>> modes = project.network.modes

        # We can get a mode as an immutable record
        >>> car_mode = modes.get('c')

        # or by name
        >>> car_mode = modes.get_by_name('car')

        # and write changes explicitly
        >>> modes.update('c', description='personal autos only', alpha=0.95)

        # Adding a new mode to the model is an insert
        >>> modes.insert(mode_id='k', mode_name='flying_car')
        'k'

        >>> project.close()
    """

    name = "modes"
    key = "mode_id"
    record_name = "ModeRecord"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._select_by_name_sql = _SELECT_ONE_SQL.format(table=self.name, key="mode_name", columns="*")

    def get_by_name(self, mode_name: str) -> Any:
        """Get a mode record by its descriptive name.

        :Arguments:
            **mode_name** (:obj:`str`): Descriptive mode name.

        :Returns:
            **mode** (:obj:`Any`): Generated frozen record for the mode.
        """
        self._refresh_record_type()
        row = self._connection._connection.execute(self._select_by_name_sql, [mode_name]).fetchone()
        if row is None:
            raise ValueError(f"Mode {mode_name} does not exist in the model")
        return self._build_record(row)
