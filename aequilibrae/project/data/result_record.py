import pandas as pd
import json

from aequilibrae.project.network.safe_class import SafeClass


class ResultRecord(SafeClass):
    def __init__(self, data_set: dict, project):
        super().__init__(data_set, project)
        self._exists: bool
        self.__dict__["_exists"] = True

    def save(self):
        """Saves results record to the project database"""
        with self.project.db_connection as conn:
            sql = "SELECT COUNT(*) FROM results WHERE table_name=?"

            if conn.execute(sql, [self.table_name]).fetchone()[0] == 0:
                data = [
                    str(self.table_name),
                    str(self.procedure),
                    str(self.procedure_id),
                    json.dumps(self.procedure_report),
                    str(self.timestamp),
                    str(self.description),
                ]
                conn.execute(
                    "INSERT INTO results (table_name, procedure, procedure_id, procedure_report, timestamp, description)"
                    " VALUES(?,?,?,?,?,?)",
                    data,
                )

            for key, value in self.__dict__.items():
                if key != "table_name" and key in self.__original__:
                    v_old = self.__original__.get(key, None)
                    if value != v_old and value:
                        self.__original__[key] = value
                        conn.execute(f"UPDATE results SET '{key}'=? WHERE table_name=?", [value, self.table_name])

    def delete(self):
        """Deletes this results record and the underlying data from disk"""
        with self.project.db_connection as project_conn, self.project.results_connection as results_conn:
            project_conn.execute("DELETE FROM results WHERE table_name=?", [self.table_name])
            results_conn.execute(f"DROP TABLE IF EXISTS {self.table_name}")

        self.__dict__["_exists"] = False

    @property
    def report(self):
        """Retrieves the underlying report and decodes from JSON"""
        return json.loads(self.__dict__["procedure_report"])

    def get_data(self) -> pd.DataFrame:
        """Returns the results for further computation

        Returns:
            **results** (:obj:`pd.DataFrame`): DataFrame object
        """
        with self.project.results_connection as conn:
            return pd.read_sql(f"SELECT * FROM {self.table_name}", conn)

    def __setattr__(self, instance, value) -> None:
        with self.project.db_connection as conn:
            sql = f"Select count(*) from results where LOWER({instance})=?"
            qry_value = sum(conn.execute(sql, [str(value).lower()]).fetchone())
            if qry_value > 0:
                if instance == "table_name":
                    raise ValueError("Another results with this table_name already exists")

        if instance == "report":
            self.__dict__[instance] = json.dumps(value)
        else:
            self.__dict__[instance] = value
