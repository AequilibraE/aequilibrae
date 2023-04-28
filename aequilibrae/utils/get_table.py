import pandas as pd


def get_table(table_name, conn):
    """
    Selects table from database.

    :Arguments:
         **table_name** (:obj:`str`): desired table name
         **conn** (:obj:`sqlite3.Connection`): database connection
    """

    return pd.read_sql(f"SELECT * FROM {table_name};", con=conn)
