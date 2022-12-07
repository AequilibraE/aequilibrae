import pandas as pd


def get_table(table_name, conn):

    return pd.read_sql(f"SELECT * FROM {table_name};", con=conn)
