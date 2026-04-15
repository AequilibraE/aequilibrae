from os import PathLike
from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd

from aequilibrae.utils.db_utils import read_and_close
from aequilibrae.utils.get_table import get_geo_table


class DataLoader:
    def __init__(self, path_to_file: PathLike, table_name: str):
        self.__pth_file = Path(path_to_file)
        self.table_name = table_name

    def load_table(self) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
        with read_and_close(self.__pth_file, spatial=True) as conn:
            return get_geo_table(self.table_name, conn)
