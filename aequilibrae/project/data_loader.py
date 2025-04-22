from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd

from aequilibrae.utils.db_utils import commit_and_close
from aequilibrae.utils.get_table import get_geo_table


class DataLoader:
    def __init__(self, path_to_file: Union[Path, str], table_name: str):
        self.__pth_file = path_to_file
        self.table_name = table_name

    def load_table(self) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
        with commit_and_close(self.__pth_file, spatial=True) as conn:
            return get_geo_table(self.table_name, conn)
