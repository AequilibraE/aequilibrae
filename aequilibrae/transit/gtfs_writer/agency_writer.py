import csv
from os.path import join
from typing import List, Union

import pandas as pd

from aequilibrae.transit.transit_elements import Agency


def write_agencies(agencies: Union[List[Agency], pd.DataFrame], folder_path: str):
    headers = ["agency_id", "agency_name", "agency_url", "agency_timezone"]

    if isinstance(agencies, pd.DataFrame):
        df = agencies.reindex(columns=headers).copy()
    else:
        data = [[ag.agency_id, ag.agency, "https://vms.taps.anl.gov/tools/polaris/", pd.NA] for ag in agencies]
        df = pd.DataFrame(data, columns=headers)

    df.to_csv(join(folder_path, "agency.txt"), quoting=csv.QUOTE_NONNUMERIC, index=False)
