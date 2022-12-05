import csv
from os.path import join
from typing import List

import pandas as pd

from polarislib.network.transit.transit_elements import Agency


def write_agencies(agencies: List[Agency], folder_path: str, timezone):
    headers = ["agency_id", "agency_name", "agency_url", "agency_timezone"]

    data = [[ag.agency_id, ag.agency, "https://vms.taps.anl.gov/tools/polaris/", timezone] for ag in agencies]
    df = pd.DataFrame(data, columns=headers)
    df.to_csv(join(folder_path, "agency.txt"), quoting=csv.QUOTE_NONNUMERIC, index=False)
