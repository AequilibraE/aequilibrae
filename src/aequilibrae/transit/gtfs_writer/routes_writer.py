import csv
from os.path import join
from typing import List, Union

import pandas as pd

from aequilibrae.transit.transit_elements import Route


def write_routes(routes: Union[List[Route], pd.DataFrame], folder_path: str):
    headers = ["route_id", "agency_id", "route_short_name", "route_long_name", "route_desc", "route_type"]
    if isinstance(routes, pd.DataFrame):
        df = routes.reindex(columns=headers).copy()
    else:
        data = [
            [
                rt.route_id,
                rt.agency_id,
                rt.route_short_name,
                rt.route_long_name,
                str({"description": rt.route_desc, "route": rt.route}),
                rt.route_type,
            ]
            for rt in routes
        ]
        df = pd.DataFrame(data, columns=headers)

    df.to_csv(join(folder_path, "routes.txt"), quoting=csv.QUOTE_NONNUMERIC, index=False)
