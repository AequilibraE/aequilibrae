import csv
from os.path import join
from typing import List

import pandas as pd

from aequilibrae.transit.transit_elements import Route


def write_routes(routes: List[Route], folder_path: str):
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

    headers = ["route_id", "agency_id", "route_short_name", "route_long_name", "route_desc", "route_type"]
    df = pd.DataFrame(data, columns=headers)
    df.to_csv(join(folder_path, "routes.txt"), quoting=csv.QUOTE_NONNUMERIC, index=False)
