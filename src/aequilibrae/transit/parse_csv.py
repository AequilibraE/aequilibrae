import copy
import csv
from io import TextIOWrapper

import numpy as np
from numpy.lib.recfunctions import append_fields

from aequilibrae.transit.column_order import Nullable


def _cast(value: str, dtype: type):
    # int data may be given as a float string
    return int(value.split(".")[0]) if dtype is int else dtype(value)


def parse_csv(file_name: str, column_order={}):  # noqa B006
    tot: list[list[str]] = []
    if isinstance(file_name, str):
        csvfile = open(file_name, encoding="utf-8-sig")
    else:
        csvfile = TextIOWrapper(file_name, encoding="utf-8-sig")

    contents = csv.reader(csvfile, delimiter=",", quotechar='"')
    numcols = 0
    for row in contents:
        if not len("".join(row).strip()):
            continue
        broken: list[str] = [x.encode("ascii", errors="ignore").decode().strip() for x in row]

        if not numcols:
            numcols = len(broken)
        else:
            # Rows may omit their trailing empty fields
            if len(broken) < numcols:
                broken.extend([""] * (numcols - len(broken)))

        tot.append(broken)
    titles: list[str] = tot.pop(0)
    csvfile.close()
    if tot:
        lowercase_titles = [title.lower() for title in titles]
        data = np.rec.fromrecords(tot, names=lowercase_titles)
    else:
        return empty()

    missing_cols_names = [x for x in column_order.keys() if x not in data.dtype.names]
    for col in missing_cols_names:
        data = append_fields(data, col, np.array([""] * len(tot)), usemask=False)

    if column_order:
        col_names = [x for x in column_order.keys() if x in data.dtype.names]
        data = data[col_names]

        # Define sizes for the string variables
        column_order = copy.deepcopy(column_order)
        nullable = {}
        for c in col_names:
            if column_order[c] is str:
                column_order[c] = object
            elif isinstance(column_order[c], Nullable):
                # Cast after the conversion below, so empty fields become None rather than zero
                nullable[c] = column_order[c].dtype
                column_order[c] = object
            else:
                if data[c].dtype.char.upper() in ["U", "S"]:
                    data[c] = np.where(data[c] == "", "0", data[c])

        new_data_dt = [(f, column_order[f]) for f in col_names]

        if int(data.shape.__len__()) > 0:
            # handle the case of int data given as a float string
            for j, (_, dtype) in enumerate(new_data_dt):
                if dtype is int:
                    for item in data:
                        item[j] = item[j].split(".")[0]

            data = np.array(data, dtype=new_data_dt)
            for c, dtype in nullable.items():
                data[c] = [None if x == "" else _cast(x, dtype) for x in data[c]]
            return data
        else:
            return data
    else:
        return data


class empty:
    shape = [0]
