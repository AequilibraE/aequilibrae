import numpy as np
import csv


def parse_csv(file_name: str, column_order=[]):
    tot = []
    with open(file_name, encoding='utf-8-sig') as csvfile:
        contents = csv.reader(csvfile, delimiter=',', quotechar='\"')
        for row in contents:
            tot.append(row)
    titles = tot.pop(0)
    data = np.core.records.fromrecords(tot, names=titles)

    if column_order:
        col_names = [x for x in column_order.keys() if x in data.dtype.names]
        data = data[col_names]

        # Define sizes for the string variables
        column_order = copy.deepcopy(column_order)
        for c in col_names:
            if column_order[c] is str:
                if data[c].dtype.char.upper() == "S":
                    column_order[c] = data[c].dtype
                else:
                    column_order[c] = "S16"

        new_data_dt = [(f, column_order[f]) for f in col_names]

        if int(data.shape.__len__()) > 0:
            return np.array(data, new_data_dt)
        else:
            return data
    else:
        return data
