with open("../aequilibrae/paths/parameters.pxi", "r") as fl:
    cts = [c.rstrip() for c in fl.readlines()]

with open("../aequilibrae/paths/parameters.pxi", "w") as fl:
    for c in cts:
        if "MINOR_VRSN" in c:
            fl.write(f"{c}-qgis\n")
        else:
            fl.write(f"{c}\n")
