def replace_in_file(file_path, text_orig, suffix):
    with open(file_path, "r") as fl:
        cts = [c.rstrip() for c in fl.readlines()]

    with open(file_path, "w") as fl:
        for c in cts:
            if text_orig in c:
                fl.write(f"{c}{suffix}\n")
            else:
                fl.write(f"{c}\n")


replace_in_file("../aequilibrae/paths/parameters.pxi", "MINOR_VRSN", "-qgis")
replace_in_file("../requirements.txt", "numpy", "<1.22")
