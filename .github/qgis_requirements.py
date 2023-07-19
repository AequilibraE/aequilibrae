def replace_in_file(file_path, text_orig, suffix):
    with open(file_path, "r") as fl:
        cts = [c.rstrip() for c in fl.readlines()]

    with open(file_path, "w") as fl:
        for c in cts:
            if text_orig in c:
                c = c.replace(text_orig, f"{text_orig}{suffix}")
            fl.write(f"{c}\n")


replace_in_file("../requirements.txt", "numpy", "<1.22")
replace_in_file("../pyproject.toml", "numpy", "<1.22")

replace_in_file("../requirements.txt", "pandas", "<1.2")
replace_in_file("../pyproject.toml", "pandas", "<1.2")

replace_in_file("../requirements.toml", "scipy", "<1.11")
replace_in_file("../pyproject.toml", "scipy", "<1.11")

replace_in_file("../__version__.py", "{minor_version}", ".dev0")
