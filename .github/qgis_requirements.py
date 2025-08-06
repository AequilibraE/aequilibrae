import datetime


def replace_in_file(file_path, text_orig, suffix):
    with open(file_path, "r") as fl:
        cts = [c.rstrip() for c in fl.readlines()]

    with open(file_path, "w") as fl:
        for c in cts:
            if text_orig in c and suffix not in c:
                c = c.replace(text_orig, f"{text_orig}{suffix}")
            fl.write(f"{c}\n")


def replace_version(file_path, suffix):
    today = datetime.date.today()
    start_of_year = datetime.date(today.year, 1, 1)
    days_elapsed = (today - start_of_year).days + 1  # +1 to include today

    with open(file_path, "r") as fl:
        cts = [c.rstrip() for c in fl.readlines()]

    with open(file_path, "w") as fl:
        for c in cts:
            if "version =" in c.lower():
                q = c.split(".")
                q[-1] = q[-1].replace('"', "").replace("'", "")
                q.append(f'dev{days_elapsed}"')
                c = ".".join(q)
                print(q, c)
            fl.write(f"{c}\n")


replace_in_file("pyproject.toml", "numpy", "<1.99")
replace_version("./aequilibrae/__init__.py", ".dev")
