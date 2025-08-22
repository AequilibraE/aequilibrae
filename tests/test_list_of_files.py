from pathlib import Path

from aequilibrae.project import about


def test_files_are_listed():
    base_pth = Path(about.__file__).parent / "database_specification"

    for fldr in ["network", "transit"]:
        pth = base_pth / fldr / "tables"

        files_on_disk = sorted(x.stem for x in pth.glob("*.sql"))
        with open(pth / "table_list.txt") as f:
            table_list = sorted(line.strip().lower() for line in f if line.strip())
        assert (
            files_on_disk == table_list
        ), f"SQL files on disk are different from listed as project tables in {fldr}:[{files_on_disk}]"
        assert len(files_on_disk) > 0


def test_trigger_files_are_listed():
    base_pth = Path(about.__file__).parent / "database_specification"
    for fldr in ["network", "transit"]:
        pth = base_pth / fldr / "triggers"

        files_on_disk = sorted(x.stem for x in pth.glob("*.sql"))
        with open(pth / "triggers_list.txt") as f:
            table_list = sorted(line.strip().lower() for line in f if line.strip())
        assert (
            files_on_disk == table_list
        ), f"SQL files on disk are different from listed trigger files in {fldr}:[{files_on_disk}]"
        assert len(files_on_disk) > 0


def test_trigger_names():
    base_pth = Path(about.__file__).parent / "database_specification"

    all_triggers = []
    for fldr in ["network", "transit"]:
        pth = base_pth / fldr / "triggers"

        for f in pth.glob("*.sql"):
            with open(pth / f, "r") as file:
                lines = file.readlines()
            for line in lines:
                line = line.rstrip()
                if line[:2] == "--":
                    continue
                new_line = line.replace("  ", " ")
                while new_line != line:
                    line = new_line
                    new_line = new_line.replace("  ", " ")

                if "CREATE TRIGGER" in new_line.upper():
                    t = new_line.upper()
                    assert "AEQUILIBRAE_" in t
                    for portion in t.split(" "):
                        if "AEQUILIBRAE_" in portion:
                            all_triggers.append(portion)

        repeated = set()
        for trigger in all_triggers:
            if trigger in repeated:
                print(f"TRIGGER REPEATED: {trigger}")
            else:
                repeated.add(trigger)
        assert len(all_triggers) == len(set(all_triggers)), "We have triggers with repeated names"
