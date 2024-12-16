import os
import re
import mmap
import sys
from pathlib import Path

project_dir = Path(__file__).parent.parent
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))


def replace_regex(event):
    for root, directories, files in os.walk("docs/source"):
        dirs = ["_auto_examples", "_generated", "_latex", "_static", "images"]
        directories[:] = [d for d in directories if d not in dirs]
        for file in files:
            if file.endswith((".drawio", ".txt", ".omx", ".png")):
                continue
            full_path = os.path.join(root, file)
            with open(full_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as s:
                content = s.read().decode("utf-8")

                in_path = r"www.aequilibrae.com/latest/" if file == "conf.py" else r"www.aequilibrae.com/latest/python/"
                out_path = (
                    rf"www.aequilibrae.com/{event}/" if file == "conf.py" else rf"www.aequilibrae.com/{event}/python/"
                )

                # Find and replace regex containing the website
                modified_content = re.sub(in_path, out_path, content)

                events = event.split("/")
                modified_content = re.sub(
                    r"www.aequilibrae.com/latest/qgis/",
                    rf"www.aequilibrae.com/{events[0]}/qgis/",
                    modified_content,
                )

                # Write modified content back to file
                with open(full_path, "w", encoding="utf-8") as w:
                    w.write(modified_content)
