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
            if file.endswith((".drawio", ".txt", ".omx")):
                continue
            full_path = os.path.join(root, file)
            print(full_path)
            with open(full_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as s:
                content = s.read().decode("utf-8")

                # Find and replace regex containing the site
                modified_content = re.sub(r"www.aequilibrae.com/latest/", rf"www.aequilibrae.com/{event}/", content)

                # Write modified content back to file
                with open(full_path, "w", encoding="utf-8") as w:
                    w.write(modified_content)
