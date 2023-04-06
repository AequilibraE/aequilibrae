import sys
from pathlib import Path

project_dir = Path(__file__).parent.parent
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))

from aequilibrae.utils.create_example import create_example

project = create_example("/tmp/test_project")
