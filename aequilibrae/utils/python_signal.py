import importlib.util as iutil
import os
import warnings
from random import choice

missing_tqdm = iutil.find_spec("tqdm") is None

if not missing_tqdm:
    notebook = iutil.find_spec("ipywidgets") is not None
    if notebook:
        from tqdm.notebook import tqdm  # type: ignore
    else:
        from tqdm import tqdm  # type: ignore

qgis = iutil.find_spec("qgis") is not None

show_status = os.environ.get("AEQ_SHOW_PROGRESS", "FALSE") == "TRUE"


class PythonSignal:  # type: ignore
    """
    This class only manages where the updating information will flow to, either emitting signals
    to the QGIS interface to update is progress bars or to update the terminal progress bars
    powered by tqdm

    Structure of data is the following:

    ['action', 'bar hierarchy', 'value', 'text', 'master']

    'action': 'start', 'update', or 'finished_*_processing' (the last one applies in QGIS)
    'position': Position (0 for top, 1 for bottom)
    'value': Numerical value for the action (total or current)
    'text': Whatever label to be updated
    'master': The corresponding master bar for this task
    """

    deactivate = not show_status  # by default don't use progress bars in tests

    def __init__(self, object):
        self.color = choice(["green", "magenta", "cyan", "blue", "red", "yellow"])
        self.pbar = None  # type: tqdm
        self.keydata = {}
        self.position = 0

    def emit(self, val):
        if self.deactivate:
            return
        if val[0] == "set_position":
            self.position = val[1]

        if val[0] == "finished":
            if self.pbar is not None:
                self.pbar.close()

        elif val[0] == "refresh":
            if self.pbar is not None:
                self.pbar.refresh()

        elif val[0] == "reset":
            if self.pbar is not None:
                self.pbar.reset()

        elif val[0] == "key_value":
            self.keydata[val[1]] = val[2]

        elif val[0] == "start":
            if missing_tqdm and not qgis:
                self.deactivate = True
                warnings.warn("No progress bars will be shown. Please install tqdm to see them")
            if self.pbar is not None:
                self.pbar.close()
            desc = str(val[2]).rjust(50)
            self.pbar = tqdm(total=val[1], colour=self.color, leave=False, desc=desc, position=self.position)

        elif val[0] == "update":
            self.pbar.update(val[1] - self.pbar.n)
            if len(val) > 2:
                desc = str(val[2]).rjust(50)
                if self.pbar.desc != desc:
                    self.pbar.set_description(desc, refresh=True)

        elif val[0] == "set_text":
            desc = str(val[1]).rjust(50)
            if self.pbar.desc != desc:
                self.pbar.set_description(desc, refresh=True)
