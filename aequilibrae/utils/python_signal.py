import importlib.util as iutil
import os
import warnings
from random import choice

from aequilibrae.utils.qgis_utils import inside_qgis

missing_tqdm = iutil.find_spec("tqdm") is None

if not missing_tqdm:
    notebook = iutil.find_spec("ipywidgets") is not None
    if notebook:
        from tqdm.notebook import tqdm  # type: ignore
    else:
        from tqdm import tqdm  # type: ignore

show_status = os.environ.get("AEQ_SHOW_PROGRESS", "TRUE") == "TRUE"


class PythonSignal:  # type: ignore
    """
    This class provides a pure python equivalent of the Signal passing infrastructure present in QGIS.
    It takes updates in the same format as the QGIS progress bar manager used in QAequilibrae and translates
    them into TQDM syntax.

    The expected structure of update data is a list where the first element is string describing the desired action:

        ['action', *args]

    The currently supported formats for actions are listed here:

        1. ['finished']                            - close out the current progress bar
        2. ['refresh']                             - force the current progress bar to refresh
        3. ['reset']                               - reset the current progress bar
        4. ['start', num_elements: int, desc: str] - start a new progress bar
        5. ['set_position', pos: int]              - set the position of the current progress bar
        6. ['set_text',  desc: str]                - set the description of the current progress bar
        7. ['update', pos: int, desc: str]         - set both pos and desc of current progress bar

    """

    def __init__(self, object):
        self.color = choice(["green", "magenta", "cyan", "blue", "red", "yellow"])
        self.pbar = None  # type: tqdm
        self.keydata = {}
        self.position = 0
        self.deactivate = not show_status  # by default don't use progress bars in tests

    def emit(self, val):
        if self.deactivate:
            return

        action = val[0]

        # handle actions which just send a signal onto the progress bar
        if action in ["finished", "refresh", "reset"]:
            if self.pbar is not None:
                method = {"finished": "close", "refresh": "refresh", "reset": "reset"}[action]
                getattr(self.pbar, method)()

        elif action == "set_position":
            self.position = val[1]

        elif action == "set_text":
            desc = str(val[1]).ljust(50)
            if self.pbar is not None and self.pbar.desc != desc:
                self.pbar.set_description(desc, refresh=True)

        elif action == "start":
            if missing_tqdm and not inside_qgis:
                self.deactivate = True
                warnings.warn("No progress bars will be shown. Please install tqdm to see them")
                return

            # Close any existing bars
            if self.pbar is not None:
                self.pbar.close()

            # Create a new bar with the given capacity
            desc = str(val[2]).ljust(50)
            self.pbar = tqdm(
                total=val[1], colour=self.color, leave=False, desc=desc, position=self.position, mininterval=0.1
            )

        elif action == "update":
            self.pbar.update(val[1] - self.pbar.n)
            if len(val) > 2:
                desc = str(val[2]).ljust(50)
                if self.pbar.desc != desc:
                    self.pbar.set_description(desc, refresh=True)
