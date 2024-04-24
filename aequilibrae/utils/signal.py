import importlib.util as iutil


def noop(_):
    pass


if iutil.find_spec("PyQt5") is not None:
    from PyQt5.QtCore import pyqtSignal as SIGNAL  # type: ignore

    noop(SIGNAL.__class__)  # This should be no-op but it stops PyCharm from "optimising" the above import
else:
    from aequilibrae.utils.python_signal import PythonSignal as SIGNAL  # type: ignore

    noop(SIGNAL.__class__)  # This should be no-op but it stops PyCharm from "optimising" the above import
