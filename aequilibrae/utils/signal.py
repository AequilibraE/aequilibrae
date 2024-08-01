import importlib.util as iutil


def noop(_):
    pass


if iutil.find_spec("qgis") is not None:
    from PyQt5.QtCore import QThread
    from PyQt5.QtCore import pyqtSignal

    class SIGNAL(QThread):
        signal = pyqtSignal(str)

        def emit(self, val):
            self.signal.emit(val)

    noop(SIGNAL.__class__)  # This should be no-op but it stops PyCharm from "optimising" the above import
else:
    from aequilibrae.utils.python_signal import PythonSignal as SIGNAL  # type: ignore

    noop(SIGNAL.__class__)  # This should be no-op but it stops PyCharm from "optimising" the above import
