from aequilibrae.utils.qgis_utils import inside_qgis


def noop(_):
    pass


if inside_qgis:
    from PyQt5.QtCore import pyqtSignal as SIGNAL  # type: ignore

    noop(SIGNAL.__class__)  # This should be no-op but it stops PyCharm from "optimising" the above import
else:
    from aequilibrae.utils.python_signal import PythonSignal as SIGNAL  # type: ignore

    noop(SIGNAL.__class__)  # This should be no-op but it stops PyCharm from "optimising" the above import
