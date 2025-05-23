from aequilibrae.utils.qgis_utils import inside_qgis


def noop(_):
    pass


if inside_qgis:
    from qgis.PyQt.QtCore import pyqtSignal as SIGNAL  # type: ignore

    noop(SIGNAL.__class__)  # This should be no-op but it stops PyCharm from "optimising" the above import
else:
    from aequilibrae.utils.python_signal import PythonSignal as SIGNAL  # type: ignore

    noop(SIGNAL.__class__)  # This should be no-op but it stops PyCharm from "optimising" the above import


class simple_progress(object):
    """A `tqdm` style iterable wrapper using aequilibrae signals"""

    def __init__(self, thing, signal, msg=None):
        self.thing = thing
        self.iterator = None

        try:
            num_elements = len(self.thing)
        except (TypeError, AttributeError):
            num_elements = 0

        self.msg = msg or f"{{}}/{num_elements}"
        self.signal = signal
        self.signal.emit(["start", num_elements, self.msg.format(0)])
        self.counter = 1

    def __iter__(self):
        self.iterator = iter(self.thing)
        return self

    def __next__(self):
        current = next(self.iterator)
        self.signal.emit(["update", self.counter, self.msg.format(self.counter)])
        self.counter = self.counter + 1
        return current
