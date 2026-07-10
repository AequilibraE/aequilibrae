from aequilibrae.utils.qgis_utils import inside_qgis
from typing import TYPE_CHECKING

if inside_qgis and not TYPE_CHECKING:
    from qgis.PyQt.QtCore import pyqtSignal, QThread

    class WorkerThread(QThread):
        if inside_qgis:
            jobFinished = pyqtSignal(object)

        def __init__(self, parentThread):
            QThread.__init__(self, parentThread)

        def run(self):
            self.running = True
            success = self.doWork()
            if inside_qgis:
                self.jobFinished.emit(success)

        def stop(self):
            self.running = False

else:

    class WorkerThread:
        def __init__(self, *arg):
            pass
