"""
 Original Author:  UNKNOWN. COPIED FROM STACKOVERFLOW BUT CAN'T REMEMBER EXACTLY WHERE
 """
import importlib.util as iutil

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None
if pyqt:
    from PyQt5.QtCore import QThread
    from PyQt5.QtCore import pyqtSignal
else:

    class QThread:
        def __init__(self, *arg):
            pass


class WorkerThread(QThread):
    if pyqt:
        jobFinished = pyqtSignal(object)

    def __init__(self, parentThread):
        QThread.__init__(self, parentThread)

    def run(self):
        self.running = True
        success = self.doWork()
        if pyqt:
            self.jobFinished.emit(success)

    def stop(self):
        self.running = False
        pass

    def doWork(self):
        return True

    def cleanUp(self):
        pass
