"""
 -----------------------------------------------------------------------------------------------------------
 Package:    AequilibraE

 Name:       Worker thread
 Purpose:    Implements worker thread

 Original Author:  UNKNOWN. COPIED FROM STACKOVERFLOW BUT CAN'T REMEMBER EXACTLY WHERE
 Contributors:
 Last edited by: Pedro Camargo

 Website:    www.AequilibraE.com
 Repository:  https://github.com/AequilibraE/AequilibraE

 Created:    2014-03-19
 Updated:    2018-08-20
 Copyright:   (c) AequilibraE authors
 Licence:     See LICENSE.TXT
 -----------------------------------------------------------------------------------------------------------
 """
import importlib.util as iutil

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None
if pyqt:
    from PyQt5.QtCore import QThread
    from PyQt5.QtCore import pyqtSignal
else:
    class QThread:
        pass


class WorkerThread(QThread):
    if pyqt:
        jobFinished = pyqtSignal(object)

    def __init__(self, parentThread):
        super().__init__()
        if pyqt:
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
