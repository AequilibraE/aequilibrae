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

from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSignal


class WorkerThread(QThread):
    jobFinished = pyqtSignal(object)

    def __init__(self, parentThread):
        super().__init__()
        QThread.__init__(self, parentThread)

    def run(self):
        self.running = True
        success = self.doWork()
        self.jobFinished.emit(success)

    def stop(self):
        self.running = False
        pass

    def doWork(self):
        return True

    def cleanUp(self):
        pass
