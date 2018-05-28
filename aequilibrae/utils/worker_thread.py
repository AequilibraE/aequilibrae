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
 Updated:    30/09/2016
 Copyright:   (c) AequilibraE authors
 Licence:     See LICENSE.TXT
 -----------------------------------------------------------------------------------------------------------
 """

try:
    from PyQt4.QtCore import QThread, SIGNAL
except:
    from PyQt5.QtCore import QThread
    from PyQt5.QtCore import pyqtSignal as SIGNAL


class WorkerThread(QThread):
    def __init__(self, parentThread):
        QThread.__init__(self, parentThread)

    def run(self):
        self.running = True
        success = self.doWork()
        self.emit(SIGNAL("jobFinished(PyQt_PyObject)"), success)

    def stop(self):
        self.running = False
        pass

    def doWork(self):
        return True

    def cleanUp(self):
        pass
