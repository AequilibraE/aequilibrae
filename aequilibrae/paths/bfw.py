"""
-----------------------------------------------------------------------------------------------------------
 Package:    AequilibraE

 Name:       Traffic assignment
 Purpose:    Implement BFW traffic assignment based on the AoN class

 Original Author:  Jan Zill, Pedro Camargo (c@margo.co)
 Contributors:
 Last edited by: Pedro Camargo

 Website:    www.AequilibraE.com
 Repository:  https://github.com/AequilibraE/AequilibraE

 Created:    2020-02-01
 Updated:    2020-02-01
 Copyright:   (c) AequilibraE authors
 Licence:     See LICENSE.TXT
-----------------------------------------------------------------------------------------------------------
 """
from typing import List
import importlib.util as iutil
import threading
from multiprocessing.dummy import Pool as ThreadPool
from .traffic_class import TrafficClass
from ..utils import WorkerThread
import numpy as np

from .all_or_nothing import allOrNothing

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None
if pyqt:
    from PyQt5.QtCore import pyqtSignal as SIGNAL


class bfw(WorkerThread):
    if pyqt:
        assignment = SIGNAL(object)

    def __init__(self, traffic_classes: List[TrafficClass]):
        WorkerThread.__init__(self, None)

        # To get the result of all the slices assigned to the class
        self.traffic_classes = traffic_classes

        self.report = []
        self.cumulative = 0

    def doWork(self):
        self.execute()

    def execute(self):
        if pyqt:
            self.assignment.emit(["zones finalized", 0])
