"""
Implementation of statistics class to help record and analyse the ODME procedure:
"""

import time
import numpy as np
import pandas as pd

class ODMEStats(object):
    """ Statistics of an ODME procedure """

    def __init__(self, odme) -> None:
        """
        """
        self.odme = odme
        return