"""
Implementation of results/statistics class to help record and analyse the ODME procedure:
"""

import time
import numpy as np
import pandas as pd

class ODMEResults(object):
    """ Results and statistics of an ODME procedure """

    def __init__(self, odme) -> None:
        """
        """
        self.odme = odme
        return