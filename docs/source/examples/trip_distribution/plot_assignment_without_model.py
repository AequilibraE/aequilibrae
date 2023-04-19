"""
.. _plot_assignment_without_model:

Traffic Assignment without an AequilibraE Model
===============================================

In this example, we show how to perform Traffic Assignment in AequilibraE without a model.

We are using Sioux Falls data, from TNPM.
"""
# Imports
import numpy as np

from aequilibrae.distribution import Ipf
from os.path import join
from tempfile import gettempdir
from aequilibrae.matrix import AequilibraeMatrix, AequilibraeData
