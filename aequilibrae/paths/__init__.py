"""
path computation related code
"""

__author__ = "Pedro Camargo ($Author: Pedro Camargo $)"
__version__ = "0.4.0"
__revision__ = "$Revision: 2 $"
__date__ = "$Date: 2017-02-25$"

import warnings

try:
    from .AoN import one_to_all, skimming_single_origin, path_computation, VERSION_COMPILED, update_path_trace
    from .results import *
    from .multi_threaded_aon import MultiThreadedAoN
    from .multi_threaded_skimming import MultiThreadedNetworkSkimming
    from .network_skimming import NetworkSkimming
    from .all_or_nothing import allOrNothing
except ImportError as e:
    warnings.warn("The AoN extension has not been compiled. {}".format(e.name))
from .graph import Graph
from .__version__ import binary_version, release_name, minor_version, release_version
