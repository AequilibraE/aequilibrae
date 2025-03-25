import logging
import warnings
from copy import deepcopy
from math import ceil
from os import PathLike
from typing import List

import numpy as np
import pandas as pd
from shapely.geometry.linestring import LineString
from shapely.ops import linemerge
from shapely.ops import substring

from aequilibrae.utils.aeq_signal import SIGNAL
from aequilibrae.utils.interface.worker_thread import WorkerThread


class NetworkSimplifier(WorkerThread):
    signal = SIGNAL(object)

    def __init__(self, project) -> None:

        self.network = project.network
        self.links = self.network.links

        warnings.warn("This will alter your database in place. Make sure you have a backup.")

    def simplify(self, field_name):
        """
        Simplifies the network by merging links that are shorter than a given threshold

        Args:
            *maximum_allowable_link_length* (:obj:`float`): Maximum length for output links (meters)

            *max_speed_ratio* (:obj:`float`): Maximum ratio between the fastest and slowest speed for a link to be considered for simplification
        """
        pass

    def __process_link_fields(self, candidates, link_sequence, max_speed_ratio):
        pass

    def __execute_link_deletion_and_addition(self, new_links, links_to_delete):
        pass

    def collapse_links_into_nodes(self, links: List[int]):
        pass

    def rebuild_network(self):
        """Rebuilds the network elements that would have to be rebuilt after massive network simplification"""
        pass
