Choice set generation algorithms
================================

XXXXXXXXXXXXXX

Imports
-------

.. code:: python

   import matplotlib.pyplot as plt
   import networkx as nx
   import numpy as np
   import pandas as pd
   from aequilibrae.paths.public_transport import HyperpathGenerating
   from numba import jit

   RS = 124  # random seed
   FS = (6, 6)  # figure size



References
----------

.. [1] Rieser-Schüssler, N., Balmer, M., & Axhausen, K. W. (2012). Route choice sets for very high-resolution data.
       Transportmetrica A: Transport Science, 9(9), 825–845.
       https://doi.org/10.1080/18128602.2012.671383

.. [2] Moss, J., P. V. de Camargo, C. de Freitas, and R. Imai. High-Performance Route Choice Set Generation on
       Large Networks. Presented at the ATRF, Melbourne, 2024.
