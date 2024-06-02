.. _route_choice:

Route Choice
============

AequilibraE is the first modeling package with full support for route choice, from the creation of choice sets through
multiple algorithms to the assignment of trips to the network using the traditional Path-Size logit.

Full process overview
---------------------

The estimation of route choice models based on vehicle GPS data can be explored on a family of papers scheduled to
be presented at the ATRF 2024 [1]_, [2]_, [3]_.


.. [1] Camargo, P. V. de, and R. Imai. Map-Matching Large Streams of Vehicle GPS Data into Bespoke Networks (Submitted).
       Presented at the ATRF, Melbourne, 2024.

.. [2] Moss, J., P. V. de Camargo, C. de Freitas, and R. Imai. High-Performance Route Choice Set Generation on
       Large Networks (Submitted). Presented at the ATRF, Melbourne, 2024.

.. [3] Zill, J. C., and P. V. de Camargo. State-Wide Route Choice Models (Submitted).
       Presented at the ATRF, Melbourne, Australia, 2024.



.. toctree::
    :maxdepth: 1

    route_choice/choice_set_generation.rst
    route_choice/route_choice_model.rst