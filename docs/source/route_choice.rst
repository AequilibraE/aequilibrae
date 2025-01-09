Route Choice
============

The route choice problem does not have a closed solution, and the selection of one of the many
existing frameworks for solution depends on many factors [1]_ [2]_. A common modelling framework 
in practice consists of two steps: choice set generation and the choice selection process.

AequilibraE is the first modeling package with full support for route choice, from the creation of 
choice sets through multiple algorithms to the assignment of trips to the network using the 
traditional path-size logit.

.. toctree::
    :maxdepth: 1
    :caption: Route Choice

    route_choice/choice_set_generation
    route_choice/path_size_logit
    route_choice/_auto_examples/index

References
----------

.. [1] Rieser-Schüssler, N., Balmer, M., and Axhausen, K.W. (2012). Route choice sets for very 
       high-resolution data. Transportmetrica A: Transport Science, 9(9), 825–845.
       https://doi.org/10.1080/18128602.2012.671383

.. [2] Zill, J.C. and Camargo, P.V. (2024) State-Wide Route Choice Models.
       Presented at the ATRF, Melbourne, Australia.
