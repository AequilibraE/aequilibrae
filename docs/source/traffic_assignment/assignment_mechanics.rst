Assignment mechanics
--------------------

Performing traffic assignment, or even just computing paths through a network is
always a little different in each platform, and in AequilibraE is not different.

The complexity in computing paths through a network comes from the fact that
transportation models usually house networks for multiple transport modes, so
the loads (links) available for a passenger car may be different than those available
for a heavy truck, as it happens in practice.

For this reason, all path computation in AequilibraE happens through Graph objects.
While users can operate models by simply selecting the mode they want AequilibraE to
create graphs for, Graph objects can also be manipulated in memory or even created
from networks that are :ref:`NOT housed inside an AequilibraE model <plot_assignment_without_model>`.



.. _traffic_assignment_procedure:

.. include:: traffic_assignment_procedure.rst