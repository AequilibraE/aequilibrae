Public Transport
================

Public transport data is a key element of transport planning in general [1]_. AequilibraE is 
capable of importing a General Transit Feed Specification (GTFS) to its public transport 
database. The GTFS is a standardized data format widely used in public transport planning and 
operation, and was first proposed during the 2000s', for public transit agencies to describe 
details from their services, such as schedules, stops, fares, etc [2]_. Currently,
there are two types of GTFS data:

* GTFS schedule, which contains information on routes, schedules, fares, and other details;
* GTFS realtime, which contains real-time vehicle position, trip updates, and service alerts.

The GTFS protocol is being constantly updated and so are AequilibraE's capabilities of handling
these changes. We strongly encourage you to take a look at the documentation provided 
by `Mobility Data <https://gtfs.org/documentation/schedule/reference/>`_.

In this section we also present the transit assignment models, which are mathematical tools that 
predict how passengers behave and travel in a transit network, given some assumptions and inputs.

Transit assignment models aim to answer questions such as: 

* How do transit passengers choose their routes in a complex network of lines and services?
* How can we estimate the distribution of passenger flows and the performance of transit systems?

.. seealso::
   
   * :ref:`public_transport_database`
      Database structure

.. toctree::
   :caption: Public Transport
   :maxdepth: 1
   
   public_transport/transit_graph
   public_transport/hyperpath_routing
   public_transport/_auto_examples/index

References
----------

.. [1] Pereira, R.H.M. and Herszenhut, D. (2023) Introduction to urban accessibility:
       a practical guide with R. Rio de Janeiro, IPEA. Available at:
       https://repositorio.ipea.gov.br/bitstream/11058/12689/52/Introduction_urban_accessibility_Book.pdf

.. [2] Mobility Data (2024) GTFS: Making Public Transit Data Universally Accessible. 
       Available at: https://gtfs.org/getting-started/what-is-GTFS/
