:html_theme.sidebar_secondary.remove:

.. raw:: html

    <style type="text/css">
         .bd-main .bd-content .bd-article-container {max-width: 80%;}
         .big-font {
             font-size: var(--pst-font-size-h5);
             font-weight: bolder;
         }
    </style>

Python
======

**Download documentation**: :download:`HTML <https://www.aequilibrae.com/latest/python/aequilibrae.zip>` | :download:`PDF <_static/latex/aequilibrae.pdf>`

**Previous versions**: documentation for all AequilibraE's versions are available 
:doc:`here <useful_links/version_history>`.

**Useful links**: :doc:`useful_links/installation` | :doc:`API Reference <useful_links/_autosummary/aequilibrae>` | :doc:`useful_links/development` | :doc:`useful_links/support` |  :doc:`useful_links/history` | :doc:`_auto_examples/index`

|

.. grid::

  .. grid-item-card::
      :text-align: center
      :class-footer: sd-bg-light sd-font-weight-bold

      .. rst-class:: big-font 

          :doc:`The AequilibraE Project <aequilibrae_project>`

      Get to know the structure of an AequilibraE project
      +++
      :doc:`_auto_examples/aequilibrae_project/index`

  .. grid-item-card::
      :text-align: center
      :class-footer: sd-bg-light sd-font-weight-bold

      .. rst-class:: big-font 

          :doc:`Run module <run_module>`

      Run entire model pipelines from AequilibraE
      +++
      :doc:`_auto_examples/run_module/index`

  .. grid-item-card::
      :text-align: center
      :class-footer: sd-bg-light sd-font-weight-bold

      .. rst-class:: big-font 

          :doc:`Network Manipulation <network_manipulation>`

      Create and edit networks and models
      +++
      :doc:`_auto_examples/network_manipulation/index`

.. grid::

  .. grid-item-card::
    :text-align: center
    :class-footer: sd-bg-light sd-font-weight-bold

    .. rst-class:: big-font 
      
      :doc:`Distribution Procedures <distribution_procedures>`

    Calibrate and apply gravity models and perform IPF
    +++
    :doc:`_auto_examples/distribution_procedures/index`

  .. grid-item-card::
    :text-align: center
    :class-footer: sd-bg-light sd-font-weight-bold

    .. rst-class:: big-font 
      
      :doc:`path_computation`

    Create skim matrices and compute the shortest path
    +++
    :doc:`_auto_examples/path_computation/index`

  .. grid-item-card::
    :text-align: center
    :class-footer: sd-bg-light sd-font-weight-bold

    .. rst-class:: big-font 
      
      :doc:`Traffic Assignment <static_traffic_assignment>`

    Run traffic allocation.
    +++
    :doc:`_auto_examples/traffic_assignment/index`

.. grid::

  .. grid-item-card:: 
    :text-align: center
    :class-footer: sd-bg-light sd-font-weight-bold

    .. rst-class:: big-font 

      :doc:`Public Transport <public_transport>`

    Add a transit feed or run transit assignment
    +++
    :doc:`_auto_examples/public_transport/index`

  .. grid-item-card::
    :text-align: center
    :class-footer: sd-bg-light sd-font-weight-bold

    .. rst-class:: big-font 
      
      :doc:`Route Choice <route_choice>`

    Explore the route choice models.
    +++
    :doc:`_auto_examples/route_choice/index`

  .. grid-item-card::
    :text-align: center
    :class-footer: sd-bg-light sd-font-weight-bold

    .. rst-class:: big-font 
      
      :doc:`other_applications`

    Explore other applcations of AequilibraE!
    +++
    :doc:`_auto_examples/other_applications/index`

.. toctree::
   :hidden:
   :maxdepth: 1

   aequilibrae_project
   run_module
   network_manipulation
   distribution_procedures
   path_computation
   static_traffic_assignment
   public_transport
   route_choice
   other_applications
