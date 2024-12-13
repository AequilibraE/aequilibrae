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

**Download documentation**: :download:`HTML <_static/AequilibraE.zip>` | :download:`PDF <_static/latex/aequilibrae.pdf>`

**Previous versions**: documentation for all AequilibraE's versions are available 
:doc:`here <useful_links/version_history>`.

**Useful links**: :doc:`useful_links/installation` | :doc:`useful_links/api` | 
:doc:`useful_links/development` | :doc:`useful_links/support` |  :doc:`useful_links/history`

|

.. grid::

    .. grid-item-card::
        :text-align: center
        :class-footer: sd-bg-light sd-font-weight-bold

        .. rst-class:: big-font 

            :doc:`The AequilibraE Project <aequilibrae_project>`

        Get to know the structure of an AequilibraE project
        +++
        :doc:`aequilibrae_project/_auto_examples/index`

    .. grid-item-card::
        :text-align: center
        :class-footer: sd-bg-light sd-font-weight-bold

        .. rst-class:: big-font 

            :doc:`Project Components <project_components>`

        Get to know the components of each AequilibraE project
        +++
        :doc:`project_components/_auto_examples/index`

    .. grid-item-card::
        :text-align: center
        :class-footer: sd-bg-light sd-font-weight-bold

        .. rst-class:: big-font 

            :doc:`Network Manipulation <network_manipulation>`

        Create and edit networks and models
        +++
        :doc:`network_manipulation/_auto_examples/index`

.. grid::
    
   .. grid-item-card::
      :text-align: center
      :class-footer: sd-bg-light sd-font-weight-bold

      .. rst-class:: big-font 
        
        :doc:`Traffic Assignment <static_traffic_assignment>`

      Run traffic allocation.
      +++
      :doc:`traffic_assignment/_auto_examples/index`

   .. grid-item-card:: 
      :text-align: center
      :class-footer: sd-bg-light sd-font-weight-bold

      .. rst-class:: big-font 

         :ref:`Transit Assignment <transit_assignment>`

      Perform transit assignment for your transit data!
      +++
      :doc:`transit_assignment/_auto_examples/index`

   .. grid-item-card::
      :text-align: center
      :class-footer: sd-bg-light sd-font-weight-bold

      .. rst-class:: big-font 
        
        :ref:`Route Choice <route_choice>`

      Explore the route choice models.
      +++
      :doc:`route_choice/_auto_examples/index`

.. 
    .. raw:: html

    <footer class="prev-next-footer d-print-none">
        <div class="prev-next-area">
            <a class="right-next" href="aequilibrae_project.html" title="next page">
            <div class="prev-next-info">
                <p class="prev-next-subtitle">next</p>
                <p class="prev-next-title">The AequilibraE Project</p>
            </div>
            <i class="fa-solid fa-angle-right"></i>
            </a>
        </div>
    </footer>

.. toctree::
   :hidden:
   :maxdepth: 1

   aequilibrae_project
   project_components
   network_manipulation
   static_traffic_assignment
   transit_assignment/index
   route_choice/index
