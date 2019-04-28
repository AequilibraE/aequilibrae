###########
AequilibraE
###########

AequilibraE is the first comprehensive Python package for transportation modeling, and it aims to provide all the
resources not easily available from other open-source packages in the Python (NumPy, really) ecosystem.


Project status
##############


.. image:: https://travis-ci.org/AequilibraE/aequilibrae.svg?branch=master
    :target: https://travis-ci.org/AequilibraE/aequilibrae

.. image:: https://coveralls.io/repos/github/AequilibraE/aequilibrae/badge.svg?branch=master
    :target: https://coveralls.io/github/AequilibraE/aequilibrae?branch=master

.. image:: https://codecov.io/gh/AequilibraE/aequilibrae/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/AequilibraE/aequilibrae

.. image:: https://readthedocs.org/projects/aequilibrae/badge/?version=latest
    :target: https://aequilibrae.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

What is available
#################

* Synthetic gravity/IPF
* Traffic assignment


What is not available yet
#########################
* GTFS exporters
* Traffic assignment
* Thorough documentation
* AEQ Project

What is not planned to be available any time soon
#################################################
All resources that are 


History
#######
Before there was AequilibraE, there was a  need for something like AequilibraE out there.

The very early days
*******************
student that needed low level access to outputs of standard algorithms used
in transportation modelling (e.g. path files from traffic assignment) and had that denied by the maker of the commercial 
software he normally used. There, the `first scratch of a traffic assignment procedure
<www.xl-optim.com/python-traffic-assignment>`_
was born.   
After that, there were a couple of scripts developed to implement synthetic gravity models (calibration and application)
that were develop for a government Think-tank in Brazil `IPEA <www.ipea.gov.br>`_.
Around the same time, another student needed a piece of code that transformed a GIS link layer into a proper graph,
where each link would become the connection between two nodes.

The first take on a release software
************************************
Having all those algorithms at hand, it made sense combining them into something more people could use, and by them it
seemed that QGIS was the way to go, so I developed the `very first version of AequilibraE
<http://www.xl-optim.com/introducing_aequilibrae>`_.
It was buggy as hell and there was very little, if any, software engineering built into it, and I slowly improved things.  

The first reasonable version
****************************
* AFTER DEVELOPING THE GRAPH AND OTHER OBJECTS

Evolving into proper software
*****************************
* TWO SEPARATE REPOS
* UNIT TESTS
* NEW COLLABORATORS
* API DOCUMENTATION




Other relevant Repositories
###########################

External hyperlinks, like `Python <http://www.python.org/>`_.

The other most important repository associated with this project is the one for the `QGIS GUI
<https://github.com/AequilibraE/AequilibraE-GUI>`_  That is where everything started.

The other important repository is the one with the `examples of usage <https://github.com/AequilibraE/examples_api>`_ of
this code. That repository will problay be replaced with proper documentation for this project, but one thing at a time.