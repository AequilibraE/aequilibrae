Matrix estimation via AequilibraE's ODME Procedure
---------------------------------------------------

Origin-Destination Matrix Estimation involves fitting an initial estimate for an
OD matrix to more accurately represent count volume data. Within AequilibraE the
**ODME** object handles this procedure. The **ODME** object relies heavily on the
input **TrafficAssignment** object, details of which :ref:`are given here <assignment_mechanics>`.

AequilibraE ODME Overview
~~~~~~~~~~~~~~~~~~~~~~~~~
The **ODME** infrastructure involves 2 layers of iterative processes (ie a bilevel optimisation
problem). There is a so called 'outer loop' and for each iteration within such an outer loop
there is an 'inner loop'. The outer (or upper) optimisation task involves actually attempting
to converge upon a local (close to the initial demand matrices) solution which accurately fits
the count volume data. The inner (or lower) optimisation problem involves making certain assumptions
about the current state of the problem (see :ref:`technical documentation <at some link>`` for
details) and attempting to solve the problem based on these assumptions. For various algorithms
the process of optimising within this inner loop is what typically defines each iterative
gradient based algorithm. This inner loop involves iteratively finding factor matrices which 
based on the current assumptions/constraints should optimise the solution best. Implementation of
algorithms themselves can be seen in the **ScalingFactors** class, whilst the main loop infrastructure
is contained in the **ODME** class. There is also a results/statistics gathering class called
**ODMEResults** which operates alongside the **ODME** class.

ODME Parameters
~~~~~~~~~~~~~~~
There are a few parameters the user must specify upon initialisation of the ODME object:

* **assignment**: A TrafficAssignment object containing all necessary information to assign flows
  to each link given input demand matrices (see :ref:`the TrafficAssignment page <assignment_mechanics>`
  for details).

* **count_volumes**: The data collected on which **ODME** will attempt to optimise the demand matrices.
  These should be given in a pandas dataframe as shown in :ref:`plot_matrix_estimation`.

* **stop_crit**: Given as a dictionary (see :ref:`plot_matrix_estimation` for details) which provide the
  necessary criterion for controlling the iterative process - in particular both the maximum number of
  iterations for both the inner and outer loops, and the convergence threshold (a threshold for an objective
  function determining when to exit the loop) for both the inner and outer loop. In particular, note that
  the inner convergence criterion specifies a change in convergence rather than a fixed threshold, since 
  we do not always expect an inner loop to necessarily converge to 0.

* **algorithm**: The algorithm to be used - currently there are 3 algorithms implemented and "spiess" 
  is currently the most reliable.

* **alpha**: This is a hyper-parameter used currently only for the regularised spiess algorithm.
  See the :ref:`technical documentation <some link here>` for details.

Execution
~~~~~~~~~

Algorithms
~~~~~~~~~~

For now, the only algorithms available in AequilibraE are:

* Geometric Mean: 
This algorithm is usable for very simple cases (very few count volumes), however it typically is slower
and less likely to converge in more complex cases - it was originally used just as a testing algorithm
during development.

* Spiess:
The original paper by Spiess (1990) is :ref:`given here <REFERENCE HERE>` and we have implemented the
algorithm shown there. See the :ref:`technical documentation <some link here>` for details on this. The 
main goal of Spiess at each inner loop is to minimise the following objective function:
.. math:: Z(demand) = \sum\limits_{a \in counts} \left(flow(a, demand) - count(a))^2

* Regularised Spiess:
This is a more novel algorithm intended to try and ensure the solution we obtain is closer to 
the initial set of demand matrices. How tightly we control this is dependent on the input 
hyper-parameter alpha. See the :ref:`technical documentation <some link here>` for details. This
procedure still requires testing to determine how useful it is - and users should feel free to
try it out for themselves.

Results
~~~~~~~
To obtain and view the results of an ODME procedure, 

Stopping Criterion
~~~~~~~~~~~~~~~~~~
The main hyper-parameter's to each iterative gradient based ODME procedure are the stopping criterion
(although some algorithms have additional parameters). 
