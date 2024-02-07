Matrix estimation via AequilibraE's ODME Procedure
---------------------------------------------------

Origin-Destination Matrix Estimation involves fitting an initial estimate for an
OD matrix to more accurately represent count volume data. Within AequilibraE the
**ODME** object handles this procedure. The **ODME** object relies heavily on the
input **TrafficAssignment** object, details of which :ref:`are given here <assignment_mechanics>`.

AequilibraE ODME Overview
~~~~~~~~~~~~~~~~~~~~~~~~~
The **ODME** infrastructure involves 2 layers of iterative processes (ie a bilevel optimisation
problem). There is a 'outer loop' and for each iteration within such an outer loop
there is an 'inner loop'. The outer (or upper) optimisation task involves actually attempting
to converge upon a local (close to the initial demand matrices) solution which accurately fits
the count volume data. Within the inner (or lower) optimisation problem we make the assumption that
paths will remain constant with small changes to the demand matrices and attempt to optimise with
respect to that (see :ref:`technical documentation <some link here>` for details). At the beginning
of each outer iteration we recompute paths to compensate for congestion generated over the previous 
inner loop. For various algorithms the process of optimising within this inner loop is what typically 
defines each iterative gradient based algorithm. This inner loop involves iteratively finding factor
matrices which based on the current assumptions/constraints should optimise the solution best. Implementation
of algorithms themselves can be seen in the **ScalingFactors** class, whilst the main loop infrastructure
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
See the :ref:`plot_matrix_estimation` page for a full example on how to execute **ODME**. Given that
you have already created an appropriate **TrafficAssignment** object and have the **count_volumes** dataframe
loaded you can execute the following:

.. code-block:: python

    odme = ODME(assignment, count_volumes)
    odme.execute(verbose=True, print_rate=5)

See :ref:`plot_matrix_estimation` for all optional arguments. During execution **verbose** will print out
updates to the terminal during runtime regarding how many outer loops have passed and will print at the rate
specified by **print_rate**.

Algorithms
~~~~~~~~~~

For now, the only algorithms available in AequilibraE are:

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

Stopping Criterion
~~~~~~~~~~~~~~~~~~
The main hyper-parameter's to each iterative gradient based ODME procedure are the stopping criterion
(although some algorithms have additional parameters). 

Results
~~~~~~~
There are 2 ways to extract the results of **ODME** - you can load them in memory with the
**get_demands()** method or save them to disk using the **save_to_project** method. However,
aside from these we may also want to determine the effectiveness of the **ODME** procedure itself.
Within the :ref:`plot_matrix_estimation` notebook there are a number of examples of such plots
showing how the error in link flows and factor size proceeds over various iterations. In particular,
the plots on link flow errors are useful to determine if the solution is converging appropriately 
(although for regularised spiess this is not alway intended to directly converge). Another important
plot is the cumulative factor distribution - this is useful for comparing different algorithms/runs of
**ODME** in order to determine the relative change to the intial demand matrices. Refer to the example
notebook at :ref:`plot_matrix_estimation` for more, here is the main code for obtaining results:
.. codeblock:: python

    odme.save_to_project("example_doc", "example_doc.omx", project=project)
    new_demands = odme.get_demands()

    iteration_stats = odme.results.get_iteration_statistics() # Statistics over iterations
    link_stats = odme.results.get_link_statistics() # Statistics tracking links
    cumulative_factors = odme.results.get_cumulative_factors() # Cumulative factor distribution
