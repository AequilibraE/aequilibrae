:orphan:

IPF Performance
===============

The use of iterative proportional fitting (IPF) is quite common on processes
involving doubly-constraining matrices, such as synthetic gravity models and
fractional split models (aggregate destination-choice models).

As this is a commonly used algorithm, we have implemented it in Cython, where
we can take full advantage of multi-core CPUs. We have also implemented the ability
of using both 32-bit and 64-bit floating-point seed matrices, which has direct impact
on cache use and consequently computational performance.

In this section, we compare the runtime of AequilibraE's current implementation of IPF, 
with a general IPF algorithm written in pure Python, available 
`here <https://github.com/joshchea/python-tdm/blob/master/scripts/CalcDistribution.py>`_.

The figure below compares AequilibraE's IPF runtime with one core with the benchmark Python
code. From the figure below, we can notice that the runtimes were practically the same for the
instances with 1,000 zones or less. As the number of zones increases, AequilibraE demonstrated to be slightly faster
than the benchmark python code, while applying IPF to a 32-bit NumPy array (``np.float32``) was significantly faster.

It's worth mentioning that the user can set up a threshold for AequilibraE's IPF function, 
as well as use more than one core to speed up the fitting process.

.. image:: ../../images/ipf_runtime_aequilibrae_vs_benchmark.png
    :align: center
    :alt: AequilibraE's IPF runtime

As IPF is an embarrassingly-parallel workload, it is more relevant to look at the performance of the
AequilibraE implementations, starting by comparing the implementation performance for inputs in 32 vs 64
bits using 32 threads.

.. image:: ../../images/ipf_runtime_32vs64bits.png
    :align: center
    :alt: AequilibraE's IPF runtime 32 vs 64 bits

The difference is staggering, with the 32-bit implementation being twice as fast as the 64-bit one for large matrices.
It is also worth noting that differences in results between the outputs between these two versions are incredibly
small (RMSE < 1.1e-10), and therefore unlikely to be relevant in most applications.

We can also look at performance gain across matrix sizes and number of cores, and it becomes clear
that the 32-bit version scales significantly better than its 64-bit counterpart, showing significant performance
gains up to 16 threads, while the latter stops showing much improvement beyond 8 threads, likely due to limitations
on cache size.

.. image:: ../../images/ipf_runtime_vs_num_cores.png
    :align: left
    :alt: number of cores used in IPF for 64 bit matrices

.. image:: ../../images/ipf_runtime_vs_num_cores32bits.png
    :align: right
    :alt: number of cores used in IPF for 32 bit matrices

In conclusion, AequilibraE's IPF implementation is over 11 times faster than its pure Python counterpart for
large matrices on a workstation, largely due to the use of Cython and multi-threading, but also due to the use of a
32-bit version of the algorithm.

These tests were run on a Threadripper 3970x (released in 2019) workstation with 32 cores (64 threads) @ 3.7 GHz
and 256 Gb of RAM. The code is provided below for reference.
