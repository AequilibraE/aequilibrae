IPF Performance
===============

The use of iterative proportional fitting (IPF) is quite common on processes
involving doubly-constraining matrices, such as synthetic gravity models and
fractional split models (aggregate destination-choice models).

As this is a commonly used algorithm, we have implemented it in Cython, where
we can take full advantage of multi-core CPUs. We have also implemented the ability
of using both 32-bit and 64-bit floating-point seed matrices, which has direct impact
on cache use and consequently computational performance.


In this section, we compare the
runtime of AequilibraE's current implementation of IPF, 
with a general IPF algorithm written in pure Python, available `here <https://github.com/joshchea/python-tdm/blob/master/scripts/CalcDistribution.py>`_.

The figure below compares AequilibraE's IPF runtime with one core with the benchmark Python
code. From the figure below, we can notice that the runtimes were practically the same for the
instances with 1,000 zones or less. As the number of zones increases, AequilibraE demonstrated to be slightly faster
than the benchmark python code, while applying IPF to a 32-bit NumPy array (np.float32) was significantly faster.
It's worth mentioning that
the user can set up a threshold for AequilibraE's IPF function, as well as use more than one
core to speed up the fitting process.

.. image:: ../images/ipf_runtime_aequilibrae_vs_benchmark.png
    :align: center
    :alt: AequilibraE's IPF runtime

As IPF is an embarrassingly-parallel workload, it is more relevant to look at the performance of the
AequilibraE implementations, starting by comparing the implementation performance for inputs in 32 vs 64
bits using 32 threads.

.. image:: ../images/ipf_runtime_32vs64bits.png
    :align: center
    :alt: AequilibraE's IPF runtime 32 vs 64 bits

The difference is staggering, with the 32-bit implementation being twice as fast as the 64-bit one for large matrices.
It is also worth noting that differences in results between the outputs between these two versions are incredibly
small (RMSE < 1.1e-10), and therefore unlikely to be relevant in most applications.

We can also look at performance gain across matrix sizes and number of cores, and it becomes clear
that the 32-bit version scales significantly better than its 64-bit counterpart, showing significant performance
gains up to 16 threads, while the latter stops showing much improvement beyond 8 threads, likely due to limitations
on cache size.

.. image:: ../images/ipf_runtime_vs_num_cores.png
    :align: left
    :alt: number of cores used in IPF for 64 bit matrices

.. image:: ../images/ipf_runtime_vs_num_cores32bits.png
    :align: right
    :alt: number of cores used in IPF for 32 bit matrices


These tests were run on a Threadripper 3970x (released in 2019) workstation with 32 cores (64 threads) @ 3.7 GHz
and 256 Gb of RAM. With 32 cores in use, performing IPF took 0.105s on a 10,000 zones matrix,
and 0.224 seconds on a 15,000 matrix. The code is provided below for convenience

.. _code-block-for-ipf-benchmarking:
.. code-block:: python

    # %%
    from copy import deepcopy
    from time import perf_counter
    import numpy as np
    import pandas as pd
    from aequilibrae.distribution.ipf_core import ipf_core
    from tqdm import tqdm

    # %%
    # From:
    # https://github.com/joshchea/python-tdm/blob/master/scripts/CalcDistribution.py

    def CalcFratar(ProdA, AttrA, Trips1, maxIter=10):
        '''Calculates fratar trip distribution
           ProdA = Production target as array
           AttrA = Attraction target as array
           Trips1 = Seed trip table for fratar
           maxIter (optional) = maximum iterations, default is 10
           Returns fratared trip table
        '''
        # print('Checking production, attraction balancing:')
        sumP = ProdA.sum()
        sumA = AttrA.sum()
        # print('Production: ', sumP)
        # print('Attraction: ', sumA)
        if sumP != sumA:
            # print('Productions and attractions do not balance, attractions will be scaled to productions!')
            AttrA = AttrA*(sumP/sumA)
        else:
            pass
            # print('Production, attraction balancing OK.')
        # Run 2D balancing --->
        for balIter in range(0, maxIter):
            ComputedProductions = Trips1.sum(1)
            ComputedProductions[ComputedProductions == 0] = 1
            OrigFac = (ProdA/ComputedProductions)
            Trips1 = Trips1*OrigFac[:, np.newaxis]

            ComputedAttractions = Trips1.sum(0)
            ComputedAttractions[ComputedAttractions == 0] = 1
            DestFac = (AttrA/ComputedAttractions)
            Trips1 = Trips1*DestFac
        return Trips1

    # %%
    mat_sizes = [500, 750, 1000, 1500, 2500, 5000, 7500, 10000, 15000
    cores_to_use = [1, 2, 4, 8, 16, 32]

    # %%
    #Benchmarking
    bench_data = []
    cores = 1
    repetitions = 5
    iterations = 100
    for zones in mat_sizes:
        for repeat in tqdm(range(repetitions), f"Repetitions for zone size {zones}"):
            mat1 = np.random.rand(zones, zones)
            target_prod = np.random.rand(zones)
            target_atra = np.random.rand(zones)
            target_atra *= target_prod.sum()/target_atra.sum()

            aeq_mat = deepcopy(mat1)
            # We use a nonsensical negative tolerance to force it to run all iterations
            # and set warning for non-convergence to false, as we know it won't converge
            t = perf_counter()
            ipf_core(aeq_mat, target_prod, target_atra, max_iterations=iterations, tolerance=-5, cores=cores, warn=False)
            aeqt = perf_counter() - t

            aeq_mat32 = np.array(mat1, np.float32)
            # We now run the same thing with a seed matrix in single-precision (float 32 bits) instead of double as above (64 bits)
            t = perf_counter()
            ipf_core(aeq_mat32, target_prod, target_atra, max_iterations=iterations, tolerance=-5, cores=cores, warn=False)
            aeqt2 = perf_counter() - t

            bc_mat = deepcopy(mat1)
            t = perf_counter()
            x = CalcFratar(target_prod, target_atra, bc_mat, maxIter=iterations)

            bench_data.append([zones, perf_counter() - t, aeqt, aeqt2])

    # %%
    bench_df = pd.DataFrame(bench_data, columns=["Zones in the model", "PythonCode", "AequilibraE", "AequilibraE-32bits"])
    bench_df.groupby(["Zones in the model"]).mean().plot.bar()

    # %%
    bench_df.groupby(["Zones in the model"]).mean()

    # %%
    #Benchmarking 32 threads
    bench_data_parallel = []
    cores = 32
    repetitions = 5
    iterations = 100
    for zones in mat_sizes:
        for repeat in tqdm(range(repetitions), f"Repetitions for zone size {zones}"):
            mat1 = np.random.rand(zones, zones)
            target_prod = np.random.rand(zones)
            target_atra = np.random.rand(zones)
            target_atra *= target_prod.sum()/target_atra.sum()

            aeq_mat = deepcopy(mat1)
            # We use a nonsensical negative tolerance to force it to run all iterations
            # and set warning for non-convergence to false, as we know it won't converge
            t = perf_counter()
            ipf_core(aeq_mat, target_prod, target_atra, max_iterations=iterations, tolerance=-5, cores=cores, warn=False)
            aeqt = perf_counter() - t

            aeq_mat32 = np.array(mat1, np.float32)
            # We now run the same thing with a seed matrix in single-precision (float 32 bits) instead of double as above (64 bits)
            t = perf_counter()
            ipf_core(aeq_mat32, target_prod, target_atra, max_iterations=iterations, tolerance=-5, cores=cores, warn=False)
            aeqt2 = perf_counter() - t

            rmse = np.sqrt(np.mean((aeq_mat-aeq_mat32)**2))

            bench_data_parallel.append([zones, aeqt, aeqt2, rmse])

    # %%
    bench_df_parallel = pd.DataFrame(bench_data_parallel, columns=["Zones in the model", "AequilibraE", "AequilibraE-32bits", "rmse"])
    bench_df_parallel.groupby(["Zones in the model"]).mean()[[ "AequilibraE", "AequilibraE-32bits"]].plot.bar()

    # %%
    bench_df_parallel.groupby(["Zones in the model"]).mean()

    # %%
    #Benchmarking
    aeq_data = []
    repetitions = 1
    iterations = 50
    for zones in mat_sizes:
        for cores in tqdm(cores_to_use,f"Zone size: {zones}"):
            for repeat in range(repetitions):
                mat1 = np.random.rand(zones, zones)
                target_prod = np.random.rand(zones)
                target_atra = np.random.rand(zones)
                target_atra *= target_prod.sum()/target_atra.sum()

                aeq_mat = deepcopy(mat1)
                t = perf_counter()
                ipf_core(aeq_mat, target_prod, target_atra, max_iterations=iterations, tolerance=-5, cores=cores)
                aeqt = perf_counter() - t

                aeq_data.append([zones, cores, aeqt])

    # %%
    aeq_df = pd.DataFrame(aeq_data, columns=["zones", "cores", "time"])
    aeq_df = aeq_df[aeq_df.zones>1000]
    aeq_df = aeq_df.groupby(["zones", "cores"]).mean().reset_index()
    aeq_df = aeq_df.pivot_table(index="zones", columns="cores", values="time")
    for cores in cores_to_use[::-1]:
        aeq_df.loc[:, cores] /= aeq_df[1]
    aeq_df.transpose().plot()