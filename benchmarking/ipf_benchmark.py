"""Times the IPF of the installed AequilibraE version for 50 iterations, the best of three runs.

Example: python benchmarking/ipf_benchmark.py --sizes 1000 5000 --cores 1 24 --output ipf.csv
"""

from argparse import ArgumentParser
from time import perf_counter

import numpy as np
import pandas as pd

from aequilibrae.distribution.cython.ipf_core import ipf_core

parser = ArgumentParser()
parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 2500, 5000, 10000])
parser.add_argument("--cores", type=int, nargs="+", default=[1, 0])
parser.add_argument("--output")
args = parser.parse_args()

rng = np.random.default_rng(42)
results = []
for zones in args.sizes:
    seed, productions, attractions = rng.random((zones, zones)), rng.random(zones) * 1000, rng.random(zones) * 1000
    attractions *= productions.sum() / attractions.sum()
    for dtype in ["float64", "float32"]:
        for cores in args.cores:
            times = []
            for _ in range(3):
                trips = seed.astype(dtype)
                start = perf_counter()
                ipf_core(trips, productions, attractions, 50, -1, cores, warn=False)
                times.append(perf_counter() - start)
            results.append((zones, dtype, cores, min(times)))
            print(*results[-1], flush=True)

if args.output:
    pd.DataFrame(results, columns=["zones", "dtype", "cores", "seconds"]).to_csv(args.output, index=False)
