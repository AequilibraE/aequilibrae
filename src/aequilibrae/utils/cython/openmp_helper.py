import os


def omp_get_num_threads() -> int:
    return 1


def omp_get_max_threads() -> int:
    return os.cpu_count() or 1
