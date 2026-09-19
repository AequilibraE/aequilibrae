cimport openmp


def omp_get_num_threads() -> int:
    return openmp.omp_get_num_threads()


def omp_get_max_threads() -> int:
    return openmp.omp_get_max_threads()
