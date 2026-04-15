import multiprocessing as mp


def set_cores(cores_count: int):
    if isinstance(cores_count, int):
        if cores_count < 0:
            return max(1, mp.cpu_count() + cores_count)
        if cores_count == 0:
            return mp.cpu_count()
        elif cores_count > 0:
            return min(mp.cpu_count(), cores_count)
    else:
        raise ValueError("Number of cores needs to be an integer")
