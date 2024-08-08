"""
Tests of the hyperpath module.

py.test aequilibrae/tests/paths/test_hyperpath.py

Bell's network construction by François Pacull
https://aetperf.github.io/2023/05/10/Hyperpath-routing-in-the-context-of-transit-assignment.html
"""

import numpy as np
import pandas as pd
from unittest import TestCase
from aequilibrae.paths.public_transport import HyperpathGenerating


def create_vertices(n):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    xv, yv = np.meshgrid(x, y, indexing="xy")
    vertices = pd.DataFrame()
    vertices["x"] = xv.ravel()
    vertices["y"] = yv.ravel()
    return vertices


def create_edges(n, seed):
    m = 2 * n * (n - 1)
    tail = np.zeros(m, dtype=np.uint32)
    head = np.zeros(m, dtype=np.uint32)
    k = 0
    for i in range(n - 1):
        for j in range(n):
            tail[k] = i + j * n
            head[k] = i + 1 + j * n
            k += 1
            tail[k] = j + i * n
            head[k] = j + (i + 1) * n
            k += 1

    edges = pd.DataFrame()
    edges["tail"] = tail
    edges["head"] = head

    rng = np.random.default_rng(seed=seed)
    edges["trav_time"] = rng.uniform(0.0, 1.0, m)
    edges["delay_base"] = rng.uniform(0.0, 1.0, m)
    return edges


def create_SF_network(dwell_time=1.0e-6, board_alight_ratio=0.5):
    """
    Example network from Spiess, H. and Florian, M. (1989).
    Optimal strategies: A new assignment model for transit networks.
    Transportation Research Part B 23(2), 83-102.

    This network has 13 vertices and 24 edges.
    """

    boarding_time = board_alight_ratio * dwell_time
    alighting_time = board_alight_ratio * dwell_time

    line1_freq = 1.0 / (60.0 * 12.0)
    line2_freq = 1.0 / (60.0 * 12.0)
    line3_freq = 1.0 / (60.0 * 30.0)
    line4_freq = 1.0 / (60.0 * 6.0)

    # stop A
    # 0 stop vertex
    # 1 boarding vertex
    # 2 boarding vertex

    # stop X
    # 3 stop vertex
    # 4 boarding vertex
    # 5 alighting vertex
    # 6 boarding vertex

    # stop Y
    # 7  stop vertex
    # 8  boarding vertex
    # 9  alighting vertex
    # 10 boarding vertex
    # 11 alighting vertex

    # stop B
    # 12 stop vertex
    # 13 alighting vertex
    # 14 alighting vertex
    # 15 alighting vertex

    tail = []
    head = []
    trav_time = []
    freq = []
    vol = []

    # edge 0
    # stop A : to line 1
    # boarding edge
    tail.append(0)
    head.append(2)
    freq.append(line1_freq)
    trav_time.append(boarding_time)
    vol.append(0.5)

    # edge 1
    # stop A : to line 2
    # boarding edge
    tail.append(0)
    head.append(1)
    freq.append(line2_freq)
    trav_time.append(boarding_time)
    vol.append(0.5)

    # edge 2
    # line 1 : first segment
    # on-board edge
    tail.append(2)
    head.append(15)
    freq.append(np.inf)
    trav_time.append(25.0 * 60.0)
    vol.append(0.5)

    # edge 3
    # line 2 : first segment
    # on-board edge
    tail.append(1)
    head.append(5)
    freq.append(np.inf)
    trav_time.append(7.0 * 60.0)
    vol.append(0.5)

    # edge 4
    # stop X : from line 2
    # alighting edge
    tail.append(5)
    head.append(3)
    freq.append(np.inf)
    trav_time.append(alighting_time)
    vol.append(0.0)

    # edge 5
    # stop X : in line 2
    # dwell edge
    tail.append(5)
    head.append(6)
    freq.append(np.inf)
    trav_time.append(dwell_time)
    vol.append(0.5)

    # edge 6
    # stop X : from line 2 to line 3
    # transfer edge
    tail.append(5)
    head.append(4)
    freq.append(line3_freq)
    trav_time.append(dwell_time)
    vol.append(0.0)

    # edge 7
    # stop X : to line 2
    # boarding edge
    tail.append(3)
    head.append(6)
    freq.append(line2_freq)
    trav_time.append(boarding_time)
    vol.append(0.0)

    # edge 8
    # stop X : to line 3
    # boarding edge
    tail.append(3)
    head.append(4)
    freq.append(line3_freq)
    trav_time.append(boarding_time)
    vol.append(0.0)

    # edge 9
    # line 2 : second segment
    # on-board edge
    tail.append(6)
    head.append(11)
    freq.append(np.inf)
    trav_time.append(6.0 * 60.0)
    vol.append(0.5)

    # edge 10
    # line 3 : first segment
    # on-board edge
    tail.append(4)
    head.append(9)
    freq.append(np.inf)
    trav_time.append(4.0 * 60.0)
    vol.append(0.0)

    # edge 11
    # stop Y : from line 3
    # alighting edge
    tail.append(9)
    head.append(7)
    freq.append(np.inf)
    trav_time.append(alighting_time)
    vol.append(0.0)

    # edge 12
    # stop Y : from line 2
    # alighting edge
    tail.append(11)
    head.append(7)
    freq.append(np.inf)
    trav_time.append(alighting_time)
    vol.append(0.0)

    # edge 13
    # stop Y : from line 2 to line 3
    # transfer edge
    tail.append(11)
    head.append(10)
    freq.append(line3_freq)
    trav_time.append(dwell_time)
    vol.append(0.0833333333333)

    # edge 14
    # stop Y : from line 2 to line 4
    # transfer edge
    tail.append(11)
    head.append(8)
    freq.append(line4_freq)
    trav_time.append(dwell_time)
    vol.append(0.4166666666666)

    # edge 15
    # stop Y : from line 3 to line 4
    # transfer edge
    tail.append(9)
    head.append(8)
    freq.append(line4_freq)
    trav_time.append(dwell_time)
    vol.append(0.0)

    # edge 16
    # stop Y : in line 3
    # dwell edge
    tail.append(9)
    head.append(10)
    freq.append(np.inf)
    trav_time.append(dwell_time)
    vol.append(0.0)

    # edge 17
    # stop Y : to line 3
    # boarding edge
    tail.append(7)
    head.append(10)
    freq.append(line3_freq)
    trav_time.append(boarding_time)
    vol.append(0.0)

    # edge 18
    # stop Y : to line 4
    # boarding edge
    tail.append(7)
    head.append(8)
    freq.append(line4_freq)
    trav_time.append(boarding_time)
    vol.append(0.0)

    # edge 19
    # line 3 : second segment
    # on-board edge
    tail.append(10)
    head.append(14)
    freq.append(np.inf)
    trav_time.append(4.0 * 60.0)
    vol.append(0.0833333333333)

    # edge 20
    # line 4 : first segment
    # on-board edge
    tail.append(8)
    head.append(13)
    freq.append(np.inf)
    trav_time.append(10.0 * 60.0)
    vol.append(0.4166666666666)

    # edge 21
    # stop Y : from line 1
    # alighting edge
    tail.append(15)
    head.append(12)
    freq.append(np.inf)
    trav_time.append(alighting_time)
    vol.append(0.5)

    # edge 22
    # stop Y : from line 3
    # alighting edge
    tail.append(14)
    head.append(12)
    freq.append(np.inf)
    trav_time.append(alighting_time)
    vol.append(0.0833333333333)

    # edge 23
    # stop Y : from line 4
    # alighting edge
    tail.append(13)
    head.append(12)
    freq.append(np.inf)
    trav_time.append(alighting_time)
    vol.append(0.4166666666666)

    edges = pd.DataFrame(
        data={
            "tail": tail,
            "head": head,
            "trav_time": trav_time,
            "freq": freq,
            "volume_ref": vol,
        }
    )
    # waiting time is in average half of the period
    edges["freq"] *= 2.0

    demand = pd.DataFrame({"origin_vertex_id": [0], "destination_vertex_id": [12], "demand": [1.0]})

    return edges, demand


class TestHyperPath(TestCase):
    def _setUp(self, network="bell", n=10, alpha=10.0, seed=124) -> None:
        """
        Use our own setup method to allow specifying args for network creation
        """
        if network == "bell":
            self.vertices = create_vertices(n)
            self.edges = create_edges(n, seed=seed)

            delay_base = self.edges.delay_base.values
            indices = np.where(delay_base == 0.0)
            delay_base[indices] = 1.0
            freq_base = 1.0 / delay_base
            freq_base[indices] = np.inf
            self.edges["freq_base"] = freq_base

            if alpha == 0.0:
                self.edges["freq"] = np.inf
            else:
                self.edges["freq"] = self.edges.freq_base / alpha

            self.demand = pd.DataFrame(
                {
                    "orig_vert_idx": self.vertices.index[:10],
                    "dest_vert_idx": np.flip(self.vertices.index[-10:]),
                    "demand": np.full(10, 1),
                }
            )

        elif network == "SF":
            self.edges, self.demand = create_SF_network(dwell_time=0.0)

        else:
            raise KeyError(f'Unknown network type "{network}"')

    def tearDown(self) -> None:
        try:
            del self.vertices, self.edges, self.demand
        except NameError:
            pass
        except AttributeError:
            pass

    def test_bell_assign_parallel_agreement(self) -> None:
        self._setUp(network="bell")

        hp = HyperpathGenerating(self.edges)

        # breakpoint()
        results = []
        for threads in [1, 2, 4]:
            hp.assign(
                self.demand["orig_vert_idx"].values.astype(np.uint32),
                self.demand["dest_vert_idx"].values.astype(np.uint32),
                self.demand["demand"].values.astype(np.float64),
                check_demand=True,
                threads=threads,
            )
            results.append(hp._edges.copy(deep=True))

        for result in results[1:]:
            pd.testing.assert_frame_equal(results[0], result)

    def test_SF_run_01(self):
        self._setUp(network="SF")

        hp = HyperpathGenerating(self.edges)
        hp.run(origin=0, destination=12, volume=1.0)

        np.testing.assert_allclose(self.edges["volume_ref"].values, hp._edges["volume"].values, rtol=1e-05, atol=1e-08)

        u_i_vec_ref = np.array(
            [
                1.66500000e03,
                1.47000000e03,
                1.50000000e03,
                1.14428572e03,
                4.80000000e02,
                1.05000000e03,
                1.05000000e03,
                6.90000000e02,
                6.00000000e02,
                2.40000000e02,
                2.40000000e02,
                6.90000000e02,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ]
        )

        np.testing.assert_allclose(u_i_vec_ref, hp.u_i_vec, rtol=1e-08, atol=1e-08)

    def test_SF_assign_01(self):
        self._setUp(network="SF")

        hp = HyperpathGenerating(self.edges)

        hp.assign(
            self.demand["origin_vertex_id"].values.astype(np.uint32),
            self.demand["destination_vertex_id"].values.astype(np.uint32),
            self.demand["demand"].values.astype(np.float64),
            check_demand=True,
        )

        np.testing.assert_allclose(self.edges["volume_ref"].values, hp._edges["volume"].values, rtol=1e-05, atol=1e-08)
