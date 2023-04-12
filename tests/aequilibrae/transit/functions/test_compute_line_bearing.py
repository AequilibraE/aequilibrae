import pytest

from aequilibrae.transit.functions.compute_line_bearing import compute_line_bearing


class TestComputeLineBearing:
    def test_compute_line_bearing(self):
        assert pytest.approx(compute_line_bearing((-1.5639, -43.7397), (-1.5818, -43.7347))) == 15.6066221
        assert pytest.approx(compute_line_bearing((-1.5865, -43.7183), (-1.5808, -43.7064))) == 64.40597075
        assert pytest.approx(compute_line_bearing((-1.5641, -43.6992), (-1.5623, -43.7177))) == 84.44276779
