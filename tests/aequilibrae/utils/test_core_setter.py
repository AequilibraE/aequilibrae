from aequilibrae.utils.cython.openmp_helper import omp_get_max_threads

from aequilibrae.utils.core_setter import (
    DEFAULT_THREADING_THRESHOLD,
    ELEMENTWISE_CORES_CAP,
    resolve_cores,
    resolve_elementwise_cores,
    resolve_threading_threshold,
)


class TestResolveCores:
    def test_reads_parameters(self, monkeypatch):
        monkeypatch.delenv("AEQ_CPUS", raising=False)
        assert resolve_cores({"cpus": 12}) == min(12, omp_get_max_threads())

    def test_env_var_wins_over_parameters(self, monkeypatch):
        monkeypatch.setenv("AEQ_CPUS", "4")
        assert resolve_cores({"cpus": 12}) == min(4, omp_get_max_threads())

    def test_values_are_clamped(self, monkeypatch):
        monkeypatch.setenv("AEQ_CPUS", "0")
        assert resolve_cores({}) == omp_get_max_threads()

        monkeypatch.setenv("AEQ_CPUS", "-2")
        assert resolve_cores({}) == max(1, omp_get_max_threads() - 2)

    def test_missing_key_resolves_to_all_cores(self, monkeypatch):
        monkeypatch.delenv("AEQ_CPUS", raising=False)
        assert resolve_cores({}) == omp_get_max_threads()

    def test_unparseable_values_resolve_to_all_cores(self, monkeypatch):
        monkeypatch.setenv("AEQ_CPUS", "garbage")
        assert resolve_cores({"cpus": 12}) == omp_get_max_threads()

        monkeypatch.delenv("AEQ_CPUS")
        assert resolve_cores({"cpus": None}) == omp_get_max_threads()


class TestResolveThreadingThreshold:
    def test_reads_parameters(self, monkeypatch):
        monkeypatch.delenv("AEQ_THREADING_THRESHOLD", raising=False)
        assert resolve_threading_threshold({"threading_threshold": 5000}) == 5000

    def test_env_var_wins_over_parameters(self, monkeypatch):
        monkeypatch.setenv("AEQ_THREADING_THRESHOLD", "77777")
        assert resolve_threading_threshold({"threading_threshold": 5000}) == 77777

    def test_missing_key_resolves_to_default(self, monkeypatch):
        monkeypatch.delenv("AEQ_THREADING_THRESHOLD", raising=False)
        assert resolve_threading_threshold({}) == DEFAULT_THREADING_THRESHOLD


class TestResolveElementwiseCores:
    def default_for(self, cores):
        return min(cores, ELEMENTWISE_CORES_CAP)

    def test_default_caps_team_size(self, monkeypatch):
        monkeypatch.delenv("AEQ_ELEMENTWISE_CPUS", raising=False)
        assert resolve_elementwise_cores({}, 16) == self.default_for(16)

    def test_default_never_exceeds_cores(self, monkeypatch):
        monkeypatch.delenv("AEQ_ELEMENTWISE_CPUS", raising=False)
        assert resolve_elementwise_cores({}, 2) == self.default_for(2)
        assert resolve_elementwise_cores({}, 1) == 1

    def test_reads_parameters(self, monkeypatch):
        monkeypatch.delenv("AEQ_ELEMENTWISE_CPUS", raising=False)
        assert resolve_elementwise_cores({"elementwise_cpus": 2}, 16) == 2

    def test_env_var_wins_over_parameters(self, monkeypatch):
        monkeypatch.setenv("AEQ_ELEMENTWISE_CPUS", "3")
        assert resolve_elementwise_cores({"elementwise_cpus": 2}, 16) == 3

    def test_explicit_values_follow_set_cores_conventions(self, monkeypatch):
        monkeypatch.setenv("AEQ_ELEMENTWISE_CPUS", "0")
        assert resolve_elementwise_cores({}, 16) == omp_get_max_threads()

        monkeypatch.setenv("AEQ_ELEMENTWISE_CPUS", "-2")
        assert resolve_elementwise_cores({}, 16) == max(1, omp_get_max_threads() - 2)

    def test_unparseable_values_resolve_to_default(self, monkeypatch):
        monkeypatch.setenv("AEQ_ELEMENTWISE_CPUS", "garbage")
        assert resolve_elementwise_cores({}, 16) == self.default_for(16)

        monkeypatch.delenv("AEQ_ELEMENTWISE_CPUS")
        assert resolve_elementwise_cores({"elementwise_cpus": None}, 16) == self.default_for(16)


class TestAssignmentResultsResolution:
    def test_env_vars_flow_into_results(self, monkeypatch):
        from aequilibrae.paths.results import AssignmentResults

        monkeypatch.setenv("AEQ_CPUS", "-2")
        monkeypatch.setenv("AEQ_THREADING_THRESHOLD", "5000")
        monkeypatch.setenv("AEQ_ELEMENTWISE_CPUS", "3")
        res = AssignmentResults()
        assert res.cores == max(1, omp_get_max_threads() - 2)
        assert res.threading_threshold == 5000
        assert res.elementwise_cores == 3

    def test_set_cores_keeps_threshold_unless_given(self, monkeypatch):
        from aequilibrae.paths.results import AssignmentResults

        monkeypatch.delenv("AEQ_CPUS", raising=False)
        monkeypatch.delenv("AEQ_THREADING_THRESHOLD", raising=False)
        monkeypatch.delenv("AEQ_ELEMENTWISE_CPUS", raising=False)
        res = AssignmentResults()
        assert res.threading_threshold == DEFAULT_THREADING_THRESHOLD

        res.set_cores(2)
        assert res.cores == 2
        assert res.threading_threshold == DEFAULT_THREADING_THRESHOLD

        res.set_cores(2, 123)
        assert res.threading_threshold == 123

    def test_elementwise_cores_track_cores_unless_pinned(self, monkeypatch):
        from aequilibrae.paths.results import AssignmentResults

        monkeypatch.delenv("AEQ_CPUS", raising=False)
        monkeypatch.delenv("AEQ_THREADING_THRESHOLD", raising=False)
        monkeypatch.delenv("AEQ_ELEMENTWISE_CPUS", raising=False)
        res = AssignmentResults()

        res.set_cores(2)
        assert res.elementwise_cores == 2

        res.set_cores(0)
        assert res.elementwise_cores == min(omp_get_max_threads(), ELEMENTWISE_CORES_CAP)

        res.set_cores(0, elementwise_cores=1)
        assert res.elementwise_cores == 1
