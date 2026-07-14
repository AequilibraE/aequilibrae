import multiprocessing as mp

from aequilibrae.utils.core_setter import DEFAULT_THREADING_THRESHOLD, resolve_cores, resolve_threading_threshold


class TestResolveCores:
    def test_reads_parameters(self, monkeypatch):
        monkeypatch.delenv("AEQ_CPUS", raising=False)
        assert resolve_cores({"cpus": 12}) == 12

    def test_env_var_wins_over_parameters(self, monkeypatch):
        monkeypatch.setenv("AEQ_CPUS", "4")
        assert resolve_cores({"cpus": 12}) == 4

    def test_missing_key_resolves_to_all_cores(self, monkeypatch):
        monkeypatch.delenv("AEQ_CPUS", raising=False)
        assert resolve_cores({}) == mp.cpu_count()

    def test_unparseable_values_resolve_to_all_cores(self, monkeypatch):
        monkeypatch.setenv("AEQ_CPUS", "garbage")
        assert resolve_cores({"cpus": 12}) == mp.cpu_count()

        monkeypatch.delenv("AEQ_CPUS")
        assert resolve_cores({"cpus": None}) == mp.cpu_count()


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


class TestAssignmentResultsResolution:
    def test_env_vars_flow_into_results(self, monkeypatch):
        from aequilibrae.paths.results import AssignmentResults

        monkeypatch.setenv("AEQ_CPUS", "-2")
        monkeypatch.setenv("AEQ_THREADING_THRESHOLD", "5000")
        res = AssignmentResults()
        assert res.cores == max(1, mp.cpu_count() - 2)
        assert res.threading_threshold == 5000

    def test_set_cores_keeps_threshold_unless_given(self, monkeypatch):
        from aequilibrae.paths.results import AssignmentResults

        monkeypatch.delenv("AEQ_CPUS", raising=False)
        monkeypatch.delenv("AEQ_THREADING_THRESHOLD", raising=False)
        res = AssignmentResults()
        assert res.threading_threshold == DEFAULT_THREADING_THRESHOLD

        res.set_cores(2)
        assert res.cores == 2
        assert res.threading_threshold == DEFAULT_THREADING_THRESHOLD

        res.set_cores(2, 123)
        assert res.threading_threshold == 123
