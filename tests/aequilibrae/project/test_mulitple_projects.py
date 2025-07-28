from aequilibrae.context import get_active_project
import pytest


class TestMultipleProjects:
    def test_current_project_is_active_project(self, empty_project):
        assert empty_project is get_active_project()

    def test_switch_project(self, empty_project):
        # Create a new project instance by re-activating the fixture
        empty_project.deactivate()
        empty_project.activate()
        assert empty_project is get_active_project()

    def test_reactivate_project(self, empty_project):
        empty_project.deactivate()
        empty_project.activate()
        assert empty_project is get_active_project()

    def test_raises_when_inactive(self, empty_project):
        empty_project.deactivate()
        with pytest.raises(FileNotFoundError):
            get_active_project()

    def test_close_project_deactivates(self, empty_project):
        empty_project.close()
        assert get_active_project(must_exist=False) is None
