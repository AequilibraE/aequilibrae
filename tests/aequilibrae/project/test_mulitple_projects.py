from aequilibrae.context import get_active_project
import pytest


class TestMultipleProjects:
    @pytest.fixture(scope="class")
    def project(self, create_empty_project_session):
        return create_empty_project_session()

    def test_current_project_is_active_project(self, project):
        assert project is get_active_project()

    def test_switch_project(self, create_empty_project):
        proj2 = create_empty_project()
        assert proj2 is get_active_project()

    def test_reactivate_project(self, project, create_empty_project):
        create_empty_project()
        project.activate()
        assert project is get_active_project()

    def test_raises_when_inactive(self, project):
        project.deactivate()
        with pytest.raises(FileNotFoundError):
            get_active_project()

    def test_close_project_deactivates(self, project):
        project.close()
        assert get_active_project(must_exist=False) is None
