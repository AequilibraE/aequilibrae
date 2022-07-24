import logging


_current_project = None


def activate_project(project):
    global _current_project
    _current_project = project


def get_active_project(must_exist=True):
    if not _current_project and must_exist:
        raise FileNotFoundError("There is no active Project set")
    return _current_project


def get_logger():
    proj = get_active_project(must_exist=False)
    if proj is None or proj.logger is None:
        return logging.getLogger("aequilibrae")
    return proj.logger
