import os
import tempfile
import logging

from aequilibrae.parameters import Parameters


class Log:
    """API entry point to the log file contents

    .. code-block:: python

        >>> from aequilibrae import Project

        >>> project = Project.from_path("/tmp/test_project")

        >>> log = project.log()

        # We get all entries for the log file
        >>> entries = log.contents()

        # Or clear everything (NO UN-DOs)
        >>> log.clear()
    """

    def __init__(self, project_base_path: str):
        self.log_file_path = os.path.join(project_base_path, "aequilibrae.log")

    def contents(self) -> list:
        """Returns contents of log file

        :Return:
            **log_contents** (:obj:`list`): List with all entries in the log file
        """

        with open(self.log_file_path, "r") as file:
            return [x.strip() for x in file.readlines()]

    def clear(self):
        """Clears the log file. Use it wisely"""
        with open(self.log_file_path, "w") as _:
            pass


def _setup_logger():
    # CREATE THE GLOBAL LOGGER

    par = Parameters._default
    do_log = par["system"]["logging"]
    temp_folder = tempfile.gettempdir()

    logger = logging.getLogger("aequilibrae")
    logger.setLevel(logging.DEBUG)

    if not len(logger.handlers) and do_log:
        log_file = os.path.join(temp_folder, "aequilibrae.log")
        logger.addHandler(get_log_handler(log_file))
    return logger


def get_log_handler(log_file: str, ensure_file_exists=True):
    """Return a log handler that writes to the given log_file"""
    if os.path.exists(log_file) and not os.path.isfile(log_file):
        raise FileExistsError(f"{log_file} is not a valid file")

    if ensure_file_exists:
        open(log_file, "a").close()

    formatter = logging.Formatter("%(asctime)s;%(levelname)s ; %(message)s")
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    handler.name = "aequilibrae"
    handler.setLevel(logging.DEBUG)
    return handler


global_logger = logger = _setup_logger()
