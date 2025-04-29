import logging
from pathlib import Path


class Log:
    """API entry point to the log file contents

    .. code-block:: python

        >>> project = Project()
        >>> project.new(project_path)

        >>> log = project.log()

        # We get all entries for the log file
        >>> entries = log.contents()

        # Or clear everything (NO UN-DOs)
        >>> log.clear()
    """

    def __init__(self, project_base_path: Path):
        self.log_file_path = project_base_path / "aequilibrae.log"

    def contents(self) -> list:
        """Returns contents of log file

        :Returns:
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
    logger = logging.getLogger("aequilibrae")
    logger.setLevel(logging.DEBUG)
    return logger


def get_log_handler(log_file: Path, ensure_file_exists=True):
    """Return a log handler that writes to the given log_file"""
    if log_file.exists() and not log_file.is_file():
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
