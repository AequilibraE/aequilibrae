from pathlib import Path


class Log:
    """API entry point to the log file contents

    .. code-block:: python

        >>> project = Project.new(project_path)

        >>> log = project.log()

        # We get all entries for the log file
        >>> entries = log.contents()

        # Or clear everything (NO UN-DOs)
        >>> log.clear()

        >>> project.close()
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
