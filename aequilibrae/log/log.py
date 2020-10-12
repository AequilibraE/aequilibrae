from os.path import join


class Log:
    """API entry point to the log file contents

    ::

        from aequilibrae import Project

        p = Project()
        p.open('path/to/project/folder')

        log = p.log()

        # We get all entries for the log file
        entries = log.contents()

        # Or clear everything (NO UN-DOs)
        log.clear()
    """

    def __init__(self, project_base_path: str):
        self.log_file_path = join(project_base_path, "aequilibrae.log")

    def contents(self) -> list:
        """Returns contents of log file

        Return:
            *log_contents* (:obj:`list`): List with all entries in the log file
        """

        with open(self.log_file_path, 'r') as file:
            return [x.strip() for x in file.readlines()]

    def clear(self):
        """Clears the log file. Use it wisely"""
        with open(self.log_file_path, 'w') as _:
            pass
