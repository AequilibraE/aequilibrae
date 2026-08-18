import hashlib
import os
import tempfile
from pathlib import Path


class ProjectPathLock:
    """Cross-process exclusive lock for one canonical project path."""

    def __init__(self, project_path):
        self.path = Path(project_path).expanduser().resolve(strict=False)
        identity = os.path.normcase(str(self.path))
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        directory = Path(tempfile.gettempdir()) / "aequilibrae-project-locks"
        directory.mkdir(parents=True, exist_ok=True)
        self.lock_path = directory / f"{digest}.lock"
        self._file = None

    def acquire(self):
        if self._file is not None:
            return
        file = self.lock_path.open("a+")
        try:
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                import msvcrt

                file.seek(0)
                if file.tell() == 0:
                    file.write("0")
                    file.flush()
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            file.close()
            raise RuntimeError(f"project path is already open or being upgraded: {self.path}") from error
        self._file = file

    def release(self):
        if self._file is None:
            return
        try:
            if os.name == "nt":  # pragma: no cover
                import msvcrt

                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None
