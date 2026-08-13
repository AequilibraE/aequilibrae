import logging
import os
import sys
from contextlib import nullcontext

from aequilibrae.utils.qgis_utils import inside_qgis

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


DEFAULT_FORMAT = "%(asctime)s;%(levelname)s ; %(message)s"


class AequilibraETQDMStreamHandler(logging.StreamHandler):
    """
    Logging handler that writes messages using tqdm.write to avoid interfering with progress bars.

    :Arguments:
        **args** (:obj:`*args`): Variable length argument list passed to the parent logging.StreamHandler.

        **tqdm_class** (:obj:`type`, optional): The tqdm class to use for writing. Defaults to the imported tqdm.

        **kwargs** (:obj:`**kwargs`): Arbitrary keyword arguments passed to the parent logging.StreamHandler.
    """

    def __init__(self, *args, tqdm_class=tqdm, show_progress: bool | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.__show_progress = (
            show_progress if show_progress is not None else os.environ.get("AEQ_SHOW_PROGRESS", "TRUE") == "TRUE"
        )
        self.tqdm_class = tqdm_class

        if self.__show_progress and self.tqdm_class is None:
            raise ValueError(
                "show_progress (or AEQ_SHOW_PROGRESS) was True but the provided tqdm is None or tqdm failed to import"
            )

    def emit(self, record):
        """Emits a record.

        Args:
            record (logging.LogRecord): The log record to emit.
        """
        if not self.__show_progress:
            super().emit(record)
            return

        try:
            msg = self.format(record)
            self.tqdm_class.write(msg, file=self.stream, end=self.terminator)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


AequilibraEStreamHandler = (
    AequilibraETQDMStreamHandler if tqdm is not None and not inside_qgis else logging.StreamHandler
)


def basic_config(level: int = logging.INFO, stream=sys.stdout, format: str = DEFAULT_FORMAT) -> logging.Handler | None:
    """
    Configures the root logger for AequilibraE.

    Sets up a specific handler that works well with tqdm progress bars if available
    and not running inside QGIS. It disables propagation to avoid interference from
    root loggers.

    :Arguments:
        **level** (:obj:`int`, optional): The logging level to set. Defaults to logging.INFO.

        **stream** (:obj:`IO`, optional): The stream to write logs to. Defaults to sys.stdout.

        **format** (:obj:`str`, optional): The log format string. Defaults to DEFAULT_FORMAT.

    :Returns:
        **handler** (:obj:`logging.Handler`): The handler attached to the 'aequilibrae' logger. If the logger already
            has a handler writing to standard out or standard error, returns None.
    """
    logger = logging.getLogger("aequilibrae")

    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and (
            handler.stream == sys.stderr or handler.stream == sys.stdout
        ):
            return  # if something else has already been configured then we don't want to do anything

    # We disable log propagation up the chain because we don't want the handlers installed on the root logger messing
    # with our progress bars.
    logger.propagate = False
    logger.setLevel(level)

    handler = AequilibraEStreamHandler(stream)
    handler.setFormatter(logging.Formatter(format))

    logger.addHandler(handler)

    return handler


def debug_bridge(logger: logging.Logger):
    """
    Context manager yielding a :obj:`Bridge` when ``logger`` is enabled for DEBUG, or ``None`` otherwise.

    The Bridge surfaces DEBUG-level messages emitted from C++/Cython (e.g. which priority queue the
    path finding is using). Since a Bridge runs a monitoring thread whose teardown can take up to its
    polling interval, it is only spun up when those messages would actually be logged.

    :Arguments:
        **logger** (:obj:`logging.Logger`): The logger the Bridge should dispatch to.

    :Returns:
        **context manager**: yields a :obj:`Bridge` or ``None``.
    """
    from aequilibrae.utils.cython.bridge import Bridge

    return Bridge(logger) if logger.isEnabledFor(logging.DEBUG) else nullcontext()


def default_log_file_config(handler: logging.Handler, format: str = DEFAULT_FORMAT):
    """
    Attaches a file handler to the AequilibraE logger with default formatting.

    :Arguments:
        **handler** (:obj:`logging.Handler`): The logging handler (usually a FileHandler) to attach.

        **format** (:obj:`str`, optional): The log format string to use for this handler. Defaults to DEFAULT_FORMAT.
    """
    logger = logging.getLogger("aequilibrae")

    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)
    # We do not want to set a level on this handler because that should be controlled by the logger, and optionally set
    # by the user
