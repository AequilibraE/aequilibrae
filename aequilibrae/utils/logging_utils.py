import logging
import sys

from aequilibrae.utils.qgis_utils import inside_qgis

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


DEFAULT_FORMAT = "%(asctime)s;%(levelname)s ; %(message)s"


class AequilibraETQDMStreamHandler(logging.StreamHandler):
    def __init__(self, *args, tqdm_class=tqdm, **kwargs):
        super().__init__(*args, **kwargs)
        self.tqdm_class = tqdm_class

    def emit(self, record):
        try:
            msg = self.format(record)
            self.tqdm_class.write(msg, file=self.stream, end=self.terminator)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


AequilibraEStreamHandler = AequilibraETQDMStreamHandler if tqdm is not None and not inside_qgis else logging.StreamHandler


def basic_config(level: int = logging.INFO, stream=sys.stdout, format: str = DEFAULT_FORMAT):
    logger = logging.getLogger("aequilibrae")

    # We disable log propagation up the chain because we don't want the handlers installed on the root logger messing
    # with our progress bars.
    logger.propagate = False
    logger.setLevel(level)

    handler = AequilibraEStreamHandler(stream)
    handler.setFormatter(logging.Formatter(format))

    logger.addHandler(handler)


def default_log_file_config(handler: logging.Handler, format: str = DEFAULT_FORMAT):
    logger = logging.getLogger("aequilibrae")

    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)
    # We do not want to set a level on this handler because that should be controlled by the logger, and optionally set
    # by the user
