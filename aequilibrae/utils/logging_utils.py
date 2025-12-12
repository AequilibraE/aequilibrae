import logging
import sys

from aequilibrae.utils.qgis_utils import inside_qgis

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


class TQDMStreamHandler(logging.StreamHandler):
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


AequilibraEStreamHandler = TQDMStreamHandler if tqdm is not None and not inside_qgis else logging.StreamHandler

# class ActiveScenarioFilter(logging.Filter):
#     pass


def basic_config(level: int = logging.INFO, stream=sys.stdout, format: str = logging.BASIC_FORMAT):
    logger = logging.getLogger("aequilibrae")
    if logger.handlers:
        return

    # We disable log propagation up the chain because we don't want the handlers installed on the root logger messing
    # with our progress bars.
    logger.propagate = False
    logger.setLevel(level)

    handler = AequilibraEStreamHandler(stream)
    handler.setFormatter(logging.Formatter(format))

    logger.addHandler(handler)
