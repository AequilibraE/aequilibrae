"""
Logging to terminal
===================

On this example we show how to make all log messages show in the terminal.
"""

# %%
## Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join
from aequilibrae.utils.create_example import create_example
from aequilibrae import logger
import logging
import sys

# %%
# We create the example project inside our temp folder
fldr = join(gettempdir(), uuid4().hex)
project = create_example(fldr)

# %%
# We the project open, we can tell the logger to direct all messages to the terminal as well
stdout_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s;%(name)s;%(levelname)s ; %(message)s")
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

# %% hidden
# From here down it is just showing was is in the log already
log = project.log()
log_data = log.contents()
project.close()

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

img = Image.new('RGB', (1000, 17 * (1 + len(log_data))), color=(0, 0, 0))
d = ImageDraw.Draw(img)
for i, txt in enumerate(log_data):
    d.text((10, 10 + i * 17), txt, fill=(0, 255, 0))

# displaying the image for the gallery
plt.imshow(img)
