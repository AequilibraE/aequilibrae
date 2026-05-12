"""
Meson isn't able to automatically find the installation location of cysignals when installed as a package so we use
the same trick sagemath [1] does resolve it ourselves.

[1] https://github.com/sagemath/sage/blob/fe5b13e61edc48c51ab91d457693d793c9c5b192/src/meson.build#L29-L45
"""

from os.path import dirname, relpath

import cysignals

print(relpath(dirname(cysignals.__file__)))
