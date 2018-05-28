import tempfile
import glob
import os


def clean_after_tests():
    p = tempfile.gettempdir() + '/aequilibrae_*'
    for f in glob.glob(p):
        try:
            os.unlink(f)
        except:
            pass

clean_after_tests()