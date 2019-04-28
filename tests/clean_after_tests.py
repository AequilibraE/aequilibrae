import glob
import os
import tempfile


def clean_after_tests():
    p = tempfile.gettempdir() + "/aequilibrae_*"
    for f in glob.glob(p):
        try:
            os.unlink(f)
        except Exception as err:
            print(err.__str__())


clean_after_tests()
