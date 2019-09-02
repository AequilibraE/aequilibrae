import os

with open(os.path.join(os.path.dirname(__file__), "parameters.pxi"), "r") as a:
    for i in a.readlines():
        if "VERSION" in i:
            version = i[10:]
        if "MINOR_VRSN" in i:
            minor_version = i[13:]
        if "release_name" in i:
            release_name = i[16:-1]
        if "binary" in i:
            binary_version = i[17:-1]

release_version = str(version) + "." + str(minor_version)
