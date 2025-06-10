import platform
import urllib.request
import zipfile
from os import listdir
from os.path import join
from pathlib import Path
from tempfile import gettempdir

pth = "https://github.com/AequilibraE/aequilibrae/releases/download/V0.6.0.post1/mod_spatialite-NG-win-amd64.zip"

outfolder = gettempdir()

dest_path = join(outfolder, "mod_spatialite-NG-win-amd64.zip")
urllib.request.urlretrieve(pth, dest_path)

fldr = join(outfolder, "temp_data")
zipfile.ZipFile(dest_path).extractall(fldr)

if "WINDOWS" in platform.platform().upper():
    # We now set sqlite. Only needed in the windows server in Github
    plats = {
        "x86": "https://sqlite.org/2020/sqlite-dll-win32-x86-3320100.zip",
        "x64": "https://sqlite.org/2020/sqlite-dll-win64-x64-3320100.zip",
    }

    outfolder = "C:/"
    zip_path64 = join(outfolder, "sqlite-dll-win64-x64-3320100.zip")
    urllib.request.urlretrieve(plats["x64"], zip_path64)

    zip_path86 = join(outfolder, "sqlite-dll-win32-x86-3320100.zip")
    urllib.request.urlretrieve(plats["x86"], zip_path86)

    root = "C:/hostedtoolcache/windows/Python/"
    file = "sqlite3.dll"
    for d in listdir(root):
        target_dir = Path(root) / d / "x64" / "DLLs"
        if (target_dir / file).exists():
            zipfile.ZipFile(zip_path64).extractall(target_dir)
            print(f"Replaces {target_dir / file}")
