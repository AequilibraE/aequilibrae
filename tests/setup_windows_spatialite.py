import urllib.request
import platform
import zipfile
import os
import yaml
from os.path import join, dirname
from os import walk

pth = 'https://github.com/AequilibraE/aequilibrae/releases/download/V0.6.0.post1/mod_spatialite-NG-win-amd64.zip'

outfolder = dirname(os.path.abspath(__file__))

dest_path = join(outfolder, "mod_spatialite-NG-win-amd64.zip")
urllib.request.urlretrieve(pth, dest_path)

fldr = join(outfolder, 'temp_data')
zipfile.ZipFile(dest_path).extractall(fldr)

file = join(dirname(outfolder), "aequilibrae/parameters.yml")
print(file)
with open(file, "r") as yml:
    parameters = yaml.load(yml, Loader=yaml.SafeLoader)

parameters['system']['spatialite_path'] = fldr

with open(file, "w") as stream:
    yaml.dump(parameters, stream, default_flow_style=False)

if 'WINDOWS' in platform.platform().upper():
    from glob import glob
    start_dir = 'C:/hostedtoolcache/windows/Python/'
    pattern = "sqlite3.dll"
    for dir, _, _ in walk(start_dir):
        print(glob(join(dir, pattern)))

        # We now set sqlite. Only needed in thge windows server in Github
    plats = {'x86': 'https://sqlite.org/2020/sqlite-dll-win32-x86-3320100.zip',
             'x64': 'https://sqlite.org/2020/sqlite-dll-win64-x64-3320100.zip'}

    outfolder = 'C:/'
    zip_path64 = join(outfolder, "sqlite-dll-win64-x64-3320100.zip")
    urllib.request.urlretrieve(plats['x64'], zip_path64)

    zip_path86 = join(outfolder, "sqlite-dll-win32-x86-3320100.zip")
    urllib.request.urlretrieve(plats['x86'], zip_path86)

    root = 'C:/hostedtoolcache/windows/Python/'
    file = 'sqlite3.dll'
    for d, subD, f in walk(root):
        if file in f:
            if 'x64' in d:
                zipfile.ZipFile(zip_path64).extractall(d)
            else:
                zipfile.ZipFile(zip_path86).extractall(d)
