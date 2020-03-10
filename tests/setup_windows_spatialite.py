import urllib.request
import zipfile
import os
import yaml
from os.path import join, dirname

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
