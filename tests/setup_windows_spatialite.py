import urllib.request
import zipfile
from os.path import join
import os

pth = 'https://github.com/AequilibraE/aequilibrae/releases/download/V0.6.0.post1/mod_spatialite-NG-win-amd64.zip'


outfolder = os.path.dirname(os.path.abspath(__file__))
dest_path = join(outfolder, "mod_spatialite-NG-win-amd64.zip")
urllib.request.urlretrieve(pth, dest_path)

zipfile.ZipFile(dest_path).extractall(outfolder)

bin_folder = join(outfolder, 'mod_spatialite-NG-win-amd64')
print(bin_folder)

import sqlite3

a = sqlite3.connect(join(outfolder, 'trash.db'))
a.enable_load_extension(True)
a.load_extension("mod_spatialite")

