import urllib.request
import zipfile
from os.path import join, dirname
import os

pth = 'https://github.com/AequilibraE/aequilibrae/releases/download/V0.6.0.post1/mod_spatialite-NG-win-amd64.zip'


outfolder = dirname(dirname(os.path.abspath(__file__)))
outfolder = join(outfolder, 'aequilibrae/project')

dest_path = join(outfolder, "mod_spatialite-NG-win-amd64.zip")
urllib.request.urlretrieve(pth, dest_path)

zipfile.ZipFile(dest_path).extractall(outfolder)
