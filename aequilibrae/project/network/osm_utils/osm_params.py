import requests
from aequilibrae.parameters import Parameters

par = Parameters().parameters["osm"]
accept_language = par["accept_language"]
memory = 0

user_agent = "AequilibraE (https://github.com/aequilibrae/aequilibrae-GUI)"
referer = "AequilibraE (https://github.com/aequilibrae/aequilibrae-GUI)"

http_headers = requests.utils.default_headers()
http_headers.update(
    {"User-Agent": user_agent, "referer": referer, "Accept-Language": accept_language, "format": "json"}
)
