import requests

overpass_endpoint = "http://192.168.9.108:32784/api"
# overpass_endpoint = "http://overpass-api.de/api"

user_agent = "AequilibraE (https://github.com/aequilibrae/aequilibrae-GUI)"
referer = "AequilibraE (https://github.com/aequilibrae/aequilibrae-GUI)"
accept_language = "en"

http_headers = requests.utils.default_headers()
http_headers.update(
    {"User-Agent": user_agent, "referer": referer, "Accept-Language": accept_language, "format": "json"}
)

nominatim_endpoint = "https://nominatim.openstreetmap.org/"

max_attempts = 50

timeout = 540
memory = 0
max_query_area_size = 200 * 1000 * 200 * 1000
sleeptime = 0
