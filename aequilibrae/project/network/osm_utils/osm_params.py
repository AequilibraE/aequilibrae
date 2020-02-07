import requests

overpass_endpoint = "http://overpass-api.de/api"

user_agent = "AequilibraE (https://github.com/aequilibrae/aequilibrae-GUI)"
referer = "AequilibraE (https://github.com/aequilibrae/aequilibrae-GUI)"
accept_language = "en"

http_headers = requests.utils.default_headers()
http_headers.update(
    {"User-Agent": user_agent, "referer": referer, "Accept-Language": accept_language, "format": "json"}
)

nominatim_endpoint = "https://nominatim.openstreetmap.org/"

max_attempts = 5

timeout = 180
memory = None
max_query_area_size = 50 * 1000 * 50 * 1000
