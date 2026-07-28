from aequilibrae.project.network.osm.place_getter import placegetter


def test_placegetter(mock_nominatim_api):
    _ = mock_nominatim_api
    place, report = placegetter("Vatican City")
    place = [round(x, 1) for x in place]
    assert place == [12.4, 41.9, 12.5, 41.9], "Returned the wrong boundingbox for Vatican City"

    place, report = placegetter("Just a random place with no bear in reality")
    assert place is None, "Returned a bounding box for a place that does not exist"
