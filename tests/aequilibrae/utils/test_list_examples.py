from aequilibrae.examples import list_examples


def test_list_examples():
    data = list_examples()
    for x in ["coquimbo", "nauru", "sioux_falls"]:
        assert x in data
