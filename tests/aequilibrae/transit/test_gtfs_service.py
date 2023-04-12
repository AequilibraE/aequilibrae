import pytest
from random import randint
from datetime import datetime, timedelta
from aequilibrae.transit.transit_elements import Service
from tests.aequilibrae.transit.random_word import randomword


class TestService:
    @pytest.fixture
    def today(self):
        tdy = datetime.today()
        today = tdy
        return today

    @pytest.fixture
    def past(self, today):
        past = today - timedelta(days=randint(1, 100))
        past = past
        return past

    @pytest.fixture
    def data_dict(self, past, today):
        return {
            "service_id": randomword(randint(0, 40)),
            "monday": 1,
            "tuesday": 1,
            "wednesday": 1,
            "thursday": 1,
            "friday": 1,
            "saturday": 1,
            "sunday": 1,
            "start_date": past,
            "end_date": today,
        }

    def test__populate(self, data_dict, today, past):
        s = Service()

        s._populate(tuple(data_dict.values()), list(data_dict.keys()))
        for key, val in s.__dict__.items():
            if key in data_dict:
                assert val == data_dict[key], "Service population with record failed"

        time_span = (today - past).days + 1  # n + 1 intervals

        assert time_span == len(s.dates), "Returned the wrong dates for service"

        # Test with no weekdays available
        for key, val in data_dict.items():
            if val == 1:
                data_dict[key] = 0

        s = Service()
        s._populate(tuple(data_dict.values()), list(data_dict.keys()))
        assert 0 == len(s.dates), "Returned too many dates for service"

        data_dict[randomword(randint(1, 15))] = randomword(randint(1, 20))
        s = Service()
        with pytest.raises(KeyError):
            s._populate(tuple(data_dict.values()), list(data_dict.keys()))
