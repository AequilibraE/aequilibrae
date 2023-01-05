from unittest import TestCase
from random import randint
from datetime import datetime, timedelta
from aequilibrae.transit.transit_elements import Service
from tests.aequilibrae.transit.random_word import randomword


class TestService(TestCase):
    def setUp(self) -> None:

        tdy = datetime.today()
        past = tdy - timedelta(days=randint(1, 100))

        self.data = {
            "service_id": randomword(randint(0, 40)),
            "monday": 1,
            "tuesday": 1,
            "wednesday": 1,
            "thursday": 1,
            "friday": 1,
            "saturday": 1,
            "sunday": 1,
            "start_date": f"{past.year}{past.month:02d}{past.day:02d}",
            "end_date": f"{tdy.year}{tdy.month:02d}{tdy.day:02d}",
        }

        self.today = tdy
        self.past = past

    def test__populate(self):
        s = Service()

        s._populate(tuple(self.data.values()), list(self.data.keys()))
        for key, val in s.__dict__.items():
            if key in self.data:
                self.assertEqual(val, self.data[key], "Service population with record failed")

        time_span = (self.today - self.past).days + 1  # n + 1 intervals

        self.assertEqual(time_span, len(s.dates), "Returned the wrong dates for service")

        # Test with no weekdays available
        for key, val in self.data.items():
            if val == 1:
                self.data[key] = 0

        s = Service()
        s._populate(tuple(self.data.values()), list(self.data.keys()))
        self.assertEqual(0, len(s.dates), "Returned too many dates for service")

        self.data[randomword(randint(1, 15))] = randomword(randint(1, 20))
        s = Service()
        with self.assertRaises(KeyError):
            s._populate(tuple(self.data.values()), list(self.data.keys()))
