from unittest import TestCase
from datetime import datetime, timedelta
import time
from random import randint
from aequilibrae.transit.date_tools import to_seconds, to_time_string, one_day_before, create_days_between
from aequilibrae.transit.date_tools import day_of_week, format_date


class TestDateTools(TestCase):
    def test_to_seconds(self):
        t = to_seconds("18:45:36")
        self.assertEqual(t, 18 * 3600 + 45 * 60 + 36, "to_seconds failed to return right values")

        t = to_seconds("1:5:6")
        self.assertEqual(t, 1 * 3600 + 5 * 60 + 6, "to_seconds failed to return right values")

        with self.assertRaises(ValueError):
            t = to_seconds("")

        t = to_seconds("1 day, 1:5:6")
        self.assertEqual(t, 25 * 3600 + 5 * 60 + 6, "to_seconds failed to return right values")

        t = to_seconds("2 day, 1:5:6")
        self.assertEqual(t, 49 * 3600 + 5 * 60 + 6, "to_seconds failed to return right values")

        t = to_seconds("-1 day, 1:5:6")
        self.assertEqual(t, -23 * 3600 + 5 * 60 + 6, "to_seconds failed to return right values")

    def test_to_time_string(self):
        with self.assertRaises(TypeError):
            _ = to_time_string("18:45:36")

        t = to_time_string(None)
        self.assertEqual(t, None, "to_time_string failed to return None")

        t = to_time_string(86399)
        self.assertEqual(t, "23:59:59", "to_time_string failed to return right values")

        t = to_time_string("86398")
        self.assertEqual(t, "23:59:58", "to_time_string failed to return right values")

        t = to_time_string(86401)
        self.assertEqual(t, "1 day, 0:00:01", "to_time_string failed to return right values")

        t = to_seconds("2 day, 1:5:6")
        self.assertEqual(t, 49 * 3600 + 5 * 60 + 6, "to_seconds failed to return right values")

    def test_one_day_before(self):
        t = one_day_before("2020-01-01")
        self.assertEqual(t, "2019-12-31")

        t = one_day_before("2020-03-01")
        self.assertEqual(t, "2020-02-29")

        t = one_day_before("2020-02-29")
        self.assertEqual(t, "2020-02-28")

        with self.assertRaises(ValueError):
            _ = one_day_before("2020-13-01")

        with self.assertRaises(TypeError):
            _ = one_day_before(123456)

    def test_create_days_between(self):
        tdy = datetime.today()
        days = randint(1, 100)
        past = tdy - timedelta(days=days)

        old = f"{past.year}-{past.month:02d}-{past.day:02d}"
        today = f"{tdy.year}-{tdy.month:02d}-{tdy.day:02d}"

        interval = create_days_between(old, today)
        self.assertEqual(len(interval), days + 1, "create_days_between returned wrong value")

        interval = create_days_between(today, old)
        self.assertEqual(len(interval), 0, "create_days_between returned wrong value")

        interval = create_days_between(today, today)
        self.assertEqual(len(interval), 1, "create_days_between returned wrong value")

    def test_day_of_week(self):
        tdy = datetime.today()
        days = randint(1, 100)
        past = tdy - timedelta(days=days)

        old = f"{past.year}-{past.month:02d}-{past.day:02d}"
        today = f"{tdy.year}-{tdy.month:02d}-{tdy.day:02d}"

        for date in [today, old]:
            should = time.strptime(date, "%Y-%m-%d").tm_wday
            actually = day_of_week(date)
            self.assertEqual(should, actually, "day_of_week returned the wrong value")

        with self.assertRaises(TypeError):
            day_of_week(100)

    def test_format_date(self):
        for i in range(100):
            year = randint(1000, 2020)
            month = randint(1, 12)
            day = randint(1, 28)
            date = f"{year}{month:02d}{day:02d}"
            formatted = format_date(date)
            self.assertEqual(formatted, f"{year}-{month:02d}-{day:02d}")
