import time
from datetime import timedelta, datetime
from typing import Union, Any, Optional

day_sequence = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def to_seconds(value: Optional[str]) -> Union[int, Any]:
    if value is None or not isinstance(value, str):
        return value

    day = 0.0
    if "day" in value:
        day = float(value[: value.find(" d")]) * 24 * 3600
        value = value[value.find(", ") + 2 :]

    # convert the string to second of day
    split = value.split(":")
    if len(split) != 3:
        raise ValueError(f"Time {value} does not have an appropriate format")

    hours, minutes, seconds = map(int, split)
    return hours * 3600 + minutes * 60 + seconds + day


def to_time_string(value: Union[int, None]) -> Union[str, None]:
    # it is ok to pass None
    if value is None:
        return value

    if isinstance(value, str):
        if value.isdigit():
            value = int(value)

    if not isinstance(value, int):
        raise TypeError(f"Time {value} is not integer, but it should")

    return str(timedelta(seconds=value))


def one_day_before(date_object):
    start_seconds = date_object.timestamp() + 12 * 60 * 60
    return time.strftime("%Y-%m-%d", time.localtime(start_seconds - 24 * 60 * 60))


def create_days_between(range_start_date, range_end_date):
    numdays = (range_end_date - range_start_date).days + 1
    return [(range_start_date + timedelta(days=x)).strftime("%Y-%m-%d") for x in range(numdays)]


def day_of_week(date: str):
    return datetime.fromisoformat(date).weekday()


def format_date(date: str) -> str:
    return "-".join([date[:4], date[4:6], date[6:]])
