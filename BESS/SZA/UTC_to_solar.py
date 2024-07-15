from datetime import datetime, timedelta
from typing import Union

import numpy as np


def UTC_offset_hours(lon: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculates the offset in hours from UTC based on the given longitude.

    Args:
        lon (Union[float, np.ndarray]): The longitude in degrees.

    Returns:
        Union[float, np.ndarray]: The calculated offset in hours from UTC.
    """
    # Convert longitude to radians and calculate the offset in hours from UTC
    return np.radians(lon) / np.pi * 12


def UTC_to_solar(time_UTC: datetime, lon: float) -> datetime:
    """
    Calculates the solar time at the given longitude based on the given UTC time.

    Args:
        time_UTC (datetime.datetime): The UTC time to calculate the solar time for.
        lon (float): The longitude in degrees.

    Returns:
        datetime.datetime: The calculated solar time.
    """
    # Calculate the solar time at the given longitude
    return time_UTC + timedelta(hours=(np.radians(lon) / np.pi * 12))


def day_of_year(time_UTC: datetime, lon: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculates the day of year based on the given UTC time and longitude.

    Args:
        time_UTC (datetime.datetime): The UTC time to calculate the day of year for.
        lon (Union[float, np.ndarray]): The longitude in degrees.

    Returns:
        Union[float, np.ndarray]: The calculated day of year.
    """
    # Calculate the day of year at the given longitude
    doy_UTC = time_UTC.timetuple().tm_yday
    hour_UTC = time_UTC.hour + time_UTC.minute / 60 + time_UTC.second / 3600
    offset = UTC_offset_hours(lon)
    hour_of_day = hour_UTC + offset
    doy = doy_UTC
    # Adjust the day of year if the hour of day is outside the range [0, 24]
    doy = np.where(hour_of_day < 0, doy - 1, doy)
    doy = np.where(hour_of_day > 24, doy + 1, doy)

    return doy


def hour_of_day(time_UTC: datetime, lon: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculates the hour of day based on the given UTC time and longitude.

    Args:
        time_UTC (datetime.datetime): The UTC time to calculate the hour of day for.
        lon (Union[float, np.ndarray]): The longitude in degrees.

    Returns:
        Union[float, np.ndarray]: The calculated hour of day.
    """
    # Calculate the hour of day at the given longitude
    hour_UTC = time_UTC.hour + time_UTC.minute / 60 + time_UTC.second / 3600
    offset = UTC_offset_hours(lon)
    hour_of_day = hour_UTC + offset
    # Adjust the hour of day if it is outside the range [0, 24]
    hour_of_day = np.where(hour_of_day < 0, hour_of_day + 24, hour_of_day)
    hour_of_day = np.where(hour_of_day > 24, hour_of_day - 24, hour_of_day)

    return hour_of_day
