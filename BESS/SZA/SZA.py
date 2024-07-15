import numpy as np


from .UTC_to_solar import *
from .daylight_hours import *
from .solar_zenith_angle import *


def calculate_SZA_from_doy_and_hour(lat: Union[float, np.ndarray], lon: Union[float, np.ndarray], doy: Union[float, np.ndarray], hour: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculates the solar zenith angle (SZA) in degrees based on the given UTC time, latitude, longitude, day of year, and hour of day.

    Args:
        lat (Union[float, np.ndarray]): The latitude in degrees.
        lon (Union[float, np.ndarray]): The longitude in degrees.
        doy (Union[float, np.ndarray]): The day of year.
        hour (Union[float, np.ndarray]): The hour of the day.

    Returns:
        Union[float, np.ndarray]: The calculated solar zenith angle in degrees.
    """
    day_angle_rad = day_angle_rad_from_doy(doy)
    solar_dec_deg = solar_dec_deg_from_day_angle_rad(day_angle_rad)
    SZA = sza_deg_from_lat_dec_hour(lat, solar_dec_deg, hour)

    return SZA

def calculate_SZA_from_datetime(time_UTC: datetime, lat: float, lon: float):
    """
    Calculates the solar zenith angle (SZA) in degrees based on the given UTC time, latitude, and longitude.

    Args:
        time_UTC (datetime.datetime): The UTC time to calculate the SZA for.
        lat (float): The latitude in degrees.
        lon (float): The longitude in degrees.

    Returns:
        float: The calculated solar zenith angle in degrees.
    """
    # Calculate the day of year based on the UTC time and longitude
    doy = day_of_year(time_UTC, lon)
    # Calculate the hour of the day based on the UTC time and longitude
    hour = hour_of_day(time_UTC, lon)
    # Calculate the solar zenith angle in degrees based on the latitude, solar declination angle, and hour of the day
    SZA = calculate_SZA_from_doy_and_hour(lat, lon, doy, hour)

    # Return the calculated solar zenith angle
    return SZA