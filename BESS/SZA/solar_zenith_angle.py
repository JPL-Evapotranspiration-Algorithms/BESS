"""
This module calculates solar zenith angle.

Developed by Gregory Halverson in the Jet Propulsion Laboratory Year-Round Internship Program (Columbus Technologies and Services, ANRE Tech.), in coordination with the ECOSTRESS mission and master's thesis studies at California State University, Northridge.
"""
from numpy.core.umath import radians, arccos, sin, cos, degrees

__author__ = 'Gregory Halverson'


def sza_deg_from_lat_dec_hour(latitude, solar_dec_deg, hour):
    """
    This function calculates solar zenith angle from longitude, solar declination, and solar time.
    SZA calculated by this function matching SZA provided by MOD07 to within 0.4 degrees.    
    
    :param latitude: latitude in degrees
    :param solar_dec_deg: solar declination in degrees
    :param hour: solar time in hours
    :return: solar zenith angle in degrees
    """
    # convert angles to radians
    latitude_rad = radians(latitude)
    solar_dec_rad = radians(solar_dec_deg)
    hour_angle_deg = hour * 15.0 - 180.0
    hour_angle_rad = radians(hour_angle_deg)
    sza_rad = arccos(
        sin(latitude_rad) * sin(solar_dec_rad) + cos(latitude_rad) * cos(solar_dec_rad) * cos(hour_angle_rad))
    sza_deg = degrees(sza_rad)

    return sza_deg
