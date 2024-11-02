"""
This module calculates solar azimuth.

Developed by Gregory Halverson at the Jet Propulsion Laboratory
"""
import warnings

from numpy import arcsin, cos
from numpy import degrees
from numpy import radians
from numpy import sin

__author__ = 'Gregory Halverson'


def calculate_solar_azimuth(solar_dec_deg, sza_deg, hour):
    """
    This function calculates solar azimuth.
    :param latitude: latitude in degrees
    :param solar_dec_deg: solar declination in degrees
    :param sza_deg: solar zenith angle in degrees
    :return: solar azimuth in degrees
    """
    # convert angles to radians
    warnings.filterwarnings('ignore')
    solar_dec_rad = radians(solar_dec_deg)
    sza_rad = radians(sza_deg)
    hour_angle_deg = hour * 15.0 - 180.0
    hour_angle_rad = radians(hour_angle_deg)
    # https: // en.wikipedia.org / wiki / Solar_azimuth_angle
    # solar_azimuth_rad = arccos((sin(solar_dec_rad) - cos(sza_rad) * sin(latitude_rad)) / (sin(sza_rad) * cos(latitude_rad)))
    solar_azimuth_rad = arcsin(-1.0 * sin(hour_angle_rad) * cos(solar_dec_rad) / sin(sza_rad))
    solar_azimuth_deg = degrees(solar_azimuth_rad)
    warnings.resetwarnings()

    return solar_azimuth_deg
