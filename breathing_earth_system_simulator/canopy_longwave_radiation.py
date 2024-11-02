import numpy as np


def canopy_longwave_radiation(
        LAI: np.ndarray,
        SZA: np.ndarray,
        Ts_K: np.ndarray,
        Tf_K: np.ndarray,
        Ta_K: np.ndarray,
        epsa: np.ndarray,
        epsf: float,
        epss: float,
        ALW_min: float = None,
        intermediate_min: float = None,
        intermediate_max: float = None):
    """
    =============================================================================

    Module     : Canopy longwave radiation transfer
    Input      : leaf area index (LAI) [-],
               : extinction coefficient for longwave radiation (kd) [m-1],
               : extinction coefficient for beam radiation (kb) [m-1],
               : air temperature (Ta) [K],
               : soil temperature (Ts) [K],
               : foliage temperature (Tf) [K],
               : clear-sky emissivity (epsa) [-],
               : soil emissivity (epss) [-],
               : foliage emissivity (epsf) [-].
    Output     : total absorbed LW by sunlit leaves (Q_LSun),
               : total absorbed LW by shade leaves (Q_LSh).
    References : Wang, Y., Law, R. M., Davies, H. L., McGregor, J. L., & Abramowitz, G. (2006).
                 The CSIRO Atmosphere Biosphere Land Exchange (CABLE) model for use in climate models and as an offline model.


    Conversion from MatLab by Robert Freepartner, JPL/Raytheon/JaDa Systems
    March 2020

    =============================================================================
    """
    # SZA[SZA > 89.0] = 89.0
    SZA = np.clip(SZA, None, 89)
    kb = 0.5 / np.cos(SZA * np.pi / 180.0)  # Table A1 in Ryu et al 2011
    kd = 0.78  # Table A1 in Ryu et al 2011

    # Stefan_Boltzmann_constant
    sigma = 5.670373e-8  # [W m-2 K-4] (Wiki)

    # Long wave radiation flux densities from air, soil and leaf
    La = np.clip(epsa * sigma * Ta_K ** 4, 0, None)
    Ls = np.clip(epss * sigma * Ts_K ** 4, 0, None)
    Lf = np.clip(epsf * sigma * Tf_K ** 4, 0, None)

    # For simplicity
    kd_LAI = kd * LAI

    # Absorbed longwave radiation by sunlit leaves
    ALW_sunlit = np.clip(
        np.clip(Ls - Lf, intermediate_min, None) * kd * (np.exp(-kd_LAI) - np.exp(-kb * LAI)) / (
                kd - kb) + kd * np.clip(La - Lf,
                                        intermediate_min, intermediate_max) * (
                1.0 - np.exp(-(kb + kd) * LAI)), ALW_min, None) / (kd + kb)  # Eq. (44)

    # Absorbed longwave radiation by shaded leaves
    ALW_shaded = np.clip(
        (1.0 - np.exp(-kd_LAI)) * np.clip(Ls + La - 2 * Lf, intermediate_min, intermediate_max) - ALW_sunlit,
        ALW_min,
        None)  # Eq. (45)

    # Absorbed longwave radiation by soil
    ALW_soil = np.clip((1.0 - np.exp(-kd_LAI)) * Lf + np.exp(-kd_LAI) * La, ALW_min, None)  # Eq. (41)

    return ALW_sunlit, ALW_shaded, ALW_soil, Ls, La, Lf
