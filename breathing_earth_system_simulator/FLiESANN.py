"""
Forest Light Environmental Simulator (FLiES)
Artificial Neural Network Implementation
for the Breathing Earth Systems Simulator (BESS)
"""

import logging
import warnings
from os.path import join, abspath, dirname
from time import process_time

import numpy as np
import pandas as pd
import rasters as rt

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # from keras.engine.saving import load_model
    from keras.models import load_model

__author__ = "Gregory Halverson, Robert Freepartner"


DEFAULT_WORKING_DIRECTORY = "."
DEFAULT_FLIES_INTERMEDIATE = "FLiES_intermediate"

DEFAULT_MODEL_FILENAME = join(abspath(dirname(__file__)), "FLiESANN.h5")
SPLIT_ATYPES_CTYPES = True

DEFAULT_PREVIEW_QUALITY = 20
DEFAULT_INCLUDE_PREVIEW = True
DEFAULT_RESAMPLING = "cubic"
DEFAULT_SAVE_INTERMEDIATE = True
DEFAULT_SHOW_DISTRIBUTION = True
DEFAULT_DYNAMIC_ATYPE_CTYPE = False

logger = logging.getLogger(__name__)


def determine_atype(KG_climate: np.ndarray, COT: np.ndarray, dynamic: bool = True) -> np.ndarray:
    atype = np.full(KG_climate.shape, 1, dtype=np.uint16)

    if dynamic:
        atype = np.where((COT == 0) & ((KG_climate == 5) | (KG_climate == 6)), 1, atype)
        atype = np.where((COT == 0) & ((KG_climate == 3) | (KG_climate == 4)), 2, atype)
        atype = np.where((COT == 0) & (KG_climate == 1), 4, atype)
        atype = np.where((COT == 0) & (KG_climate == 2), 5, atype)
        atype = np.where((COT > 0) & ((KG_climate == 5) | (KG_climate == 6)), 1, atype)
        atype = np.where((COT > 0) & ((KG_climate == 3) | (KG_climate == 4)), 2, atype)
        atype = np.where((COT > 0) & (KG_climate == 2), 5, atype)
        atype = np.where((COT > 0) & (KG_climate == 1), 4, atype)

    return atype


def determine_ctype(KG_climate: np.ndarray, COT: np.ndarray, dynamic: bool = True) -> np.ndarray:
    ctype = np.full(KG_climate.shape, 0, dtype=np.uint16)

    if dynamic:
        ctype = np.where((COT == 0) & ((KG_climate == 5) | (KG_climate == 6)), 0, ctype)
        ctype = np.where((COT == 0) & ((KG_climate == 3) | (KG_climate == 4)), 0, ctype)
        ctype = np.where((COT == 0) & (KG_climate == 1), 0, ctype)
        ctype = np.where((COT == 0) & (KG_climate == 2), 0, ctype)
        ctype = np.where((COT > 0) & ((KG_climate == 5) | (KG_climate == 6)), 1, ctype)
        ctype = np.where((COT > 0) & ((KG_climate == 3) | (KG_climate == 4)), 1, ctype)
        ctype = np.where((COT > 0) & (KG_climate == 2), 1, ctype)
        ctype = np.where((COT > 0) & (KG_climate == 1), 3, ctype)

    return ctype


def process_FLiES_ANN(
        atype: np.ndarray,
        ctype: np.ndarray,
        COT: np.ndarray,
        AOT: np.ndarray,
        vapor_gccm: np.ndarray,
        ozone_cm: np.ndarray,
        albedo: np.ndarray,
        elevation_km: np.ndarray,
        SZA: np.ndarray,
        ANN_model=None,
        split_atypes_ctypes=SPLIT_ATYPES_CTYPES) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    if ANN_model is None:
        ANN_model = load_model(DEFAULT_MODEL_FILENAME)

    ctype_flat = np.array(ctype).flatten()
    atype_flat = np.array(atype).flatten()
    COT_flat = np.array(COT).flatten()
    AOT_flat = np.array(AOT).flatten()
    vapor_gccm_flat = np.array(vapor_gccm).flatten()
    ozone_cm_flat = np.array(ozone_cm).flatten()
    albedo_flat = np.array(albedo).flatten()
    elevation_km_flat = np.array(elevation_km).flatten()
    SZA_flat = np.array(SZA).flatten()

    inputs = pd.DataFrame({
        "ctype": ctype_flat,
        "atype": atype_flat,
        "COT": COT_flat,
        "AOT": AOT_flat,
        "vapor_gccm": vapor_gccm_flat,
        "ozone_cm": ozone_cm_flat,
        "albedo": albedo_flat,
        "elevation_km": elevation_km_flat,
        "SZA": SZA_flat
    })

    if split_atypes_ctypes:
        inputs["ctype0"] = np.float32(inputs.ctype == 0)
        inputs["ctype1"] = np.float32(inputs.ctype == 1)
        inputs["ctype3"] = np.float32(inputs.ctype == 3)
        inputs["atype1"] = np.float32(inputs.ctype == 1)
        inputs["atype2"] = np.float32(inputs.ctype == 2)
        inputs["atype4"] = np.float32(inputs.ctype == 4)
        inputs["atype5"] = np.float32(inputs.ctype == 5)

        inputs = inputs[
            ["ctype0", "ctype1", "ctype3", "atype1", "atype2", "atype4", "atype5", "COT", "AOT", "vapor_gccm",
            "ozone_cm", "albedo", "elevation_km", "SZA"]]

    outputs = ANN_model.predict(inputs)
    shape = COT.shape
    tm = np.clip(outputs[:, 0].reshape(shape), 0, 1).astype(np.float32)
    puv = np.clip(outputs[:, 1].reshape(shape), 0, 1).astype(np.float32)
    pvis = np.clip(outputs[:, 2].reshape(shape), 0, 1).astype(np.float32)
    pnir = np.clip(outputs[:, 3].reshape(shape), 0, 1).astype(np.float32)
    fduv = np.clip(outputs[:, 4].reshape(shape), 0, 1).astype(np.float32)
    fdvis = np.clip(outputs[:, 5].reshape(shape), 0, 1).astype(np.float32)
    fdnir = np.clip(outputs[:, 6].reshape(shape), 0, 1).astype(np.float32)

    return tm, puv, pvis, pnir, fduv, fdvis, fdnir


def process_FLiES(
        doy: np.ndarray,
        albedo: np.ndarray,
        COT: np.ndarray = None,
        AOT: np.ndarray = None,
        vapor_gccm: np.ndarray = None,
        ozone_cm: np.ndarray = None,
        elevation_km: np.ndarray = None,
        SZA: np.ndarray = None,
        KG_climate: np.ndarray = None):
    COT = np.clip(COT, 0, None)
    COT = rt.where(COT < 0.001, 0, COT)
    atype = determine_atype(KG_climate, COT)
    ctype = determine_ctype(KG_climate, COT)

    prediction_start_time = process_time()
    tm, puv, pvis, pnir, fduv, fdvis, fdnir = process_FLiES_ANN(
        atype=atype,
        ctype=ctype,
        COT=COT,
        AOT=AOT,
        vapor_gccm=vapor_gccm,
        ozone_cm=ozone_cm,
        albedo=albedo,
        elevation_km=elevation_km,
        SZA=SZA
    )

    prediction_end_time = process_time()
    prediction_duration = prediction_end_time - prediction_start_time

    ##  Correction for diffuse PAR
    COT = rt.where(COT == 0.0, np.nan, COT)
    COT = rt.where(np.isfinite(COT), COT, np.nan)
    x = np.log(COT)
    p1 = 0.05088
    p2 = 0.04909
    p3 = 0.5017
    corr = np.array(p1 * x * x + p2 * x + p3)
    corr[np.logical_or(np.isnan(corr), corr > 1.0)] = 1.0
    fdvis = fdvis * corr * 0.915

    ## Radiation components
    dr = 1.0 + 0.033 * np.cos(2 * np.pi / 365.0 * doy)
    Ra = 1333.6 * dr * np.cos(SZA * np.pi / 180.0)
    Ra = rt.where(SZA > 90.0, 0, Ra)
    Rg = Ra * tm
    UV = Rg * puv
    VIS = Rg * pvis
    NIR = Rg * pnir
    # UVdiff = SSR.UV * fduv
    VISdiff = VIS * fdvis
    NIRdiff = NIR * fdnir
    # UVdir = SSR.UV - UVdiff
    VISdir = VIS - VISdiff
    NIRdir = NIR - NIRdiff

    return Ra, Rg, UV, VIS, NIR, VISdiff, NIRdiff, VISdir, NIRdir, tm, puv, pvis, pnir, fduv, fdvis, fdnir
