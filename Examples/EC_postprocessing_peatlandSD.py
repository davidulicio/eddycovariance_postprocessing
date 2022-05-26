# -*- coding: utf-8 -*-
"""
Eddy Covariance Post-Processing Script - Peatland SD
@author: David Trejo
V1.0 (2022-03-23)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import sympy
import hesseflux as hf
import warnings
import scipy 


#%% Data reading
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('Post-Processing of eddy covariance data')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('                                        ')
"ERA5 data"
path = 'C:/Users/PC5 LECS/Desktop/ERA5_chiloe/'
ds = pd.read_csv(path+'era5_data.csv')
units_era5 = pd.read_csv(path+'metadata.csv')['units']
vars_era5 = pd.read_csv(path+'metadata.csv')['long_name']
date_era5 = pd.DatetimeIndex(ds['time']) - pd.Timedelta(hours=3)  # UTC to local time
"Eddy data"
path = 'C:/Users/PC5 LECS/Desktop/Tovi data/Inputs/SDP/'
filename = 'eddypro_peatland_full_output_2022-04-12T201213_adv.csv'
eddy_data = pd.read_csv(path+filename, skiprows=[0], low_memory=(False))
eddy_units = eddy_data.iloc[0]; eddy_data = eddy_data[1:]
bool_repeated = eddy_data.duplicated(keep='first')
eddy_data = eddy_data[~bool_repeated]  # Erase repeated data
date = pd.DatetimeIndex(eddy_data['date'].astype(str) + ' ' +
                        eddy_data['time'].astype(str))
"Biomet data"
path = 'D:/Dropbox/David/LECS/eddy/Turbera/Biomet from Raw (TOA5)/'
filename = 'biomet_20140101_20220226.csv'
biomet_data = pd.read_csv(path+filename, low_memory=(False))
biomet_units = biomet_data.iloc[0]; biomet_data = biomet_data[1:]
bool_repeated = biomet_data.duplicated(keep='first')
biomet_data = biomet_data[~bool_repeated] # Erase repeated data
date_biomet = pd.DatetimeIndex(biomet_data['TIMESTAMP_1'].astype(str)+'/'+
               biomet_data['TIMESTAMP_2'].astype(str) + '/' +
               biomet_data['TIMESTAMP_3'].astype(str) + ' ' +
               biomet_data['TIMESTAMP_4'].astype(str) + ':' +
               biomet_data['TIMESTAMP_5'].astype(str))
#%% Post-Processing constants definitions
# General constants
undef = -9999  # Non valid data value
# U-star filtering
ustarmin = 0.01  # Peatland ustar minimum 
nboot = 100  # N° of boot straps for estimating the confidence interval of u* threshold
# Configurable Gap-filling
sw_dev = 50  # Max deviation of SW_IN
ta_dev = 2.5  # Max deviation of TA
vpd_dev = 5  # Max deviation of VPD
longgap = 300  # Avoid extrapolation in gaps longer than longgap days

#%% Cut data to fit the shortest period
date_i = date[0]; date_f = date[len(date)-1]
ds = ds[np.where(date_i==date_era5)[0][0]: np.where(date_f==date_era5)[0][0]]
date_era5 = date_era5[np.where(date_i==date_era5)[0][0]: np.where(date_f==date_era5)[0][0]]
biomet_data = biomet_data[np.where(date_i==date_biomet)[0][0]: np.where(date_f==date_biomet)[0][0]]
date_biomet = date_biomet[np.where(date_i==date_biomet)[0][0]: np.where(date_f==date_biomet)[0][0]]
#%% functions
def latent_heat_vaporization(TA):
    """latent_heat_vaporization(TA)
    Latent heat of vaporization as a function of air temperature (deg C).
    Uses the formula: lmbd = (2.501 - 0.00237*Tair)10^6
    Parameters
    ----------
    TA : list or list like
        Air temperature (deg C)
    Returns
    -------
    lambda : list or list like
        Latent heat of vaporization (J kg-1)
    References
    ----------
    - Stull, B., 1988: An Introduction to Boundary Layer Meteorology (p.641)
      Kluwer Academic Publishers, Dordrecht, Netherlands
    - Foken, T, 2008: Micrometeorology. Springer, Berlin, Germany.
    """
    k1   = 2.501
    k2   = 0.00237
    lmbd = ( k1 - k2 * TA ) * 1e+06
    return(lmbd)

def LE_to_ET(LE, TA):
    """LE_to_ET(LE, TA)
    Convert LE (W m-2) to ET (kg m-2 s-1, aka mm s-1).
    Parameters
    ----------
    LE : list or list like
        Latent Energy (W m-2)
    TA : list or list like
        Air temperature (deg C)
    Returns
    -------
    ET : list or list like
        Evapotranspiration (kg m-2 s-1, aka mm s-1)"""
    lmbd = latent_heat_vaporization(TA)
    ET   = LE/lmbd
    return(ET)

def var_reading(varname, data, boolean):
    """
    Reads variable from a dataframe and turns non valid data into NaN.

    Parameters
    ----------
    varname : str
        Column name of the variable.
    data : DataFrame
        Pandas DataFrame that includes the variable to read.
    boolean : Boolean
        True if the variable is convertible into a float.

    Returns
    -------
    var : array
        Data of the variable desired.

    """
    if boolean:
        var = np.float64(data[varname])
        var[var<-9998] = np.nan
    else:
        var = np.data[varname]
    return var

def dependencies_qc(variable, min_val, max_val):
    """
    Returns a boolean array with the a

    Parameters
    ----------
    variable : TYPE
        DESCRIPTION.
    min_val : TYPE
        DESCRIPTION.
    max_val : TYPE
        DESCRIPTION.

    Returns
    -------
    valid : TYPE
        DESCRIPTION.

    """
    variable[variable>max_val] = np.nan; variable[variable<min_val] = np.nan
    valid = ~np.isnan(variable)
    return valid

def quality_screening(variable, min_val, max_val, date_exclusions,
                      dependencies, foken_flags):
    """
    Returns filtered data of the variable desired using folken_flags, 
    physical possible values and date exclusion

    Parameters
    ----------
    variable : array
        Variable to screen.
    min_val : float
        Minimal possible value.
    max_val : float
        Maximal possible value.
    date_exclusions : boolean array or float
        Dates which are going to be filtered from the sample. If float it does
        not apply the filter.
    dependencies : boolean array or float
        Boolean array where false implies wrong data from the dependency variable
        thus wrong data of the desired variable. If float it does not apply the
        filter.
    foken_flags : array or float
        Flux quality flags for micrometeorological tests using Mauder and Foken
        (2004) policy. If float, it does not apply the filter.

    Returns
    -------
    variable : array
        Filtered variable.

    """
    variable = np.copy(variable)
    variable[variable>max_val] = np.nan; variable[variable<min_val] = np.nan
    variable[date_exclusions] = np.nan
    try:
        float(dependencies)
        pass
    except TypeError:
        if isinstance(dependencies, list):
            for k in range(len(dependencies)):
                dep = dependencies[k]
                variable[~dep] = np.nan
        else:
            variable[~dependencies] = np.nan
    try:
        float(date_exclusions)
        pass
    except TypeError:
        variable[date_exclusions] = np.nan
    try:
        float(foken_flags)
        pass
    except TypeError:
        variable[foken_flags==2] = np.nan
    return variable


def biomet_gap_fill(variable, era5_var, date_var, date_era5, method):
    """
    

    Parameters
    ----------
    variable : TYPE
        DESCRIPTION.
    era5_var : TYPE
        DESCRIPTION.

    Returns
    -------
    variable : TYPE
        DESCRIPTION.
    score : TYPE
        DESCRIPTION.

    """
    # valid_intersection = np.where(date_era5.intersection(date_var)) * ~(np.isnan(variable))
    variable = np.copy(variable)
    valid_intersection = date_era5.intersection(date_var)
    missing = date_era5.difference(date_var)
    invalid_data = date_var[np.isnan(variable)]
    new_var = np.zeros(len(date_era5)) * np.nan
    new_var = pd.Series(new_var, index=date_era5)
    variable = pd.Series(np.float64(variable), index=date_var)
    if method == 'replace':
        era5_var = pd.Series(np.float64(era5_var), index=date_era5)
        new_var[valid_intersection] = variable[valid_intersection]
        new_var[invalid_data] = era5_var[invalid_data]
        new_var[missing] = era5_var[missing]
    elif method =='smart replace':
        era5_var = pd.Series(np.float64(era5_var), index=date_era5)
        new_var[valid_intersection] = variable[valid_intersection]
        new_var[invalid_data] = era5_var[invalid_data]- np.nanmean(era5_var) + np.nanmean(variable)
        new_var[missing] = era5_var[missing]- np.nanmean(era5_var) + np.nanmean(variable)
    elif method=='linear fill':
        era5_var = pd.Series(np.float64(era5_var), index=date_era5)
        reg = linear_model.LinearRegression()
        era5_var2 = np.float64(era5_var[valid_intersection])
        variable2 = np.float64(variable[valid_intersection])
        valid = ~np.isnan(variable2) * ~np.isnan(era5_var2)
        era5_var2 = era5_var2[valid]; variable2 = variable2[valid]
        reg.fit(era5_var2.reshape(-1, 1), variable2)
        predicted_data = reg.predict(np.float64(era5_var).reshape(-1, 1))
        predicted_data = pd.Series(predicted_data, index=date_era5)
        new_var[valid_intersection] = variable[valid_intersection]
        new_var[invalid_data] = np.float64(predicted_data[invalid_data])
        new_var[missing] = predicted_data[missing]
    elif method=='linear replace':
        era5_var = pd.Series(np.float64(era5_var), index=date_era5)
        reg = linear_model.LinearRegression()
        era5_var2 = np.float64(era5_var[valid_intersection])
        variable2 = np.float64(variable[valid_intersection])
        valid = ~np.isnan(variable2) * ~np.isnan(era5_var2)
        era5_var2 = era5_var2[valid]; variable2 = variable2[valid]
        reg.fit(era5_var2.reshape(-1, 1), variable2)
        predicted_data = reg.predict(np.float64(era5_var).reshape(-1, 1))
        new_var = predicted_data
    elif 'none':
        new_var[valid_intersection] = variable[valid_intersection]
    else:
        print('Method no valid, try smart replace or replace as string')
    return new_var
        

def dropped(variable, window, difference):
    """
     It corrects short periods in which the time series sticks to some value that is 
     statistically different from the average value calculated over the whole
     period.

    Parameters
    ----------
    variable : TYPE
        DESCRIPTION.
    window : TYPE
        DESCRIPTION.
    difference : TYPE
        DESCRIPTION.

    Returns
    -------
    variable : TYPE
        DESCRIPTION.

    """
    variable[window] = variable[window] + difference
    return variable

def Convert(lst):
    return [ -i for i in lst ]


def esat_slope(TA,formula="Sonntag_1990"):
    """esat_slope(TA,formula="Sonntag_1990")
    Calculates saturation vapor pressure (Esat) over water and the
    corresponding slope of the saturation vapor pressure curve.
    esat (kPa) is calculated using the Magnus equation:
    esat = a * exp((b * TA) / (c + TA)) / 1000}
    where the coefficients a, b, c take different values depending on the formula used.
    The default values are from Sonntag 1990 (a=611.2, b=17.62, c=243.12). This version
    of the Magnus equation is recommended by the WMO (WMO 2008; p1.4-29). Alternatively,
    parameter values determined by Alduchov & Eskridge 1996 or Allen et al. 1998 can be
    used (see references).
    The slope of the Esat curve delta is calculated as the first derivative of the function:
      delta = dEsat / dTA
    which is solved using sympy.
    Parameters
    ----------
    TA : list or list like
        Air temperature (deg C)
    formula : string
        Formula to be used. Either Sonntag_1990 (Default), Alduchov_1996, or Allen_1998.
    Returns
    -------
    esat : list or list like
        Saturation vapor pressure (kPa)
    delat: list or list like
        Slope of the saturation vapor pressure curve (kPa K-1)
    References
    ----------
    - Sonntag D. 1990: Important new values of the physical constants of 1986, vapor
      pressure formulations based on the ITS-90 and psychrometric formulae.
      Zeitschrift fuer Meteorologie 70, 340-344.
    - World Meteorological Organization 2008: Guide to Meteorological Instruments
      and Methods of Observation (WMO-No.8). World Meteorological Organization,
      Geneva. 7th Edition.
    - Alduchov, O. A. & Eskridge, R. E., 1996: Improved Magnus form approximation of
      saturation vapor pressure. Journal of Applied Meteorology, 35, 601-609
    - Allen, R.G., Pereira, L.S., Raes, D., Smith, M., 1998: Crop evapotranspiration -
      Guidelines for computing crop water requirements - FAO irrigation and drainage
      paper 56, FAO, Rome.
    """
    if formula == "Sonntag_1990":
      a = 611.2
      b = 17.62
      c = 243.12
    elif formula == "Alduchov_1996":
      a = 610.94
      b = 17.625
      c = 243.04
    elif formula == "Allen_1998":
      a = 610.8
      b = 17.27
      c = 237.3
    else:
      raise RuntimeError("Formula for Esat_slope not recognized: "+formula+" try: Sonntag_1990, Alduchov_1996, or Allen_1998")

    _a, _b, _c, _TA = sympy.symbols("_a _b _c _TA")
    expr = _a * sympy.exp((_b * _TA) / (_c + _TA))
    expr = expr.subs([(_a,a), (_b,b), (_c,c)])
    Pa2kPa       = 0.001       # conversion pascal (Pa) to kilopascal (kPa)
    # saturation vapor pressure
    esat = sympy.lambdify(_TA, expr, "numpy")(TA)
    esat = esat * Pa2kPa
    # slope of the saturation vapor pressure curve
    d_esat = sympy.diff(expr, _TA)
    delta  = sympy.lambdify(_TA, d_esat, "numpy")(TA)
    delta = delta * Pa2kPa
    return(esat,delta)


def RH_to_VPD(RH, TA, formula="Sonntag_1990"):
    """RH_to_VPD(RH, TA, formula="Sonntag_1990")
  
    Conversion between relative humidity (RH) and vapor pressure deficit (VPD).
    Parameters
    ----------
    RH : list or list like
        Relative humidity (fraction between 0-1)
    TA : list or list like
        Air temperature (deg C)
    Returns
    -------
    VPD : list or list like
        Vapor pressure deficit (Pa)
    References
    ----------
    - Foken, T, 2008: Micrometeorology. Springer, Berlin, Germany.
    """
    RH = RH / 100
    esat, _ = esat_slope(TA, formula=formula)
    VPD     = esat - RH*esat
    return(VPD*1000)


def wind_dir_calc(u, v):
    """
    Calculates the wind direction angle using the wind speed components.

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.

    Returns
    -------
    angle : TYPE
        DESCRIPTION.

    """
    V = np.sqrt(u**(2) + v**(2))
    angle = np.arcsin(-u / V)
    angle = angle * (180 / np.pi)  # radian to degree
    angle[angle<0] = angle[angle<0] + 360
    return angle

def similar_conditions(var, dev, i, hours, method):
    if method=='valid':
        range_var = (var[i] > (var - dev)) & (var[i] < (var + dev))
        minimum_date = var.index[i] - pd.Timedelta(hours=hours)
        maximum_date = var.index[i] + pd.Timedelta(hours=hours)
        locs = (var.index > minimum_date) & (var.index < maximum_date)
        boolean = range_var & locs
    elif method=='non valid':
        minimum_date = var.index[i] - pd.Timedelta(hours=hours)
        maximum_date = var.index[i] + pd.Timedelta(hours=hours)
        locs = (var.index > minimum_date) & (var.index < maximum_date)
        boolean = locs
    return  boolean


def energy_balance_ratio(day_mask, H, LE, Rn, G, J, date):
    """
    Calculates the Energy Balance Ratio EBR defined in Mauder et al 2013.

    Parameters
    ----------
    day_mask : boolean array
        True if is daytime, False if it is not.
    H : array
        Sensible heat flux [W/m2].
    LE : array
        Latent heat flux [W/m2].
    Rn : array
        Net radiation [W/m2].
    G : array
        Soil heat flux [W/m2].
    J : array
        Storage term [W/m2].
    date : pandas date time index
        Dates of the provided data.

    Returns
    -------
    EBR : float
        Energy Balance Ratio.

    """
    night_mask = ~day_mask
    H[night_mask] = np.nan
    LE[night_mask] = np.nan
    Rn[night_mask] = np.nan
    G[night_mask] = np.nan
    J[night_mask] = np.nan
    daily_H = pd.Series(H, index=date).resample('D').agg(pd.Series.sum,
                            skipna=True)
    daily_G = pd.Series(G, index=date).resample('D').agg(pd.Series.sum,
                            skipna=True)
    daily_LE = pd.Series(LE, index=date).resample('D').agg(pd.Series.sum,
                            skipna=True)
    daily_Rn = pd.Series(Rn, index=date).resample('D').agg(pd.Series.sum,
                            skipna=True)
    daily_J = pd.Series(J, index=date).resample('D').agg(pd.Series.sum,
                            skipna=True)
    EBR = np.sum(daily_H + daily_LE) / np.sum(daily_Rn + daily_G + daily_J)
    return EBR


def energy_balance_residual_correction(EBR, F, day_mask):
    """
    Calculates the systematic error of a energy flux term.

    Parameters
    ----------
    EBR : float
        Energy Balance Ratio as defined in .
    F : array 
        Scalar flux.
    day_mask : boolean array
        True if it is daytime, False if it is not.

    Returns
    -------
    osys_f : TYPE
        DESCRIPTION.

    """
    osys_f = F * ((1 / EBR) - 1)
    osys_f[~day_mask] = np.nan
    return osys_f


# def SVR():
#     # Linear Support Vector Regression as implemented in Kang et al (2019).


def extreme_values_reading(colname, data):
    tmp = data[colname]
    var = np.zeros(len(tmp)) * np.nan
    for i in range(len(tmp)):
        try:
            var[i] = np.float(tmp.iloc[i])
        except ValueError:
            print('In '+colname+ ' there are some extreme computing values.')
            print(str(tmp.iloc[i]) + ' was turned into 0.')
            var[i] = 0
    return var

#%% Reading necessary variables - Dependencies
print('Reading necessary variables')
print('....................................')
co2_var = var_reading('co2_var', eddy_data, True)
h2o_var = extreme_values_reading('h2o_var', eddy_data)
u_var = var_reading('u_var', eddy_data, True)
v_var = var_reading('v_var', eddy_data, True)
w_var = var_reading('w_var', eddy_data, True)
ts_var = var_reading('ts_var', eddy_data, True)
h2o_var = dependencies_qc(h2o_var, 0, .8)
co2_var = dependencies_qc(co2_var, 0, 2)
u_var = dependencies_qc(u_var, 0, 20)
v_var = dependencies_qc(v_var, 0, 20)
w_var = dependencies_qc(w_var, 0, 10)
ts_var = dependencies_qc(ts_var, 0, 2)
flowrate = var_reading('flowrate_mean', eddy_data, True)
flowrate = (flowrate>0.0001) * (flowrate<0.00015)
#%% Biomet Variables
Ta = var_reading('Ta', biomet_data, True)
Pa = var_reading('Pa', biomet_data, True)
Td = var_reading('Td', biomet_data, True)
Tc = var_reading('Tc', biomet_data, True)
Rn = var_reading('Rn', biomet_data, True)
LWin = var_reading('LWin', biomet_data, True)
LWout = var_reading('LWout', biomet_data, True)
SWin = var_reading('SWin', biomet_data, True)
SWout = var_reading('SWout', biomet_data, True)
PPFD = var_reading('PPFD', biomet_data, True)
P_rain = var_reading('P_rain', biomet_data, True)
MWS = var_reading('MWS', biomet_data, True)
WD = var_reading('WD', biomet_data, True)
Ts = var_reading('Ts', biomet_data, True)
SWC = var_reading('SWC', biomet_data, True)
SHF = var_reading('SHF', biomet_data, True)
#%% Eddy Variables
Tau = var_reading('Tau', eddy_data, True)
qc_Tau = var_reading('qc_Tau', eddy_data, True)
H = var_reading('H', eddy_data, True)
qc_H = var_reading('qc_H', eddy_data, True)
LE = var_reading('LE', eddy_data, True)
qc_LE = var_reading('qc_LE', eddy_data, True)
FCO2 = var_reading('co2_flux', eddy_data, True)
qc_FCO2 = var_reading('qc_co2_flux', eddy_data, True)
h2o_flux = var_reading('h2o_flux', eddy_data, True)
qc_h2o_flux = var_reading('qc_h2o_flux', eddy_data, True)
H_strg = var_reading('H_strg', eddy_data, True)
LE_strg = var_reading('LE_strg', eddy_data, True)
co2_strg = var_reading('co2_strg', eddy_data, True)
h2o_strg = var_reading('h2o_strg', eddy_data, True)
NEE = FCO2 + co2_strg
co2_mix_ratio = var_reading('co2_mixing_ratio', eddy_data, True)
h2o_mix_ratio = extreme_values_reading('h2o_mixing_ratio', eddy_data)
air_temperature = var_reading('air_temperature', eddy_data, True)
sonic_temperature = var_reading('sonic_temperature', eddy_data, True)
air_pressure = var_reading('air_pressure', eddy_data, True)
ET = var_reading('ET', eddy_data, True)
RH = var_reading('RH', eddy_data, True)
q = extreme_values_reading('specific_humidity', eddy_data)
T_dew = var_reading('Tdew', eddy_data, True)
wind_dir = var_reading('wind_dir', eddy_data, True)
wind_speed = var_reading('wind_speed', eddy_data, True)
u_star = var_reading('u*', eddy_data, True)
VPD = extreme_values_reading('VPD', eddy_data)
L = var_reading('L', eddy_data, True)  # Monin-obukov length
#%% ERA5 vars
u_era5 = var_reading('u10', ds, True)
v_era5 = var_reading('v10', ds, True)
wd_era5 = wind_dir_calc(u_era5, v_era5)
RH_era5 = var_reading('RH', ds, True)
Ta_era5 = var_reading('t2m', ds, True)
VPD_era5 = RH_to_VPD(RH_era5/100, Ta_era5-273.15) 
#%% Quality screening - biomet variables
print('Doing quality screening on the variables')
print('....................................')
wrong_data_dates = (date_biomet.year==2016)*(date_biomet.month==9)*(date_biomet.day==15)*(date_biomet.hour>15)
Ta_qc = quality_screening(Ta, -5, 32, 0, 0, 0)
Pa_qc = quality_screening(Pa, 85000, 106000, 0, 0, 0)
Td_qc = quality_screening(Td, -25, 27, 0, 0, 0)
Tc_qc = quality_screening(Tc, -10, 32, 0, 0, 0)
Rn_qc = quality_screening(Rn, -200, 1200, 0, 0, 0)
LWin_qc = quality_screening(LWin, -300, 10, 0, 0, 0)
LWout_qc = quality_screening(LWout, -50, 50, 0, 0, 0)
SWin_qc = quality_screening(SWin, 0, 1200, 0, 0, 0)
SWout_qc = quality_screening(SWout, 0, 200, 0, 0, 0)
# wrong_data_dates = ((date_biomet.year==2016)*((date_biomet.month==2)+(date_biomet.month==3)+
                                      # (date_biomet.month==4)+
                                      # (date_biomet.month==5)))+(date_biomet.year>2020)
PPFD_qc = quality_screening(PPFD, 0, 2500, 0, 0, 0)
P_rain_qc = quality_screening(P_rain, 0, 75, 0, 0, 0)
MWS_qc = quality_screening(MWS, 0, 20, 0, 0, 0)
WD_qc = quality_screening(WD, -180, 180, 0, 0, 0)
Ts_qc = quality_screening(Ts, -5, 30, 0, 0, 0)
SWC_qc = quality_screening(SWC, 0.01, 1, 0, 0, 0)
SHF_qc = quality_screening(SHF, -15, 15, 0, 0, 0)
#%% QS - eddy data
Tau_qc = quality_screening(Tau, -3.5, 2, 0, [u_var, v_var, w_var], qc_Tau)
H_qc = quality_screening(H, -200, 630, 0, [w_var, ts_var], qc_H)
LE_qc = quality_screening(LE, -300, 650, 0, [h2o_var, w_var, ts_var, flowrate],
                          qc_LE)
FCO2_qc = quality_screening(FCO2, -100, 400, 0, [co2_var, h2o_var, w_var, ts_var],
                           qc_FCO2)
h2o_flux_qc = quality_screening(h2o_flux, -20, 20, 0, [h2o_var, flowrate],
                                qc_h2o_flux)
H_strg_qc = quality_screening(H_strg, -150, 150, 0, 0, 0)  # SH
# wrong_data_dates = (date.year==2015)*((date.month==7)+(date.month==6))
LE_strg_qc = quality_screening(LE_strg, -170, 170, 0, 0, 0) #SLE
co2_strg_qc = quality_screening(co2_strg, -40, 40, 0, 0, 0)
h2o_strg_qc = quality_screening(h2o_strg, -5, 5, 0, 0, 0)
co2_mix_ratio_qc = quality_screening(co2_mix_ratio, 150, 1200, 0, 0, 0)
h2o_mix_ratio_qc = quality_screening(h2o_mix_ratio, 0, 50, 0, 0, 0)
air_temperature_qc = quality_screening(air_temperature, -10+273.15, 32+273.15,
                                       0, 0, 0)
air_pressure_qc = quality_screening(air_pressure, 85000, 106000, 0, 0, 0)
ET_qc = quality_screening(ET, 0, 1, 0, [flowrate, h2o_var, w_var, ts_var],
                          qc_LE)
q_qc = quality_screening(q, 0, 0.02, 0, 0, 0)
# wrong_data_dates = ((date.year==2015)*(date.month>5))
RH = dropped(RH, (date.year==2021)*(date.month>9), 50)
RH = dropped(RH, (date.year==2022), 50)
RH_qc = quality_screening(RH, 5, 100, 0, 0, 0)
# VPD = dropped(VPD, (date.year==2021)*(date.month>9), -750)
# VPD = dropped(VPD, (date.year==2022), -750)
VPD_qc = quality_screening(VPD, 0, 5500, 0, 0, 0)
u_star_qc = quality_screening(u_star, 0, 8, 0, [u_var, v_var, w_var], 0)
L_qc = quality_screening(L, -3000, 3000, 0, 0, 0)
NEE_qc = FCO2_qc #+ co2_strg_qc  # Peatland NEE calculation have little to non storage term
NEE_qc = quality_screening(NEE_qc, -35, 35, 0, [co2_var, h2o_var, w_var, ts_var],
                           qc_FCO2)
#%% Non filled data - same size data
# Quality controlled data (Simple screening) - biomet data
Ta_o = biomet_gap_fill(Ta_qc, 0, date_biomet, date_era5, 'none')
Pa_o = biomet_gap_fill(Pa_qc, 0, date_biomet, date_era5, 'none')
Td_o = biomet_gap_fill(Td_qc, 0, date_biomet, date_era5, 'none')
Tc_o = biomet_gap_fill(Tc_qc, 0, date_biomet, date_era5, 'none')
Rn_o = biomet_gap_fill(Rn_qc, 0, date_biomet, date_era5, 'none')
LWin_o = biomet_gap_fill(LWin_qc, 0, date_biomet, date_era5, 'none')
LWout_o = biomet_gap_fill(LWout_qc, 0, date_biomet, date_era5, 'none')
SWin_o = biomet_gap_fill(SWin_qc, 0, date_biomet, date_era5, 'none')
SWout_o = biomet_gap_fill(SWout_qc, 0, date_biomet, date_era5, 'none')
PPFD_o = biomet_gap_fill(PPFD_qc, 0, date_biomet, date_era5, 'none')
P_rain_o = biomet_gap_fill(P_rain_qc, 0, date_biomet, date_era5, 'none')
MWS_o = biomet_gap_fill(MWS_qc, 0, date_biomet, date_era5, 'none')
WD_o = biomet_gap_fill(WD_qc, 0, date_biomet, date_era5, 'none')
Ts_o = biomet_gap_fill(Ts_qc, 0, date_biomet, date_era5, 'none')
SWC_o = biomet_gap_fill(SWC_qc, 0, date_biomet, date_era5, 'none')
SHF_o = biomet_gap_fill(SHF_qc, 0, date_biomet, date_era5, 'none')
# Quality controlled data (Simple screening) - eddy data
Tau_o = biomet_gap_fill(Tau_qc, 0, date, date_era5, 'none')
H_o = biomet_gap_fill(H_qc, 0, date, date_era5, 'none')
LE_o = biomet_gap_fill(LE_qc, 0, date, date_era5, 'none')
NEE_o = biomet_gap_fill(NEE_qc, 0, date, date_era5, 'none')
h2o_flux_o = biomet_gap_fill(h2o_flux_qc, 0, date, date_era5, 'none')
H_strg_o = biomet_gap_fill(H_strg_qc, 0, date, date_era5, 'none')
LE_strg_o = biomet_gap_fill(LE_strg_qc, 0, date, date_era5, 'none')
co2_strg_o = biomet_gap_fill(co2_strg_qc, 0, date, date_era5, 'none')
h2o_strg_o = biomet_gap_fill(h2o_strg_qc, 0, date, date_era5, 'none')
co2_mix_ratio_o = biomet_gap_fill(co2_mix_ratio_qc, 0, date, date_era5, 'none')
h2o_mix_ratio_o = biomet_gap_fill(h2o_mix_ratio_qc, 0, date, date_era5, 'none')
air_temperature_o = biomet_gap_fill(air_temperature_qc, 0, date, date_era5, 'none')
air_pressure_o = biomet_gap_fill(air_pressure_qc, 0, date, date_era5, 'none')
ET_o = biomet_gap_fill(ET_qc, 0, date, date_era5, 'none')
q_o = biomet_gap_fill(q_qc, 0, date, date_era5, 'none')
RH_o = biomet_gap_fill(RH_qc, 0, date, date_era5, 'none')
VPD_o = biomet_gap_fill(VPD_qc, 0, date, date_era5, 'none')
ustar_o = biomet_gap_fill(u_star_qc, 0, date, date_era5, 'none')
L_o = biomet_gap_fill(L_qc, 0, date, date_era5, 'none')
#%%
NEE_f = biomet_gap_fill(NEE_qc, ds['u10'], date, date_era5, 'none')
ustar_f = biomet_gap_fill(u_star_qc, ds['u10'], date, date_era5, 'none')
#%% Biomet Gapfilling
print('Gapfilling biometeorological data')
print('....................................')
Ta_f = biomet_gap_fill(Ta_qc, ds['t2m']-273.15, date_biomet, date_era5, 'linear fill')
LWin_f = biomet_gap_fill(LWin_qc, ds['msdwlwrf'], date_biomet, date_era5, 'linear fill') #downward rad
LWout_f = biomet_gap_fill(LWout_qc, ds['msnlwrf']-ds['msdwlwrf'], date_biomet,
                                                      date_era5, 'linear fill') #upward rad
SWin_f = biomet_gap_fill(SWin_qc, ds['msdwswrf'], date_biomet, date_era5, 'replace') #downward rad
SWout_f = biomet_gap_fill(SWout_qc, -ds['msnswrf']+ds['msdwswrf'], date_biomet, date_era5, 'linear fill') #downward rad
H_f = biomet_gap_fill(H_qc, ds['msshf'], date, date_era5, 'linear fill')
SWC_f = biomet_gap_fill(SWC_qc, ds['swvl1'], date_biomet, date_era5, 'linear fill')
SHF_f = biomet_gap_fill(SHF, ds['msshf']+ds['mslhf'], date_biomet, date_era5, 'linear fill')
RH_f = biomet_gap_fill(RH_qc, ds['RH'], date, date_era5, 'linear fill')  # recalcular usando rh de era5 orig
LE_f = biomet_gap_fill(LE_qc, pd.Series(Convert(ds['mslhf'])), date, date_era5,'linear fill')
Pa_f = biomet_gap_fill(Pa_qc, ds['sp'], date_biomet, date_era5, 'linear fill')
Rn_f = biomet_gap_fill(Rn_qc, ds['msnlwrf']+ds['msnswrf'], date_biomet, date_era5, 'linear fill')
Ts_f = biomet_gap_fill(Ts_qc, ds['stl1'], date_biomet, date_era5, 'linear fill')
VPD_f = biomet_gap_fill(VPD_qc, VPD_era5, date, date_era5, 'linear fill')
MWS_f = biomet_gap_fill(MWS_qc, np.sqrt(ds['u10']**(2) + ds['v10']**(2)), date_biomet, date_era5, 'replace')
wind_dir = biomet_gap_fill(wind_dir, wd_era5, date, date_era5, 'replace')
# P_rain_f = biomet_gap_fill(P_rain_qc, ds['tp']*1000, )
#%% Energy Balance Residual Correction
# Not available yet, previous Storage term calculation is needed.
# EBR = energy_balance_ratio(Rn_f>20, H_f, LE_f, Rn_f, SHF_f, J, date)
#%% u_star filtering - (Reichstein et al., 2005) & (papale et al 2006)
print('u* filtering 100 times, to asses the uncertainty of u* threshold - could take some time')
print('....................................')
df = pd.DataFrame({'NEE': NEE_f, 'USTAR':ustar_f, 'TA':Ta_f}, index=date_era5)
mask = df.index.year<2022
df = df[mask]  # full years requierement
dff = df.copy(deep=True)  # Flag
dff[:] = 0
df[df.isna()] = undef; dff[df.isna()] = 2;
isday = SWin_f > 10; isday = isday[SWin_f.index.year<2022]  # full years requirement
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    ustars, flag = hf.ustarfilter(df, flag=dff, isday=isday, undef=undef,
                                  ustarmin=ustarmin, nboot=nboot, nustarclasses=20,
                                  ntaclasses=100, plot=True)
# Cleaning data flagged as bad quality determined by the u* detection tool
nee_us = np.copy(np.float64(df['NEE']))
nee_us[nee_us==undef] = np.nan
nee_us[flag==2] = np.nan
nee_us = pd.Series(nee_us, index= df.index)

#%% Configurable Gapfilling from hesseflux ICOS method
print('Configurable Gap-filling')
print('....................................')
longgap = 90  # Avoid extrapolation in gaps longer than longgap days
df = pd.DataFrame({'H': H_f, 'LE': LE_f, 'NEE': NEE_f, 'SW_IN': SWin_f, 
                   'TA':Ta_f, 'VPD':VPD_f}, index=date_era5)
dff = df.copy(deep=True)  # Flag
dff[:] = 0
dff[df.isna()] = 2; df[df.isna()] = undef
# if available
hfill = ['H', 'LE', 'NEE',
         'SW_IN', 'TA', 'VPD']
df_f, dff_f = hf.gapfill(df, flag=dff, sw_dev=sw_dev, ta_dev=ta_dev, 
                         vpd_dev=vpd_dev,longgap=longgap, undef=undef, 
                         err=False,verbose=1)
df[df==undef] = np.nan
def _add_f(c):
    return '_'.join(c.split('_')[:-3] + ['f'] + c.split('_')[-3:])
df_f.rename(columns=_add_f,  inplace=True)
dff_f.rename(columns=_add_f, inplace=True)
df  = pd.concat([df,  df_f],  axis=1)
dff = pd.concat([dff, dff_f], axis=1)
NEE_f = var_reading('f_NEE', df_f, True)
H_f = var_reading('f_H', df_f, True)
LE_f = var_reading('f_LE', df_f, True)
plt.plot(date_era5, NEE_f)
#%% Partitioning
print('CO2 flux partitioning - Reichtein and Lasslop')
print('....................................')
# DataFrame
df = pd.DataFrame({'NEE': NEE_f, 'SW_IN':SWin_f, 'TA':Ta_f, 'VPD':VPD_f})
df[df.isna()] = undef
# Flag
dff = df.copy(deep=True).astype(int)
dff[:] = 0
dff[df.isna()] = 2
dff[df==-9999] = 2
# Separation day-night
isday = SWin_f > 10
# bool
nogppnight = False  # if True, set GPP=0 at night
# # Reichtein
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=scipy.optimize.OptimizeWarning)
    dfpart_reichtein = hf.nee2gpp(df, flag=dff, isday=isday,
                                  undef=undef, method='reichstein',
                                  nogppnight=nogppnight)
dfpart_reichtein[dfpart_reichtein==undef] = np.nan
dfpart_reichtein[dfpart_reichtein> 1000] = np.nan
gpp_r = dfpart_reichtein['GPP']; reco_r = dfpart_reichtein['RECO']
reco_r[reco_r==0] = np.nan
#%% Lasslop
dff[df==-9999] = 2
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    dfpart_lasslop = hf.nee2gpp(df, flag=dff, isday=isday,
                          undef=undef, method='lasslop',
                          nogppnight=nogppnight)
dfpart_lasslop[dfpart_lasslop==undef] = np.nan
gpp_l = dfpart_lasslop['GPP']; reco_l = dfpart_lasslop['RECO']
#%% Flux error estimates
print('Flux error estimates')
print('....................................')
hfill = ['H', 'LE', 'NEE', 'H_f', 'LE_f', 'NEE_f', 'GPP', 'RECO', 'SW_IN', 'TA', 'VPD']
# hfill = ['NEE', 'GPP', 'SW_IN', 'TA_', 'VPD']
df_f_err = hf.gapfill(df, flag=dff,
                  sw_dev=sw_dev, ta_dev=ta_dev, vpd_dev=vpd_dev,
                  longgap=longgap, undef=undef, err=True, verbose=1)
colin = list(df_f_err.columns)
df_f_err.rename(columns=_add_f,  inplace=True)
colout = list(df_f_err.columns)
df_err = pd.concat([df_f, df_f_err], axis=1)
# take flags of non-error columns
for cc in range(len(colin)):
    dff[colout[cc]] = dff[colin[cc]]
NEE_err = df_err['f_NEE']
#%% ET filling
ET_f2 = LE_to_ET(LE_f, Ta_f)
ET_f = ET_f2.copy()
ET_f[ET_o.isna()] = ET_f2[ET_o.isna()]
ET_f[ET_f < 0] = 0
#%% Data saving
df_final = pd.DataFrame({'TIMESTAMP': date_era5, 'Ta': Ta_o, 'Ta_f': Ta_f, 
                         'Pa': Pa_o, 'Pa_f':Pa_f, 'RH': RH_o, 'RH_f': RH_f,
                         'Td': Td_o, 'Tc': Tc_o, 'Rn': Rn_o, 'Rn_f': Rn_f,
                         'LWin': LWin_o, 'LWin_f': LWin_f, 'LWout': LWout_o,
                         'LWout_f': LWout_f,'SWin': SWin_o, 'SWin_f': SWin_f,
                         'SWout': SWout_o, 'SWout_f': SWout_f, 'PPFD': PPFD_o,
                         'P_rain': P_rain_o, 'MWS': MWS_o, 'MWS_f':MWS_f, 
                         'WD': WD_o, 'Ts': Ts_o, 'Ts_f': Ts_f, 'SWC': SWC_o,
                         'SWC_f': SWC_f, 'SHF': SHF_o, 'SHF_f': SHF_f, 
                         'LE': LE_o, 'LE_f': LE_f, 'H': H_o, 'H_f': H_f, 'NEE': NEE_o,
                         'NEE_f': NEE_f, 'NEE_err': NEE_err,'VPD': VPD_o, 'vPD_f': VPD_f, 'ustar': ustar_o,
                         'GPP_DT': gpp_l, 'GPP_NT': gpp_r, 'ET': ET_o,
                         'ET_f': ET_f})
df_final.to_csv(path+'LEVEL3_SDP_2013_2021.csv', index=False)
print('Saving data to a csv')
print('....................................')
