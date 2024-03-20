# -*- coding: utf-8 -*-
"""
Eddy Covariance Post-Processing Script - Omora Peatland
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
path = 'D:/ERA5_chiloe/'
# ds = pd.read_csv(path+'era5_data.csv')
# units_era5 = pd.read_csv(path+'metadata.csv')['units']
# vars_era5 = pd.read_csv(path+'metadata.csv')['long_name']
# date_era5 = pd.DatetimeIndex(ds['time']) - pd.Timedelta(hours=3)  # UTC to local time
#%% Data reading Fray Jorge
# Biomet
PATH = 'E:/FrayJorge Eddy/Output/'
FILENAME = 'eddypro_FrayJorge_biomet_2023-06-20T092652_adv.csv'
biomet_data = pd.read_csv(PATH+FILENAME)
units_bm_fj = biomet_data.iloc[0]; biomet_data = biomet_data[1:]
date_biomet = pd.DatetimeIndex(biomet_data.date.astype(str)+ ' '
                                  + biomet_data.time.astype(str))
biomet_data.index = date_biomet_data
biomet_data = biomet_data.drop(columns=['date', 'time'])
# Fulloutput
FILENAME = 'eddypro_FrayJorge_full_output_2023-06-20T092652_adv.csv'
eddy_data = pd.read_csv(PATH+FILENAME, skiprows=[0])
eddy_units = eddy_data.iloc[0]; eddy_data = eddy_data[1:]
date = pd.DatetimeIndex(eddy_data.date.astype(str)+ ' '
                                  + eddy_data.time.astype(str))
eddy_data.index = date_eddy_data
eddy_data = eddy_data.drop(columns=['date', 'time'])
eddy_data = eddy_data.astype(float, errors='ignore')
eddy_data[eddy_data==-9999] = np.nan
#%%
fco2 = pd.Series(np.float64(eddy_data['co2_flux']), index=date)
fco2[fco2<-999] = np.nan
fco2[fco2<np.nanpercentile(fco2, 0.1)] = np.nan
fco2[fco2>np.nanpercentile(fco2, 99.9)] = np.nan
fco2_daily = fco2.resample('D').mean()
# fco2_daily[fco2_daily<-1.5] = np.nan
fco2_monthly = fco2_daily.resample('M').mean()
#%%
plt.figure(figsize=(8,6))
plt.plot(fco2)
plt.ylabel(eddy_units['co2_flux'])
plt.title('30-min values of CO2 Flux at Peatland Omora')
plt.grid()
plt.figure(figsize=(8,6))
plt.plot(fco2_daily)
plt.ylabel(eddy_units['co2_flux'])
plt.title('daily values of CO2 Flux at Peatland Omora')
plt.grid()
plt.figure(figsize=(8,6))
plt.plot(fco2_monthly)
plt.ylabel(eddy_units['co2_flux'])
plt.title('monthly values of CO2 Flux at Peatland Omora')
plt.grid()
#%%
umols_to_30mingC = (12 * 10**(-6)) * 1800  #micromol/m2s a gramos de carbono/m2 30min
co2_flux = np.float64(eddy_data['co2_flux']) * umols_to_30mingC
co2_flux[co2_flux>2000] = np.nan
co2_flux[co2_flux<-2000] = np.nan
#%% Post-Processing constants definitions
# General constants
undef = -9999  # Non valid data value
# U-star filtering
ustarmin = 0.1  # Forest ustar minimum 
nboot = 100  # N° of boot straps for estimating the confidence interval of u* threshold
# Configurable Gap-filling
sw_dev = 50  # Max deviation of SW_IN
ta_dev = 2.5  # Max deviation of TA
vpd_dev = 5  # Max deviation of VPD
longgap = 35  # Avoid extrapolation in gaps longer than longgap days

#%% Cut data to fit the shortest period
date_i = date[0]; date_f = date[len(date)-2]
# ds = ds[np.where(date_i==date_era5)[0][0]: np.where(date_f==date_era5)[0][0]+1]
# date_era5 = date_era5[np.where(date_i==date_era5)[0][0]: np.where(date_f==date_era5)[0][0]+1]
biomet_data = biomet_data[np.where(date_i==date_biomet)[0][0]: np.where(date_f==date_biomet)[0][0]+1]
date_biomet = date_biomet[np.where(date_i==date_biomet)[0][0]: np.where(date_f==date_biomet)[0][0]+1]

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
    variable[~np.isfinite(variable)] = np.nan
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
    variable[~np.isfinite(variable)] = np.nan
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


def footprint(FETCHx, WD):
    """
    Function to apply a footprint analysis on the flux data. Return a boolean mask
    where "1" = data to remove.

    Parameters
    ----------
    FETCHx : List, array or Series
        Fetch Criteria to filter data. e.g.:FETCH_90 = Along-wind distance providing
        90% (cumulative) contribution to the turbulent fluxes.
    WD : List, array or Series
        Wind direction [0, 360]° from North.

    Returns
    -------
    erase : List
        Boolean list where "1" = data to remove from the fluxes.

    """
    newangles = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 160,
                          165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315,
                          322, 330, 345, 350, 361])
    distnew = np.array([270, 275, 225, 242, 297, 301, 270, 182, 225, 230, 169,
                        171, 353, 315, 308, 296, 322, 307, 309, 346, 351, 379,
                        450, 978, 1462, 1619, 1231, 270])
    erase = np.zeros_like(WD)
    for i in range(len(WD)):
        a = WD[i]
        if ~np.isnan(a):
            try:
                dist_i = distnew[np.max(np.where(a >= newangles))]
                dist_f = distnew[np.max(np.where(a >= newangles)) + 1]
                anglei = newangles[np.max(np.where(a >= newangles))]
                anglef = newangles[np.max(np.where(a >= newangles)) +1]
                #
                m = (dist_f - dist_i) / (anglef - anglei)
                n = (-m * anglei) + dist_i
                dist = (m * a) + n
                if dist < FETCHx[i]:
                    erase[i] = 1
            except ValueError:
                pass
    return erase

    
#%% Reading necessary variables - Dependencies
print('Reading necessary variables')
print('....................................')
co2_var = var_reading('co2_var', eddy_data, True)
h2o_var = var_reading('h2o_var', eddy_data, True)
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
# flowrate = var_reading('flowrate_mean', eddy_data, True)
# flowrate = (flowrate>0.0001) * (flowrate<0.00015)
#%% Biomet Variables
Ta = var_reading('Ta_1_1_1', biomet_data, True)
# Pa = var_reading('Pa_1_1_1', biomet_data, True)
# Td = var_reading('Td', biomet_data, True)
# Tc = var_reading('Tc', biomet_data, True)
Rn = var_reading('RN_1_1_1', biomet_data, True)
LWin = var_reading('Lwin_1_1_1', biomet_data, True)
LWout = var_reading('Lwout_1_1_1', biomet_data, True)
SWin = var_reading('Swin_1_1_1', biomet_data, True)
SWout = var_reading('Swout_1_1_1', biomet_data, True)
PPFD = var_reading('PPFD_1_1_1', biomet_data, True)
P_rain = var_reading('P_RAIN_1_1_1', biomet_data, True)
# MWS = var_reading('MWS', biomet_data, True)
# WD = var_reading('WD', biomet_data, True)
Ts1 = var_reading('Ts_1_1_1', biomet_data, True)
Ts2 = var_reading('Ts_2_1_1', biomet_data, True)
Ts3 = var_reading('Ts_3_1_1', biomet_data, True)
Ts4 = var_reading('Ts_4_1_1', biomet_data, True)
Ts5 = var_reading('Ts_5_1_1', biomet_data, True)
SWC1 = var_reading('SWC_1_1_1', biomet_data, True)
SWC2 = var_reading('SWC_2_1_1', biomet_data, True)
SWC3 = var_reading('SWC_3_1_1', biomet_data, True)
SWC4 = var_reading('SWC_1_1_2', biomet_data, True)
SWC5 = var_reading('SWC_1_1_2', biomet_data, True)
SWC6 = var_reading('SWC_1_1_3', biomet_data, True)
SWC7 = var_reading('SWC_1_1_4', biomet_data, True)
SWC8 = var_reading('SWC_1_1_5', biomet_data, True)
SWC9 = var_reading('SWC_1_1_6', biomet_data, True)

SHF1 = var_reading('SHF_1_1_1', biomet_data, True)
SHF2 = var_reading('SHF_2_1_1', biomet_data, True)
SHF3 = var_reading('SHF_3_1_1', biomet_data, True)
#
NR01TK = var_reading('TCNR4_C_1_1_1', biomet_data, True) + 273.15
NR01TK = pd.Series(NR01TK, index=date_biomet).resample('30min').mean()
NR01TK[NR01TK>310.15] = np.nan
NR01TK = NR01TK[1:np.where(NR01TK.index==date[len(date)-1])[0][0]]

#%% Long wave radiation correction
delt = 5.67 * 10**(-8)
# plt.plot(NR01TK-273.15)
plt.ylim([0,40])
plt.plot(date_biomet, Ta)
# plt.plot(LWin + delt*())
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
h2o_mix_ratio = var_reading('h2o_mixing_ratio', eddy_data, True)
air_temperature = var_reading('air_temperature', eddy_data, True)
sonic_temperature = var_reading('sonic_temperature', eddy_data, True)
air_pressure = var_reading('air_pressure', eddy_data, True)
ET = var_reading('ET', eddy_data, True) / 2
plt.plot(ET)
RH = var_reading('RH', eddy_data, True)
q = var_reading('specific_humidity', eddy_data, True)
T_dew = var_reading('Tdew', eddy_data, True)
wind_dir = var_reading('wind_dir', eddy_data, True)
wind_speed = var_reading('wind_speed', eddy_data, True)
u_star = var_reading('u*', eddy_data, True)
VPD = var_reading('VPD', eddy_data, True)

#%% Vars to Ameriflux
MO_LENGTH = var_reading('L', eddy_data, True)  # Monin-obukov length
L = var_reading('L', eddy_data, True)  # Monin-obukov length
FETCH_70 = var_reading('x_70%', eddy_data, True)
FETCH_90 = var_reading('x_90%', eddy_data, True)
MODEL = var_reading('model', eddy_data, True)
FETCH_MAX = var_reading('x_peak', eddy_data, True)
FETCH_offset = var_reading('x_offset', eddy_data, True)
# FETCH_FILTER	Footprint quality flag (i.e., 0, 1): 0 and 1 indicate data measured when wind coming from direction that should be discarded and kept, respectively	nondimensional
# FETCH_MAX	Distance at which footprint contribution is maximum	m
#%% ERA5 vars
# u_era5 = var_reading('u10', ds, True)
# v_era5 = var_reading('v10', ds, True)
# wd_era5 = wind_dir_calc(u_era5, v_era5)
# RH_era5 = var_reading('RH', ds, True)
# Ta_era5 = var_reading('t2m', ds, True)
# VPD_era5 = RH_to_VPD(RH_era5/100, Ta_era5-273.15) 
#%% Quality screening - biomet variables
print('Doing quality screening on the variables')
print('....................................')
# wrong_data_dates = (date_biomet.year==2014)*(date_biomet.month==7)*(date_biomet.day==25)*(date_biomet.hour>15)
Ta_qc = quality_screening(Ta, -10, 32, 0, 0, 0)
# Pa_qc = quality_screening(Pa, 96000, 103000, 0, 0, 0)
# Td_qc = quality_screening(Td, -25, 27, 0, 0, 0)
# Tc_qc = quality_screening(Tc, -10, 32, 0, 0, 0)
Rn_qc = quality_screening(Rn, -400, 1200, 0, 0, 0)
LWin_qc = quality_screening(LWin, -300, 10, 0, 0, 0)
LWout_qc = quality_screening(LWout, -50, 50, 0, 0, 0)
SWin_qc = quality_screening(SWin, 0, 1200, 0, 0, 0)
SWout_qc = quality_screening(SWout, 0, 200, 0, 0, 0)
# wrong_data_dates = ((date_biomet.year==2020)*((date_biomet.month==2)+(date_biomet.month==3)+
                                      # (date_biomet.month==4)+
                                      # (date_biomet.month==5)))+(date_biomet.year>2020)
PPFD_qc = quality_screening(PPFD, 0, 2500, 0, 0, 0)
wrong_data_dates = (pd.Timestamp('2014-01-01 00:00:00') < date_biomet) & (date_biomet < pd.Timestamp('2014-05-03 16:30:00'))
wrong_data_dates = wrong_data_dates | ((pd.Timestamp('2018-01-01 00:00:00') < date_biomet) & (date_biomet < pd.Timestamp('2018-10-24 11:30:00')))
wrong_data_dates = wrong_data_dates + (pd.Timestamp('2020-07-18 00:00:00') < date_biomet)
P_rain_qc = quality_screening(P_rain, 0, 50, wrong_data_dates, 0, 0)
# MWS_qc = quality_screening(MWS, 0, 20, 0, 0, 0)
# WD_qc = quality_screening(WD, -181, 181, 0, 0, 0)
Ts1_qc = quality_screening(Ts1, -5, 30, 0, 0, 0)
SWC1_qc = quality_screening(SWC1, 0.01, 1, 0, 0, 0)
SHF1_qc = quality_screening(SHF1, -15, 15, 0, 0, 0)
Ts2_qc = quality_screening(Ts2, -5, 30, 0, 0, 0)
SWC2_qc = quality_screening(SWC2, 0.01, 1, 0, 0, 0)
SHF2_qc = quality_screening(SHF2, -15, 15, 0, 0, 0)
Ts3_qc = quality_screening(Ts3, -5, 30, 0, 0, 0)
SWC3_qc = quality_screening(SWC3, 0.01, 1, 0, 0, 0)
SHF3_qc = quality_screening(SHF3, -15, 15, 0, 0, 0)
SWC4_qc = quality_screening(SWC4, 0.01, 1, 0, 0, 0)
SWC5_qc = quality_screening(SWC5, 0.01, 1, 0, 0, 0)
SWC6_qc = quality_screening(SWC6, 0.01, 1, 0, 0, 0)
SWC7_qc = quality_screening(SWC7, 0.01, 1, 0, 0, 0)
SWC8_qc = quality_screening(SWC8, 0.01, 1, 0, 0, 0)
SWC9_qc = quality_screening(SWC9, 0.01, 1, 0, 0, 0)

# wrong_data_dates = (date_biomet.year==2021)*(date_biomet.month>7)
NR01TK_qc = quality_screening(NR01TK, 263.15, 40+273.15, 0, 0, 0)
#%% QS - eddy data
Tau_qc = quality_screening(Tau, -3.5, 2, 0, [u_var, v_var, w_var], qc_Tau)
H_qc = quality_screening(H, -500, 850, 0, [w_var, ts_var], qc_H)
LE_qc = quality_screening(LE, -300, 650, 0, [h2o_var, w_var, ts_var],
                          qc_LE)
FCO2_qc = quality_screening(FCO2, -30, 30, 0, [co2_var, h2o_var, w_var, ts_var],
                           qc_FCO2)
h2o_flux_qc = quality_screening(h2o_flux, -20, 20, 0, [h2o_var],
                                qc_h2o_flux)
H_strg_qc = quality_screening(H_strg, -150, 150, 0, 0, 0)  # SH
wrong_data_dates = (date.year==2015)*((date.month==7)+(date.month==6))
LE_strg_qc = quality_screening(LE_strg, -170, 170, wrong_data_dates, 0, 0) #SLE
co2_strg_qc = quality_screening(co2_strg, -20, 15, 0, [co2_var, h2o_var, w_var, ts_var],
                                qc_FCO2)
h2o_strg_qc = quality_screening(h2o_strg, -5, 5, 0, 0, 0)
co2_mix_ratio_qc = quality_screening(co2_mix_ratio, 150, 1200, 0, 0, 0)
h2o_mix_ratio_qc = quality_screening(h2o_mix_ratio, 0, 50, 0, 0, 0)
air_temperature_qc = quality_screening(air_temperature, -10+273.15, 32+273.15,
                                       0, 0, 0)
air_pressure_qc = quality_screening(air_pressure, 85000, 106000, 0, 0, 0)
ET_qc = quality_screening(ET, 0, 0.5, 0, [h2o_var, w_var, ts_var],
                          qc_LE)
q_qc = quality_screening(q, 0, 0.02, 0, 0, 0)
wrong_data_dates = ((date.year==2015)*(date.month>5))
RH = dropped(RH, (date.year==2021)*(date.month>9), 50)
RH = dropped(RH, (date.year==2022), 50)
RH_qc = quality_screening(RH, 5, 100, wrong_data_dates, 0, 0)
VPD = dropped(VPD, (date.year==2021)*(date.month>9), -750)
VPD = dropped(VPD, (date.year==2022), -750)
VPD_qc = quality_screening(VPD, 0, 5500, wrong_data_dates, 0, 0)
u_star_qc = quality_screening(u_star, 0, 8, 0, [u_var, v_var, w_var], 0)
L_qc = quality_screening(L, -3000, 3000, 0, 0, 0)
NEE_qc = FCO2_qc #+ co2_strg_qc  
sonic_temperature_qc = quality_screening(sonic_temperature-273.15, -5, 32, 0, 0, 0)
wind_speed_qc = quality_screening(wind_speed, 0, np.nanpercentile(wind_speed, 99.9), 0, 0, 0)

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
Ts1_o = biomet_gap_fill(Ts1_qc, 0, date_biomet, date_era5, 'none')
SWC1_o = biomet_gap_fill(SWC1_qc, 0, date_biomet, date_era5, 'none')
SHF1_o = biomet_gap_fill(SHF1_qc, 0, date_biomet, date_era5, 'none')
Ts2_o = biomet_gap_fill(Ts2_qc, 0, date_biomet, date_era5, 'none')
SWC2_o = biomet_gap_fill(SWC2_qc, 0, date_biomet, date_era5, 'none')
SHF2_o = biomet_gap_fill(SHF2_qc, 0, date_biomet, date_era5, 'none')
Ts3_o = biomet_gap_fill(Ts3_qc, 0, date_biomet, date_era5, 'none')
SWC3_o = biomet_gap_fill(SWC3_qc, 0, date_biomet, date_era5, 'none')
SHF3_o = biomet_gap_fill(SHF3_qc, 0, date_biomet, date_era5, 'none')
# Quality controlled data (Simple screening) - eddy data
Tau_o = biomet_gap_fill(Tau_qc, 0, date, date_era5, 'none')
H_o = biomet_gap_fill(H_qc, 0, date, date_era5, 'none')
LE_o = biomet_gap_fill(LE_qc, 0, date, date_era5, 'none')
FCO2_o = biomet_gap_fill(FCO2_qc, 0, date, date_era5, 'none')
co2_strg_o = biomet_gap_fill(co2_strg_qc, 0, date, date_era5, 'none')
NEE_o = biomet_gap_fill(NEE_qc, 0, date, date_era5, 'none') + co2_strg_o
h2o_flux_o = biomet_gap_fill(h2o_flux_qc, 0, date, date_era5, 'none')
H_strg_o = biomet_gap_fill(H_strg_qc, 0, date, date_era5, 'none')
LE_strg_o = biomet_gap_fill(LE_strg_qc, 0, date, date_era5, 'none')
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
WTD_o = biomet_gap_fill(WTD_qc, 0, date_biomet, date_era5, 'none')
NR01TK_o = biomet_gap_fill(NR01TK_qc, 0, date_biomet, date_era5, 'none')
sonic_temperature_o = biomet_gap_fill(sonic_temperature_qc, 0, date, date_era5, 'none')
wind_speed_o = biomet_gap_fill(wind_speed_qc, 0, date, date_era5, 'none')
FETCH_70 = biomet_gap_fill(FETCH_70, 0, date, date_era5, 'none')
FETCH_MAX = biomet_gap_fill(FETCH_MAX, 0, date, date_era5, 'none')
FETCH_90 = biomet_gap_fill(FETCH_90, 0, date, date_era5, 'none')
#%%
# NEE_f = biomet_gap_fill(NEE_qc, ds['u10'], date, date_era5, 'none')
# ustar_f = biomet_gap_fill(u_star_qc, ds['u10'], date, date_era5, 'linear fill')

#%% Wind Direction from [-180, 180] to [0, 360]
# There is a wind direction correction applied before eddypro to correct azimuth and magnetic declination
WD2 = WD_o.copy()
WD2[WD2<0] = WD2[WD2<0] + 360
from windrose import WindroseAxes
plt.figure()
ax = WindroseAxes.from_ax()
ax.bar(WD2, MWS_o, normed=True, opening=0.8, edgecolor='white')
ax.set_legend()
#%% Footprint correction
ft90 = footprint(FETCH_90, WD2)
ft70 = footprint(FETCH_70, WD2)
ftmax = footprint(FETCH_MAX, WD2)
# data screening using footprint analysis
NEE_o[ft90==1] = np.nan
ET_o[ft90==1] = np.nan
LE_o[ft90==1] = np.nan
#%% Biomet Gapfilling
print('Gapfilling biometeorological data')
print('....................................')
Ta_f = biomet_gap_fill(Ta_qc, ds['t2m']-273.15, date_biomet, date_era5, 'linear fill')
Pa_f = biomet_gap_fill(Pa_qc, ds['sp'], date_biomet, date_era5, 'linear fill')
LWin_f = biomet_gap_fill(LWin_qc, ds['msdwlwrf'], date_biomet, date_era5, 'linear fill') #downward rad
LWout_f = biomet_gap_fill(LWout_qc, ds['msnlwrf']-ds['msdwlwrf'], date_biomet,
                                                      date_era5, 'linear fill') #upward rad
SWin_f = biomet_gap_fill(SWin_qc, ds['msdwswrf'], date_biomet, date_era5, 'replace') #downward rad
SWout_f = biomet_gap_fill(SWout_qc, -ds['msnswrf']+ds['msdwswrf'], date_biomet, date_era5, 'linear fill') #downward rad
H_f = biomet_gap_fill(H_qc, ds['msshf'], date, date_era5, 'linear fill')
SWC1_f = biomet_gap_fill(SWC1_qc, ds['swvl1'], date_biomet, date_era5, 'linear fill')
Ts1_f = biomet_gap_fill(Ts1_qc, ds['stl1'], date_biomet, date_era5, 'linear fill')
SWC2_f = biomet_gap_fill(SWC2_qc, ds['swvl1'], date_biomet, date_era5, 'linear fill')
Ts2_f = biomet_gap_fill(Ts2_qc, ds['stl1'], date_biomet, date_era5, 'linear fill')
SWC3_f = biomet_gap_fill(SWC3_qc, ds['swvl1'], date_biomet, date_era5, 'linear fill')
Ts3_f = biomet_gap_fill(Ts3_qc, ds['stl1'], date_biomet, date_era5, 'linear fill')
# SHF_f = biomet_gap_fill(SHF, ds['msshf']+ds['mslhf'], date_biomet, date_era5, 'linear fill')
RH_f = biomet_gap_fill(RH_qc, VPD_era5, date, date_era5, 'linear fill')  # recalcular usando rh de era5 orig
LE_f = biomet_gap_fill(LE_qc, pd.Series(Convert(ds['mslhf'])), date, date_era5,'linear fill')
Pa_f = biomet_gap_fill(Pa_qc, ds['sp'], date_biomet, date_era5, 'linear fill')
Rn_f = biomet_gap_fill(Rn_qc, ds['msnlwrf']+ds['msnswrf'], date_biomet, date_era5, 'linear fill')
VPD_f = biomet_gap_fill(VPD_qc, VPD_era5, date, date_era5, 'linear fill')
MWS_f = biomet_gap_fill(MWS_qc, np.sqrt(ds['u10']**(2) + ds['v10']**(2)), date_biomet, date_era5, 'replace')
wind_dir = biomet_gap_fill(wind_dir, wd_era5, date, date_era5, 'replace')
NR01TK_f = biomet_gap_fill(NR01TK_qc, ds['t2m'], date_biomet, date_era5, 'linear fill')
wind_speed_f = biomet_gap_fill(wind_speed_qc, np.sqrt(ds['u10']**(2) + ds['v10']**(2)), date, date_era5, 'linear fill')

# P_rain_f = biomet_gap_fill(P_rain_qc, ds['tp']*1000, )


#%% Long wave radiation correction for NR01 sensor
def LW_body_temp_correction(LW, NR01TK):
    delta = 5.67 * 10**(-8)
    e = LW + (delta * (NR01TK)**(4))
    return e

LWin_o = LW_body_temp_correction(LWin_o, NR01TK_f)
LWout_o = LW_body_temp_correction(LWout_o, NR01TK_f)
LWin_f = LW_body_temp_correction(LWin_f, NR01TK_f)
LWout_f = LW_body_temp_correction(LWout_f, NR01TK_f)
NETRAD_o =  LWin_o - LWout_o + SWin_o - SWout_o
NETRAD_f =  LWin_f - LWout_f + SWin_f - SWout_f
#%% Energy Balance Residual Correction
# Not available yet, previous Storage term calculation is needed.
# EBR = energy_balance_ratio(Rn_f>20, H_f, LE_f, Rn_f, SHF_f, J, date)
#%% u_star filtering - (Reichstein et al., 2005) & (papale et al 2006)
print('u* filtering 100 times, to asses the uncertainty of u* threshold - could take some time')
print('....................................')
df = pd.DataFrame({'NEE': NEE_o, 'USTAR':ustar_o, 'TA':Ta_f}, index=date_era5)
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
df = pd.DataFrame({'H': H_f, 'LE': LE_f, 'NEE': NEE_o, 'CO2_strg': co2_strg_o,
                   'ustar': ustar_o, 'SW_IN': SWin_f, 'SHF1': SHF1_o,
                   'SHF2': SHF2_o,'SHF3': SHF3_o,
                   'TA':Ta_f, 'VPD':VPD_f}, index=date_era5)
dff = df.copy(deep=True)  # Flag
dff[:] = 0
dff[df.isna()] = 2; df[df.isna()] = undef
# if available
hfill = ['H', 'LE', 'NEE',
         'SW_IN', 'TA', 'VPD', 'CO2_strg']
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
SHF1_f = var_reading('f_SHF1', df_f, True)
SHF2_f = var_reading('f_SHF2', df_f, True)
SHF3_f = var_reading('f_SHF3', df_f, True)
ustar_f = var_reading('f_ustar', df_f, True)
CO2_strg_f = var_reading('f_CO2_strg', df_f, True)
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
NEE_err = df_err['f_NEE'].iloc[:, 1]
#%% ET filling
ET_f2 = LE_to_ET(LE_f, Ta_f) * 1800  # s -> 30 min
ET_f = ET_o.copy()
ET_f[ET_o.isna()] = ET_f2[ET_o.isna()]
ET_f[ET_f < 0] = 0
plt.plot(ET_f)

#%% Data saving
df_final = pd.DataFrame({'TIMESTAMP': date_era5, 'TA': Ta_o, 'TA_F': Ta_f,
                         'T_SONIC': sonic_temperature_o, 'CO2_mixing_ratio':co2_mix_ratio_o,
                         'H2O_mixing_ratio': h2o_mix_ratio_o,
                         'SC':co2_strg_o, 'SC_F': CO2_strg_f, 
                         'FH2O': h2o_flux_o, 'FC': FCO2_o,
                         'PA': Pa_o, 'PA_F':Pa_f, 'RH': RH_o, 'RH_F': RH_f,
                         'TD': Td_o, 'T_CANOPY': Tc_o, 'NETRAD': NETRAD_o, 'NETRAD_F': NETRAD_f,
                         'LW_IN': LWin_o, 'LW_IN_F': LWin_f, 'LW_OUT': LWout_o,
                         'LW_OUT_F': LWout_f,'SW_IN': SWin_o, 'SW_IN_F': SWin_f,
                         'SW_OUT': SWout_o, 'SW_OUT_F': SWout_f, 'PPFD': PPFD_o,
                         'P_RAIN': P_rain_o, 'WS':wind_speed_o, 'WS_F': wind_speed_f,
                         'WS_MAX': MWS_o, 'WS_MAX_F':MWS_f, 
                         'WD': WD2, 'TS_1_1_1': Ts1_o, 'TS_2_1_1': Ts2_o, 'TS_3_1_1': Ts3_o,
                         'TS_F_1_1_1': Ts1_f,'TS_F_2_1_1': Ts2_f,'TS_F_3_1_1': Ts3_f,
                         'SWC_1_1_1': SWC1_o, 'SWC_2_1_1': SWC2_o, 'SWC_3_1_1': SWC3_o,
                         'SWC_F_1_1_1': SWC1_f, 'SWC_F_2_1_1': SWC2_f,'SWC_F_3_1_1': SWC3_f, 
                         'WTD': WTD_o, 'G_1_1_1': SHF1_o, 'G_2_1_1': SHF2_o, 'G_3_1_1': SHF3_o,
                         'G_F_1_1_1': SHF1_f, 'G_F_2_1_1': SHF2_f, 'G_F_3_1_1': SHF3_f,
                         'LE': LE_o, 'LE_F': LE_f, 'H': H_o,
                         'H_F': H_f, 'NEE': NEE_o, 'NEE_F': NEE_f,
                         'NEE_err': NEE_err,'VPD': VPD_o, 'VPD_F': VPD_f,
                         'USTAR': ustar_o,  'USTAR_F': ustar_f, 'GPP_DT': gpp_l,
                         'GPP_NT': gpp_r, 'ET': ET_o, 'ET_F': ET_f,
                         'RECO_DT': reco_l, 'RECO_NT':reco_r})
path = 'D:/Dropbox/David/LECS/eddy/Bosque/Salida PostProc/'
df_final.to_csv(path+'LEVEL3_SDF_2014_2022.csv', index=False)
print('Saving data to a csv')
print('....................................')


