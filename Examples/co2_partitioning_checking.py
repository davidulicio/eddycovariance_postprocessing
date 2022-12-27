# -*- coding: utf-8 -*-
"""
GPP, Reco and NEE
@author: David
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy.stats import sem
import datetime

#%% Data reading
path = 'D:/Dropbox/David/LECS/eddy/Turbera/Salida PostProc/'
file = 'LEVEL3_SDP_2014_2021_2.csv'
Ecosystem = 'Peatland'
data = pd.read_csv(path+file)
valid_years = [0, 1, 2, 3, 4, 5, 7]  # Peatland
# valid_years = [0, 2, 3, 4, 5, 6]  # Forest
#%% Conversion units
umols_to_dailygC = (12 * 10**(-6))* 86400  #micromol/m2s a gramos de carbono/m2dia
umols_to_yearlygC = (12 * 10**(-6))* 86400 *365  #micromol/m2s a gramos de carbono/m2año
umols_to_30mingC = (12 * 10**(-6)) * 1800  #micromol/m2s a gramos de carbono/m2 30min
#%% Data selection
# CO2 fluxes
nee = np.float64(data['NEE_f']) * umols_to_30mingC
nee_err = np.float64(data['NEE_err']) * umols_to_30mingC
nee_err[nee_err<-200] = np.nan  # removes non valid data
gpp_dt = np.float64(data['GPP_DT']) * umols_to_30mingC
gpp_nt = np.float64(data['GPP_NT']) * umols_to_30mingC
reco_dt = np.float64(data['Reco_DT']) * umols_to_30mingC; reco_dt[np.isnan(nee)] = np.nan
reco_nt = np.float64(data['Reco_NT']) * umols_to_30mingC; reco_nt[np.isnan(nee)] = np.nan
dates = pd.DatetimeIndex(data['TIMESTAMP'])
# Water fluxes
pp = np.float64(data['P_rain'])
ET = np.float64(data['ET_f']); ET[pp>0] = np.nan
WUE = pd.Series(gpp_dt / ET, index=dates);
valid = np.isfinite(WUE)
WUE = WUE[valid]
WUE[(WUE < 0) | (WUE > np.nanpercentile(WUE, 99))] = np.nan
vpd = np.float64(data['VPD_f']) / 100  # Pa -> hPa
IWUE = WUE * vpd[valid]
path = 'D:/Dropbox/David/LECS/eddy/'
gs = pd.read_csv(path+'peatland_gs.csv')
date2 = pd.DatetimeIndex(gs['TIMESTAMP'])
gs_mol = np.float64(gs[Ecosystem+' Gs mol'])
iwue = pd.Series(np.float64(data['GPP_DT']) / (gs_mol), index = date2) # GPP en unidades originales
# iwue[pp>0] = np.nan
valid = np.isfinite(iwue)
iwue = iwue[valid]; #
#%% 30 min plots
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8, 6))
axs[0].plot(dates, nee, label='NEE', alpha=0.6)
axs[0].plot(dates, gpp_dt, label='GPP', alpha=0.6)
axs[0].plot(dates, reco_dt, label='Reco', alpha=0.6)
axs[0].set_title(Ecosystem+' SD. CO2 Lasslop partitioning - 30 min data')
axs[0].legend(frameon=False, ncol=3, loc='upper left')
axs[0].grid()
###
axs[1].plot(dates, nee, alpha=0.6)
axs[1].plot(dates, gpp_nt, alpha=0.6)
axs[1].plot(dates, reco_nt, alpha=0.6)
axs[1].set_title(Ecosystem+' SD. CO2 Reichtein partitioning - 30 min data')
axs[1].grid()
plt.tight_layout()

"""
Para calcular acumulaciones considerar rellenar con promedios de todos los años.

Evaluar sistema de relleno de long-gaps de korea-flux
"""
#%% extreme values filtering
wrong_data = (IWUE>pd.Series.quantile(IWUE, 0.995))
# wrong_data = wrong_data + (date2.year==2016) & (Iwue_daily_b>30)
# wrong_data = wrong_data + (date2.year==2021) & (Iwue_daily_b>20)
IWUE[wrong_data] = np.nan
iwue[(iwue < 0) | (iwue > np.nanpercentile(iwue, 99))] = np.nan
iwue[iwue>2000] = np.nan
# wrong_data = (Iwue_daily_t>pd.Series.quantile(Iwue_daily_t, 0.99))
# wrong_data = wrong_data + (Iwue_daily_b.index.year==2016) & (iwue_daily_b>40)
# Iwue_t[wrong_data] = np.nan
#%%
# wrong_data = (iwue_daily_t>pd.Series.quantile(iwue_daily_t, 0.95))
# iwue_daily_t[wrong_data] = np.nan
# wrong_data = (iwue_daily_b.index.year==2016) & (iwue_daily_b>40)
# Iwue_daily_b[wrong_data] = np.nan
# wrong_data = wrong_data + (iwue_daily_b.index.year==2021) & (iwue_daily_b>40)
# iwue_daily_b[wrong_data] = np.nan

#%% Aggregations Lasslop partitioning
daily_gpp_dt = pd.Series(gpp_dt, index=dates).resample('D').agg(pd.Series.sum,
                            min_count=30) #* conc_30min_to_sec
daily_reco_dt = pd.Series(reco_dt, index=dates).resample('D').agg(pd.Series.sum,
                            min_count=30)  #* conc_30min_to_sec
daily_nee = pd.Series(nee, index=dates).resample('D').agg(pd.Series.sum,
                            min_count=30) #* umols_to_30mingC
yearly_gpp_dt = pd.Series(gpp_dt, index=dates).resample('Y').agg(pd.Series.sum,
                            skipna=True) #* conc_30min_to_sec
yearly_reco_dt = pd.Series(reco_dt, index=dates).resample('Y').agg(pd.Series.sum,
                            skipna=True) #* conc_30min_to_sec
yearly_nee = pd.Series(nee, index=dates).resample('Y').agg(pd.Series.sum,
                            skipna=True)
# Error propagation
nee_err = pd.Series(nee_err, index=dates)
yearly_nee_err = np.zeros(len(yearly_nee)) * np.nan
for i in range(len(yearly_nee)):
    yr = yearly_nee.index.year[i]
    valid = dates.year == yr
    yearly_nee_err[i] = np.sqrt(np.nansum(nee_err[valid]**(2)))
yearly_nee_err = pd.Series(yearly_nee_err, index=yearly_nee.index)
# %% Aggregations Reichtein partitioning
daily_gpp_nt = pd.Series(gpp_nt, index=dates).resample('D').agg(pd.Series.sum,
                            min_count=30) 
daily_reco_nt = pd.Series(reco_nt, index=dates).resample('D').agg(pd.Series.sum,
                            min_count=30) 
yearly_gpp_nt = pd.Series(gpp_nt, index=dates).resample('Y').agg(pd.Series.sum,
                            skipna=True)
yearly_reco_nt = pd.Series(reco_nt, index=dates).resample('Y').agg(pd.Series.sum,
                            skipna=True)
#%% Aggregations of water fluxes
yearly_gs = pd.Series(gs_mol, index=date2).resample('Y').agg(pd.Series.mean,
                                                                skipna=True)
yearly_et = pd.Series(ET, index=dates).resample('Y').agg(pd.Series.sum,
                                                                skipna=True)
# WUE[WUE>np.nanpercentile(WUE, 99)] = np.nan
yearly_wue = WUE.resample('Y').agg(pd.Series.mean,
                                   skipna=True)
yearly_Iwue = IWUE.resample('Y').agg(pd.Series.mean,
                                   skipna=True)
yearly_iwue = iwue.resample('Y').agg(pd.Series.mean,
                                   skipna=True)
#%% uWUE calculation
# functions
def quantreg(x,y,PolyDeg=1,rho=0.95,weights=None):
    '''quantreg(x,y,PolyDeg=1,rho=0.95)

    Quantile regression

    Fits a polynomial function (of degree PolyDeg) using quantile regression based on a percentile (rho).
    Based on script by Dr. Phillip M. Feldman, and based on method by Koenker, Roger, and
    Gilbert Bassett Jr. “Regression Quantiles.” Econometrica: Journal of
    the Econometric Society, 1978, 33–50.


    Parameters
    ----------
    x : list or list like
        independent variable
    y : list or list like
        dependent variable
    PolyDeg : int
        The degree of the polynomial function
    rho : float between 0-1
        The percentile to fit to, must be between 0-1
    weights : list or list like
        Vector to weight each point, must be same size as x

     Returns
    -------
    list
        The resulting parameters in order of degree from low to high
    '''
    def model(x, beta):
       """
       This example defines the model as a polynomial, where the coefficients of the
       polynomial are passed via `beta`.
       """
       if PolyDeg == 0:
           return x*beta
       else:
           return np.polyval(beta, x)

    N_coefficients=PolyDeg+1

    def tilted_abs(rho, x, weights):
       """
       OVERVIEW

       The tilted absolute value function is used in quantile regression.


       INPUTS

       rho: This parameter is a probability, and thus takes values between 0 and 1.

       x: This parameter represents a value of the independent variable, and in
       general takes any real value (float) or NumPy array of floats.
       """

       return weights * x * (rho - (x < 0))

    def objective(beta, rho, weights):
       """
       The objective function to be minimized is the sum of the tilted absolute
       values of the differences between the observations and the model.
       """
       return tilted_abs(rho, y - model(x, beta), weights).sum()

    # Build weights if they don't exits:
    if weights is None:
        weights=np.ones(x.shape)

    # Define starting point for optimization:
    beta_0= np.zeros(N_coefficients)
    if N_coefficients >= 2:
       beta_0[1]= 1.0

    # `beta_hat[i]` will store the parameter estimates for the quantile
    # corresponding to `fractions[i]`:
    beta_hat= []

    #for i, fraction in enumerate(fractions):
    beta_hat.append( fmin(objective, x0=beta_0, args=(rho,weights), xtol=1e-8,
      disp=False, maxiter=3000) )
    return(beta_hat)

# Usage
et_daily = pd.Series(ET, index=dates).resample('D').agg(pd.Series.sum, skipna=False)
et_daily[et_daily==0] = np.nan
gpp_daily =  pd.Series(gpp_dt, index=dates).resample('D').agg(pd.Series.sum, skipna=False)
vpd_daily = pd.Series(vpd, index=dates).resample('D').agg(pd.Series.mean, skipna=True)
vpd_daily[vpd_daily==0] = np.nan
gxv = gpp_daily * np.sqrt(vpd_daily)
mask = (gpp_daily<0.1*np.nanpercentile(gpp_daily, 95))
uWUEp = quantreg(et_daily, gxv, PolyDeg=0, rho=0.95)[0][0]
uWUEa = gxv / et_daily
uWUEa[mask] = np.nan
transp = (uWUEa/uWUEp)*et_daily
for k in range(len(transp)):
    if transp[k] > et_daily[k]:
        et_daily[k] = np.nan
        transp[k] = np.nan
evap = et_daily - transp
###
uWUEa[uWUEa>np.nanpercentile(uWUEa, 99)] = np.nan
#%%
yearly_uWUEa = pd.Series(uWUEa, index=et_daily.index).resample('Y').agg(pd.Series.mean, skipna=True)

#%% Means and errors
# CO2 fluxes
nee_mean = np.nanmean(yearly_nee[valid_years])
nee_se = sem(yearly_nee[valid_years])
reco_mean = np.nanmean(yearly_reco_dt[valid_years])
reco_se = sem(yearly_reco_dt[valid_years])
gpp_mean = np.nanmean(yearly_gpp_dt[valid_years])
gpp_se = sem(yearly_gpp_dt[valid_years])
err_mean = np.nanmean(yearly_nee_err[valid_years])
err_se = sem(yearly_nee_err[valid_years])
# Water fluxes
gs_mean = np.nanmean(yearly_gs[valid_years])
gs_se = sem(yearly_gs)
et_mean = np.nanmean(yearly_et[valid_years])
et_se = sem(yearly_et[valid_years])
wue_mean = np.nanmean(yearly_wue[valid_years])
wue_se = sem(yearly_wue[valid_years])
Iwue_mean = np.nanmean(yearly_Iwue[valid_years])
Iwue_se = sem(yearly_Iwue[valid_years])
iwue_mean = np.nanmean(yearly_iwue[valid_years])
iwue_se = sem(yearly_iwue[valid_years])
uwue_mean = np.nanmean(yearly_uWUEa[valid_years])
uwue_se = sem(yearly_uWUEa[valid_years])
#%%
print('CO2 Fluxes statistics')
print('''### NEE gC m-2 yr-1 ###
      '''+str(yearly_nee)); print('Su prom ='+str(nee_mean)+', su SE = '+ str(nee_se))
print('''### NEE error gC m-2 yr-1 ###
      '''+str(yearly_nee_err)); print('Su prom ='+str(err_mean)+', su SE = '+ str(err_se))
print('''### Reco gC m-2 yr-1 ###
      '''+str(yearly_reco_dt)); print('Su prom ='+str(reco_mean)+', su SE = '+ str(reco_se))
print('''### GPP gC m-2 yr-1 ###
      '''+str(yearly_gpp_dt)); print('Su prom ='+str(gpp_mean)+', su SE = '+ str(gpp_se))
print('#########################################')
print('Water Fluxes Statistics')
print('''### ET kg H2O m-2 y-1 ###
      '''+str(yearly_et)); print('Su prom ='+str(et_mean)+', su SE = '+ str(et_se))
print('''### WUE g C kg-1 H2O ###
      '''+str(yearly_wue)); print('Su prom ='+str(wue_mean)+', su SE = '+ str(wue_se))
print('''### IWUE g C hPa kg-1 H2O ###
      '''+str(yearly_Iwue)); print('Su prom ='+str(Iwue_mean)+', su SE = '+ str(Iwue_se))
print('''### Gs mol m-2 s-1 ###
      '''+str(yearly_gs)); print('Su prom ='+str(gs_mean)+', su SE = '+ str(gs_se))
print('''### iWUE μmol CO2 mol-1 H2O ###
      '''+str(yearly_iwue)); print('Su prom ='+str(iwue_mean)+', su SE = '+ str(iwue_se))
print('''### uWUE g C hPa0.5  kg-1 H2O ###
      '''+str(yearly_uWUEa)); print('Su prom ='+str(uwue_mean)+', su SE = '+ str(uwue_se))
#%% Comparison paper 2018
fi = datetime.datetime(2014, 1, 1, 00, 00)
ff = datetime.datetime(2015, 9, 30, 00, 00)
dates_mask = (daily_nee.index > fi) * (daily_nee.index < ff)
#%%
plt.figure(figsize=(10,6))
# plt.grid()
plt.plot(daily_nee, lw=.7, color='k', label='NEE')
plt.plot(daily_gpp_dt, lw=.7, color='darksalmon', label='GPP')
plt.plot(daily_reco_dt, lw=.7, color='darkcyan', label='Reco')
plt.ylabel('g C $m^{-2}$ $day^{-1}$')
plt.title(Ecosystem)
plt.legend(ncol=3)
#%% 
plt.figure(figsize=(10,6))
et_daily = pd.Series(ET, index=dates).resample('D').agg(pd.Series.sum, skipna=True)
plt.plot(et_daily)
plt.ylabel('mm day-1')
plt.title(Ecosystem)

# plt.title('NEE at Forest SD')
# plt.plot(nee)