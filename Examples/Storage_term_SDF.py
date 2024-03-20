# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:52:52 2022

@author: PC5_LECS
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.ensemble import BaggingRegressor
#%% Data reading - Measured Storage
PATH = 'D:/Dropbox/David/LECS/eddy/Bosque/Storage Term/'
FILENAME = 'CR1000XSeries_data_30min.csv'
df = pd.read_csv(PATH+FILENAME)
units = df.iloc[0]; df = df[2:]
date = pd.DatetimeIndex(df['TIMESTAMP'])
df.index = date
mask = date.year<2023
df = df[mask]
date = pd.DatetimeIndex(df['TIMESTAMP'])

df = df.astype(float, errors='ignore')
#%% Data reading - Estimated Storage
path = 'D:/Dropbox/David/LECS/eddy/Bosque/Salida PostProc/'
df2 = pd.read_csv(path+'LEVEL3_SDF_2014_2022.csv')
df2.index = pd.DatetimeIndex(df2.TIMESTAMP)

#%% Functions

def read(df, date, colname):
    var = np.float64(df[colname])
    var = pd.Series(var, index=date)
    return var

def air_density(TA, PA):
    """air_density(TA, PA)
    Air density of moist air from air temperature and pressure.
    rho = PA / (Rd * TA)
    Parameters
    ----------
    TA : list or list like
        Air temperature (deg C)
    PA : list or list like
        Atmospheric pressure (kPa)
    Returns
    -------
    rho : list or list like
        air density (g m-3)
    References
    ----------
    - Foken, T, 2008: Micrometeorology. Springer, Berlin, Germany.
    """
    # kPa2Pa = 1000  # conversion kilopascal (kPa) to pascal (Pa)
    Rd = 8.31432  # gas constant of dry air (J K-1 mol-1) (Foken 2008 p. 245)
    Kelvin = 273.15  # conversion degree Celsius to Kelvin
    TA = TA + Kelvin
    # PA = PA * kPa2Pa
    rho = PA / (Rd * TA)
    return (rho)

def storage(C, t, TA, PA, hm):
    dc = (np.array(C[1:]) - np.array(C[0:len(C)-1]))  # Delta umol mol-1
    dc_dt = dc / 1800  # umol mol-1 s-1
    storage = dc_dt * air_density(TA[1:], PA[1:]) * hm
    return storage


def ml_gapfilling(df1, df2, method, test_size):
    """
    Machine learning techniques to gapfilling timeseries using one or multiple 
    features as input. 

    Parameters
    ----------
    df1 : DataFrame
        DataFrame with its columns as the features for training and prediction.
        The index is the timestamp of the timeseries.
    df2 : DataFrame, Series
        Values of the variable to be predicted. Index must be the timestamp of 
        the timeseries.
    method : string
        String of the method to apply. The methods available are linear
        regression, 

    Returns
    -------
    prediction : Series.
        Series of the the predicted values using the method selected and given
        inputs.
    
    """
    date = df1.index.intersection(df2.index)
    df1 = df1.loc[date]; df2 = df2.loc[date]
    valid_data = ~((df1.isna().sum(axis=1)>0) + (df2.isna()))
    df1 = df1[valid_data]; df2 = df2[valid_data]
    X_train, X_test, y_train, y_test = train_test_split(df1, df2, test_size=0.2, random_state=42)
    if method == 'linear':
        reg = linear_model.LinearRegression()
    elif method == 'svm':
        reg = svm.SVR()
    elif method == 'kneighbors':
        reg = neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform')
    elif method == 'scalar':
        reg = make_pipeline(StandardScaler(),linear_model.SGDRegressor(max_iter=1000, tol=1e-3))
    elif method == 'tree':
        reg = tree.DecisionTreeRegressor()
    elif method == 'bagging':
        reg = BaggingRegressor(random_state=42)
    reg.fit(X_train, y_train)
    score_train = str(round(reg.score(X_train, y_train), 2))
    score_test = str(round(reg.score(X_test, y_test), 2))

    print("""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%""")
    print("The gapfilling method selected is: "+ method)
    print('The Test size is: '+str(test_size) + ". The Train size is: " +
          str(1-test_size))
    print("The Train Sample Score for the selected method is: "+ score_train)
    print("The Test Sample Score for the selected method is: "+ score_test)
    prediction = reg.predict(df1)
    prediction = pd.Series(prediction, index=df1.index)
    df3 = df2.copy()
    df3[df3.isna()] = prediction[df3.isna()]
    # Figure
    fig, axs = plt.subplots(3, 1, figsize=(8,8), sharex=True)
    axs[0].plot(df2)
    axs[0].set_title('Raw Data')
    axs[1].plot(prediction)
    axs[1].set_title('Model')
    axs[2].plot(df3)
    axs[2].set_title('Filled Data')
    
    return prediction
#%% Constants
umols_to_30mingC = (12 * 10**(-6)) * 1800  #micromol/m2s a gramos de carbono/m2 30min
#%% Variables
# Storage Variables
ta1 = df.T107_C_Avg.astype(float)
ta2 = df.T107_C_2_Avg.astype(float)
ta3 = df.T107_C_3_Avg.astype(float)
ta4 = df.T107_C_4_Avg.astype(float)
ta = pd.concat([ta1, ta2, ta3, ta4], axis=1).mean(axis=1)
co2 = read(df, date, 'co2_Avg')
# Eddy Variables
TA = df2.TA_F.astype(float)[date.intersection(df2.index)][1:]
PA = df2.PA_F.astype(float)[date.intersection(df2.index)][1:]
# SC = df2.SC_F.astype(float)[date.intersection(df2.index)].resample('D').sum()
#%% measured storage
hm = 22  # height
sco2_1 = storage(co2, date, ta, PA, hm) * umols_to_30mingC
sco2_1[sco2_1<np.nanpercentile(sco2_1, 1)] = np.nan 
sco2_1[sco2_1>np.nanpercentile(sco2_1, 99)] = np.nan
sco2_1 = sco2_1.resample('D').sum()
#%%#%% estimated storage
h = 40 - 22  # m
nee_f = df2.NEE_F * umols_to_30mingC
sc_f = df2.SC_F * umols_to_30mingC
# fc = df2.FC
fc = (nee_f - sc_f) * umols_to_30mingC
sco2_2 = storage(fc[date.intersection(df2.index)],
                 fc[date.intersection(df2.index)].index,
                 ta, PA, h).resample('D').sum()
sco2 = sco2_1 + sco2_2
sc_f = sc_f[date.intersection(df2.index)]
nee_f = nee_f[date.intersection(df2.index)]
#%%
onepoint_agg = sc_f.sum()
profile_agg = sco2.sum()
fco2_agg = nee_f.sum()
onepoint_porc = fco2_agg* onepoint_agg /100
profile_porc = fco2_agg * profile_agg/100
#%%
print(onepoint_agg)
print(profile_agg)
print(fco2_agg)
print(onepoint_porc)
print(profile_porc)
#%%
#%%
# SWin = df2['SW_IN_F']
# LWout = df2['LW_OUT_F']
# MWS = df2['WS_MAX_F']
# SWC = pd.concat([df2['SWC_F_1_1_1'], df2['SWC_F_2_1_1'], df2['SWC_F_3_1_1']], axis=1).mean(axis=1)
# Ta = df2['TA_F']
# Ts = pd.concat([df2['TS_F_1_1_1'], df2['TS_F_2_1_1'], df2['TS_F_3_1_1']], axis=1).mean(axis=1)
# RH = df2['RH_F']
# SHF = pd.concat([df2['G_F_1_1_1'], df2['G_F_2_1_1'], df2['G_F_3_1_1']], axis=1).mean(axis=1)
# Rn = df2['NETRAD_F']
# SC = df2['SC_F']
# USTAR = df2['USTAR_F']

# X = pd.concat([SC, USTAR, SWin, MWS, LWout, SWC, Ts, SHF, Rn, Ta], axis=1)
# X_daily = X.resample('D').mean()
# daily_pred_storage = ml_gapfilling(X, sco2, 'bagging', 0.3)
# y = sco2
# z = pd.concat([X, sco2], axis=1).corr(method='pearson')#; print(z)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%% date intersections
# sc_torre = sc_f[date.intersection(sc_f.index)].resample('D').sum()
# nee_torre = nee_f[date.intersection(sc_f.index)].resample('D').sum()
# fc = fc[date.intersection(fc.index)].resample('D').sum()
# fc_f = nee_torre - sc_torre.resample('D').sum()
#%%
# fig, axs = plt.subplots(5, 1, figsize=(8,18), sharey=True, sharex=True)
# axs[0].plot(fc_f)
# axs[0].set_title('CO2 Flux')
# axs[1].plot(sc_torre)
# axs[1].set_title('Storage estimated from top IRGA')
# axs[2].plot(sco2)
# axs[2].set_title('Storage estimated from integral')
# axs[3].plot(fc_f+sco2)
# axs[3].set_title('NEE estimated from integral')
# axs[4].plot(nee_torre)
# axs[4].set_title('NEE estimated from top IRGA')
# # Set Parameters
# axs[0].grid()
# axs[1].grid()
# axs[2].grid()
# axs[3].grid()
# axs[4].grid()
# #%%
# plt.figure()
# plt.plot(sco2_1 - sco2)

