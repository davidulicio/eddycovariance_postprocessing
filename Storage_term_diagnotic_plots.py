# -*- coding: utf-8 -*-
"""
Storage Term Data Diagnostic
@author: David Trejo
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%% Data reading
folder = '220510'  # Folder with time format YYMMDD
path = 'D:/Dropbox/David/LECS/eddy/Bosque/Storage Term/ST'+folder+'/'
filename_30min = 'CR1000XSeries_2_data_30min.dat'
filename_daily = 'CR1000XSeries_2_data_daily.dat'
data_30min = pd.read_csv(path+filename_30min, skiprows=([0,3]))
data_daily = pd.read_csv(path+filename_daily, skiprows=([0,3]))
units = data_30min.iloc[0]; data_30min = data_30min[251:]  # corté primeros días de ajuste
data_daily = data_daily[6:]
#%% Variables selection
date = pd.DatetimeIndex(data_30min['TIMESTAMP'])
Batt = np.float64(data_30min['BattV_Avg'])
temp1 = np.float64(data_30min['T107_C_Avg'])
temp2 = np.float64(data_30min['T107_C_2_Avg'])
temp3 = np.float64(data_30min['T107_C_3_Avg'])
temp4 = np.float64(data_30min['T107_C_4_Avg'])
co2 = np.float64(data_30min['co2_Avg'])
h2o = np.float64(data_30min['h2o_Avg'])
co2_std = np.float64(data_30min['co2_Std'])
h2o_std = np.float64(data_30min['h2o_Std'])

#%% 30 min data plots
fig, axs = plt.subplots(2, 2, figsize=(20,8)); axs = axs.flatten()
# Battery Volts
axs[0].plot(date, Batt)
axs[0].set_ylabel(units['BattV_Avg'])
axs[0].set_title('Battery Volts Average')
# Temperatures
axs[1].plot(date, temp1, label='Temp 1')
axs[1].plot(date, temp2, label='Temp 2')
axs[1].plot(date, temp3, label='Temp 3')
axs[1].plot(date, temp4, label='Temp 4')
axs[1].set_ylabel(units['T107_C_Avg'])
axs[1].set_title('Temperatures')
axs[1].legend()
# CO2
# axs[2].plot(date, co2, '-.')
axs[2].errorbar(date, co2, yerr=co2_std, ecolor='k')
# axs[2].legend()
axs[2].set_title('$CO_2$')
axs[2].set_ylabel(units['co2_Avg'])
# H2O
axs[3].errorbar(date, h2o, yerr=h2o_std, ecolor='k')
axs[3].set_title('$H_2 $O')
axs[3].set_ylabel(units['h2o_Avg'])
plt.tight_layout()
plt.suptitle('30-min data')
plt.savefig(path+'30min_data_'+folder+'.png', dpi=100)
#%% Daily Variables selection
date = pd.DatetimeIndex(data_daily['TIMESTAMP'])
Batt = np.float64(data_daily['BattV_Avg'])
temp1 = np.float64(data_daily['T107_C_Avg'])
temp2 = np.float64(data_daily['T107_C_2_Avg'])
temp3 = np.float64(data_daily['T107_C_3_Avg'])
temp4 = np.float64(data_daily['T107_C_4_Avg'])
co2 = np.float64(data_daily['co2_Avg'])
h2o = np.float64(data_daily['h2o_Avg'])
co2_std = np.float64(data_daily['co2_Std'])
h2o_std = np.float64(data_daily['h2o_Std'])

#%% Daily data plots
fig, axs = plt.subplots(2, 2, figsize=(20,8)); axs = axs.flatten()
# Battery Volts
axs[0].plot(date, Batt)
axs[0].set_ylabel(units['BattV_Avg'])
axs[0].set_title('Battery Volts Average')
# Temperatures
axs[1].plot(date, temp1, label='Temp 1')
axs[1].plot(date, temp2, label='Temp 2')
axs[1].plot(date, temp3, label='Temp 3')
axs[1].plot(date, temp4, label='Temp 4')
axs[1].set_ylabel(units['T107_C_Avg'])
axs[1].set_title('Temperatures')
axs[1].legend()
# CO2
# axs[2].plot(date, co2, '-.')
axs[2].errorbar(date, co2, yerr=co2_std, ecolor='k')
# axs[2].legend()
axs[2].set_title('$CO_2$')
axs[2].set_ylabel(units['co2_Avg'])
# H2O
axs[3].errorbar(date, h2o, yerr=h2o_std, ecolor='k')
# plt.fill_between(xfit, yfit - dyfit, yfit + dyfit,
                 # color='gray', alpha=0.2)
axs[3].set_title('$H_2 $O')
axs[3].set_ylabel(units['h2o_Avg'])
plt.tight_layout()
plt.suptitle('Daily data')
plt.savefig(path+'daily_data_'+folder+'.png', dpi=100)

# %% functions
class Storage_Term:
    def _init_(self):
        self.Rd = 287.058  # Specific gas constant for dry air [J kg-1 K-1]
        
    def p_dry(self, pd, T):
        """
        Calculates the density of dry air given the atmospheric conditions
    
        Parameters
        ----------
        pd : array
            Partial pressure of dry air [Pa].
        T : array
            Average of air temperature from the vertical profile [°C].
    
        Returns
        -------
        p_dry : array
            Dry air density [kg m-3].
    
        """
        
        T = T + 273.15  # °C -> K
        self.p_dry = pd / (self.Rd * T)
        
    
    def Jc(self, co2):
        """
        Calculates The storage flux term of a given scalar (co2)
    
        Parameters
        ----------
        p_dry : array
            Dry air density [kg m-3].
        co2 : array
            Integrated CO2 concentration in the vertical profile.
    
        Returns
        -------
        storage_term : TYPE
            Storage flux term (Jc).
    
        """
        # Check units
        self.storage_term = self.p_dry * co2
        return self.storage_term