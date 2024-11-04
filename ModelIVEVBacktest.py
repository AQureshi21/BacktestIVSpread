# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 08:44:09 2024

@author: aqureshi
"""
# For Weekdays
import numpy as np
import scipy as sp
import pandas as pd
import schedule
import matplotlib as plt
import matplotlib.pyplot as plt
import datetime
from datetime import date, timedelta
import time
import requests

from PropTradeSpreadClass import *
from PropTradeClassIV import *
from PullLiveDataClass import *

PTC = PropTradeLiveClassIV()
PTS = PropTradeSpreadClass('01/01/2023', '29/02/2024')

#Mkt Conditions
train_solar = PTS.GetData(PTS.GetSolarData())[0]
test_solar = PTS.GetData(PTS.GetSolarData())[1]

train_wind = PTS.GetData(PTS.GetWindData())[0]
test_wind = PTS.GetWindFcastData()

rd = PTS.GetRDData()
train_rd = rd.iloc[:8, :]
test_rd = rd.iloc[8:, :]

ic = PTC.IC
train_ic = ic.iloc[:1884, :]
test_ic = ic.iloc[1884:, :]
test_ic.index = range(len(test_ic))

test_apx = PTS.GetData(PTS.DA_APX)[1]
test_cashout = PTS.GetData(PTS.Cashout)[1]
actual_n2ex = PTS.GetData(PTS.DA)[1]

percentile_vals = PTS.percentile_array + [1, 99]
percentile_vals.sort()

closest_sell_percentiles = test_solar.copy().iloc[:, :26]
closest_sell_percentiles.iloc[:, :] = 0

closest_buy_percentiles = test_solar.copy().iloc[:, :26]
closest_buy_percentiles.iloc[:, :] = 0

sell_err = closest_sell_percentiles.copy()
buy_err = closest_buy_percentiles.copy()

d = 1
while d < len(test_solar)-1:
    date = test_apx.iloc[d, 0]
    today_apx = test_apx.iloc[d, 1:][::2]
    today_c = test_cashout.iloc[d, 1:]
    today_hourly_c = np.asarray([np.mean(today_c[2*h:2*(h)+2]) for h in range(len(today_c[::2]))])
    n2ex = actual_n2ex.iloc[d, 1:][::2]
    
    rd_loc = np.where(test_rd.Date == test_solar.iloc[d, 0])[0]
    if len(rd_loc) > 0:
        today_rd = np.asarray(test_rd.iloc[rd_loc, 1:])[0]
    else:
        today_rd = np.asarray(test_rd.iloc[0, 1:])
    
    percentile_analysis = pd.DataFrame(np.zeros((24, 26)))
    if (test_solar.iloc[d, 0] == test_wind.iloc[:, 0]).any():        
        test_wind_data = test_wind.iloc[np.where(test_solar.iloc[d, 0] == test_wind.iloc[:, 0])[0][0], 1:]
        fcast = PTC.GetForecastBacktest(date, test_cashout.iloc[d-1, 1:], test_apx.iloc[d, 1:], test_ic.iloc[d, 1:], today_rd)[48:, :]
        
        percentiles = pd.DataFrame([np.nanpercentile(fcast[::2], p, axis = 1) for p in percentile_vals])
        percentiles.index = percentile_vals
        percentiles = percentiles.T
        
        percentiles.insert(0, 'Date', [test_solar.iloc[d, 0]]*24)
        percentiles.insert(1, 'Hour', range(1, 25))
        percentiles.insert(2, 'EPEX', np.asarray(today_apx))
        percentiles.insert(3, 'SIP', today_hourly_c)
        percentiles.insert(4, 'N2EX', np.asarray(n2ex))
        
        sell = np.where(today_apx > today_hourly_c)[0]
        for s in sell:
            percentiles.iloc[s, 15:] = 0
            
        buy = np.where(today_apx < today_hourly_c)[0]
        for b in buy:
            percentiles.iloc[b, 4:15] = 0
                        
    if d > 1:
        percentiles_analysis = pd.concat([percentiles_analysis, percentiles], axis = 0) 
    else: 
        percentiles_analysis = percentiles
                    
    train_wind = pd.concat([train_wind, test_wind.iloc[d:d+1, :]], axis = 0)      
    train_solar = pd.concat([train_solar, test_solar.iloc[d:d+1, :]], axis = 0)
    train_rd = pd.concat([train_rd, test_rd.iloc[d:d+1, :]], axis = 0)
    PTS = PropTradeSpreadClass('01/01/2023', test_apx.iloc[d, 0])
    d = d + 1
