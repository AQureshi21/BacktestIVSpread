# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 08:39:55 2023

@author: aqureshi
"""
import numpy as np
import scipy as sp
import pandas as pd
import datetime as dt
from datetime import date, timedelta
from scipy.stats import ks_2samp
from scipy import stats
#from sklearn import linear_model

from KSTestClass import *
#from ADFTestClass import *
from SeasonalityClass import *
from CorrelationClass import *
from CorrelatedStochasticProcessClass import *
from OrnsteinUhlenbeckRegressorClass import *
from DayAheadClass import *
from PPAImbalanceClass import *
from ShortTermLoadForecastClass22 import *
from PullLiveDataClass import *

class PropTradeLiveClassIV:
    
    def __init__(self):
        self.sims = 1000
        self.percentile_array = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
        self.hours = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]
        self.EFA_1 = [46, 47] + list(range(6))
        self.EFA_2 = list(range(6, 14))
        self.EFA_3 = list(range(14, 22))
        self.EFA_4 = list(range(22, 30))
        self.EFA_5 = list(range(30, 38))
        self.EFA_6 = list(range(38, 46))
        self.EFA = [self.EFA_1, self.EFA_2, self.EFA_3, self.EFA_4, self.EFA_5, self.EFA_6]
        
        self.quarters = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        
        self.DA = pd.read_csv(r'C:\Users\aqureshi\OneDrive - CF Partners\Documents\Triads\ShortTermTrade\DayAheadhh.csv')
        self.DA_APX = pd.read_csv(r'C:\Users\aqureshi\OneDrive - CF Partners\Documents\Triads\ShortTermTrade\DayAheadAPXhh.csv')
        self.Cashout = pd.read_csv(r'C:\Users\aqureshi\OneDrive - CF Partners\Documents\Triads\ShortTermTrade\Cashouthh.csv')
        
        self.wind_gen_data = pd.read_csv(r'C:\Users\aqureshi\OneDrive - CF Partners\Documents\Triads\WindHHNoLeap22.csv')
        self.temperature_data = pd.read_csv(r'C:\Users\aqureshi\OneDrive - CF Partners\Documents\Triads\ShortTermTrade\TemperatureSENoLeap22.csv')
        self.gas_data = pd.read_csv(r'C:\Users\aqureshi\OneDrive - CF Partners\Documents\Triads\ShortTermTrade\GasData.csv')
        self.IC = pd.read_csv(r'C:\Users\aqureshi\OneDrive - CF Partners\Documents\Triads\ShortTermTrade\InterconnectorScheduledDA.csv')

    def ExtractTrend(self, data_list):
        PPA = PPAImbalanceClass()
        
        trendless = pd.DataFrame()
        trend = []
        for i in range(len(data_list)):
            remove_trend = PPA.RemoveTrends(data_list[i])[0]
            trendless = pd.concat([remove_trend, trendless])
            trendless.index = range(len(trendless))
            
            trend.append(PPA.RemoveTrends(data_list[i])[1][-1])

        return trendless, trend
        
    def ConvertHourlySeasonalityHH(self, da_trendless, da_seasonal_data):
        da_seasonality = np.zeros(da_trendless.iloc[:, 1:].shape)
        for i in range(len(self.hours[1:])):
            da_seasonality[:, 2*i] = da_seasonal_data[0][:, i]
            da_seasonality[:, 2*i+1] = da_seasonal_data[0][:, i]
        return da_seasonality
    
    def ConvertHourlyDeseasonalisedHH(self, da_trendless, da_seasonal_data):
        da_deseasonalised = np.zeros(da_trendless.iloc[:, 1:].shape)
        for i in range(len(self.hours[1:])):
            da_deseasonalised[:, 2*i] = da_seasonal_data[1].iloc[:, i+1]
            da_deseasonalised[:, 2*i+1] = da_seasonal_data[1].iloc[:, i+1]
        da_deseasonalised = pd.DataFrame(da_deseasonalised)
        da_deseasonalised.insert(0, 'Date', da_trendless.Date)
        da_deseasonalised.columns = da_trendless.columns.tolist()
        return da_deseasonalised
    
    def MonthData(self, combined_noise):
        PPA = PPAImbalanceClass()
        
        training_data_month = [dt.datetime.strptime(date , '%d/%m/%Y').month for date in combined_noise.Date]
        m = dt.datetime.today().date().month
        truncate_data = PPA.TruncateNoise(training_data_month, m, combined_noise)
        month_noise = truncate_data[1]
        return month_noise
    
    def MonthNoiseData(self, combined_noise, deterministic, m):
        PPA = PPAImbalanceClass()
        
        training_data_month = [dt.datetime.strptime(date , '%d/%m/%Y').month for date in combined_noise.Date]

        #m = dt.datetime.today().date().month
        truncate_data = PPA.TruncateNoise(training_data_month, m, combined_noise)
        month_noise = truncate_data[1]
        return month_noise
    
    def MonthDataDeterministic(self, combined_noise):
        PPA = PPAImbalanceClass()
        
        training_data_month = [dt.datetime.strptime(date , '%d/%m/%Y').month for date in combined_noise.Date]

        m = dt.datetime.today().date().month
        truncate_data = PPA.TruncateNoise(training_data_month, m, combined_noise)
        month_noise = truncate_data[1]
        return month_noise
    
    def GetThreshold(self, demand, m):
        training_data_month = [dt.datetime.strptime(date , '%d/%m/%Y').month for date in demand.Date]
        
        loc = np.where(np.asarray(training_data_month) == m)[0]
        demand_month = demand.iloc[loc, :]
        demand_month.index = range(len(demand_month))
        
        if np.isnan(np.asarray(demand_month.iloc[:, 1:]).astype(float)).any():
            nan_loc = np.where(np.isnan(np.asarray(demand_month.iloc[:, 1:])))
            demand_month.iloc[np.unique(nan_loc[0])[0], nan_loc[1]] = demand_month.iloc[np.unique(nan_loc[0])[0]-1, nan_loc[1]]
        
        demand_month_eve = np.max(demand_month.iloc[:, 1:], axis = 1)
        threshold = [np.percentile(demand_month_eve, 20), np.percentile(demand_month_eve, 50), np.percentile(demand_month_eve, 80)]
        return threshold, demand_month
    
    def ICRegression(self, train_c, train_ic):
        month_c = self.MonthData(train_c)
        month_ic = self.MonthData(train_ic)
        
        ic_corr = [np.corrcoef(month_c.iloc[:, i], month_ic.iloc[:, i])[0,1] for i in range(1,49)]
        high_ic_corr_loc = np.where(np.asarray(ic_corr) <= -0.60)[0]

        #MLR
        gradient = []
        intercept_values = []
        error = []
        for i in range(1, 49):
            ic = np.asarray(month_ic.iloc[:, i])
            c = np.asarray(month_c.iloc[:, i])
            slope, intercept, r_value, p_value, std_err = stats.linregress(ic.astype(float),c)
            
            gradient.append(slope)
            intercept_values.append(intercept)
            error.append(std_err)
            
        return high_ic_corr_loc, gradient, intercept_values, error
    
    def DARegression(self, train_c, train_da):
        month_c = self.MonthData(train_c)
        month_da = self.MonthData(train_da)
        
        corr = [np.corrcoef(month_c.iloc[:, i], month_da.iloc[:,i])[0,1] for i in range(1,49)]
        high_da_corr_loc = np.where(np.asarray(corr) > 0.50)[0]

        #MLR
        gradient = []
        intercept_values = []
        error = []
        for i in range(1, 49):
            d = np.asarray(month_da.iloc[:, i])
            c = np.asarray(month_c.iloc[:, i])
            slope, intercept, r_value, p_value, std_err = stats.linregress(d.astype(float),c)
            
            gradient.append(slope)
            intercept_values.append(intercept)
            error.append(std_err)
            
        return high_da_corr_loc, gradient, intercept_values, error
    
    def CashoutReturnsRegression(self, train_c):
        month_c = self.MonthData(train_c)
        
        corr = [np.corrcoef(month_c.iloc[1:, i], np.max(np.diff(month_c.iloc[:, 1:20], axis = 0), axis = 1))[0,1] for i in range(1,49)]
        #[np.corrcoef(train_c.iloc[2:, i], (np.asarray(train_c.iloc[2:, i]) - np.asarray(train_c.iloc[:-2, i])))[0,1] for i in range(1,49)]
        high_returns_corr_loc = np.where(np.asarray(corr) > 0.50)[0]

        #MLR
        gradient = []
        intercept_values = []
        error = []
        for i in range(1, 49):
            d = np.asarray(np.diff(month_c.iloc[:, i]))
            c = np.asarray(month_c.iloc[1:, i])
            slope, intercept, r_value, p_value, std_err = stats.linregress(d.astype(float),c)
            
            gradient.append(slope)
            intercept_values.append(intercept)
            error.append(std_err)
            
        return high_returns_corr_loc, gradient, intercept_values, error
    
    def CashoutVolRegression(self, train_c):
        month_c = self.MonthData(train_c)
        
        vol_corr = [np.corrcoef(month_c.iloc[2:, i], np.std(month_c.iloc[:, 1:], axis = 1)[:-2])[0,1] for i in range(1,49)]
        high_vol_corr_loc = np.where(np.asarray(vol_corr) > 0.65)[0]

        #MLR
        gradient = []
        intercept_values = []
        error = []
        for i in range(1, 49):
            g = np.asarray(np.std(month_c.iloc[:, 1:], axis = 1)[:-2])
            c = np.asarray(month_c.iloc[2:, i])
            slope, intercept, r_value, p_value, std_err = stats.linregress(g.astype(float),c)
            
            gradient.append(slope)
            intercept_values.append(intercept)
            error.append(std_err)
            
        return high_vol_corr_loc, gradient, intercept_values, error
    
    def GetTemperatureData(self):
        temperature = self.temperature_data
        temperature.index = range(len(temperature))
        return temperature
    
    def GetSolarData(self):
        STLF = ShortTermLoadForecastClass22()
        solar_gen = STLF.GetSolarGenData().iloc[::-1]
        solar_gen.index = range(len(solar_gen))
        return solar_gen
    
    def GetWindData(self):
        wind_gen_data_reversed = self.wind_gen_data[::-1]
        DCC = DataCleaningClass(wind_gen_data_reversed)
        hourly_wind_gen_reversed = DCC.ConvertHourlyBlockExplicit(range(1, 49), 'NG_Forecast')
        
        wind_gen = hourly_wind_gen_reversed.iloc[::-1]
        wind_gen.index = range(len(wind_gen))
        return wind_gen
 
    def Decompose(self, DA, C):
        PPA = PPAImbalanceClass()
        
        # Remove trend - rescale everyday with apx
        da_trendless = self.ExtractTrend([DA])[0]
        da_trend = 0
        c_trendless = self.ExtractTrend([C])[0]
        #PTC.ExtractTrend([C_21, C_20, C_19])[0]
        c_trend = 0

        # Remove seasonality
        da_seasonal_data = PPA.CaptureSeasonality(da_trendless.iloc[:, self.hours])
        da_seasonality = self.ConvertHourlySeasonalityHH(da_trendless, da_seasonal_data)
        #da_seasonality = (da_seasonality[0:365, :] + 2*da_seasonality[365:, :])/3
        da_deterministic = da_trend + da_seasonality
        da_deseasonalised = self.ConvertHourlyDeseasonalisedHH(da_trendless, da_seasonal_data)

        c_seasonal_data = PPA.CaptureSeasonalitySeparate(c_trendless)
        c_seasonality = c_seasonal_data[0]
        #c_seasonality = (c_seasonality[0:365, :] + 2*c_seasonality[365:, :])/3
        c_deterministic = c_trend + c_seasonality
        c_deseasonalised = c_seasonal_data[1]

        # To forecast 2022
        deterministic = np.zeros((len(da_trendless), 96))
        deterministic[:, 0:48] = da_deterministic
        deterministic[:, 48:] = c_deterministic
        deterministic = pd.DataFrame(deterministic)
        deterministic.insert(0, 'Date', da_deseasonalised.Date)

        # Combined noise
        combined_noise = pd.concat([da_deseasonalised, c_deseasonalised.iloc[:, 1:]], axis=1, join="inner")
        return deterministic, combined_noise
 
    def ExtractJumps(self, month_noise, i, sims, jumps):
        DAC = DayAheadClass()
        jump_fac_pos = min(DAC.GetJumpSizeParameters_ts(month_noise.iloc[:, i], 'positive'), 2)
        jump_fac_neg = min(DAC.GetJumpSizeParameters_ts(month_noise.iloc[:, i], 'negative'), 2)
        col_sig = np.std(month_noise.iloc[:, i])
        col_mu = np.percentile(month_noise.iloc[:, i], 50)
        
        if (col_mu + jump_fac_pos * col_sig) < 1000:
            if len(np.where(month_noise.iloc[:, i] > (col_mu + jump_fac_pos * col_sig))[0]) > 0:
                month_noise.iloc[:, i][np.where(month_noise.iloc[:, i] > col_mu + jump_fac_pos * col_sig)[0]] = (col_mu + jump_fac_pos * col_sig)
        else: 
            if len(np.where(month_noise.iloc[:, i] > (col_mu + jump_fac_pos * col_sig))[0]) > 0:
                previous_pos = month_noise.iloc[:, i][np.where(month_noise.iloc[:, i] > (col_mu + jump_fac_pos * col_sig))[0]-1]
                if len(np.where(month_noise.iloc[:, i] > (col_mu + jump_fac_pos * col_sig))[0]) > 1:
                    month_noise.iloc[:, i][np.where(month_noise.iloc[:, i] > col_mu + jump_fac_pos * col_sig)[0]] = previous_pos 
                else:
                    month_noise.iloc[:, i][np.where(month_noise.iloc[:, i] > col_mu + jump_fac_pos * col_sig)[0]] = previous_pos 
        
        if (col_mu - jump_fac_neg * col_sig) > -1000:
            if len(np.where(month_noise.iloc[:, i] < (col_mu - jump_fac_neg * col_sig))[0]) > 0:
                month_noise.iloc[:, i][np.where(month_noise.iloc[:, i] < col_mu - jump_fac_neg * col_sig)[0]] = col_mu - jump_fac_neg * col_sig 
        else: 
            if len(np.where(month_noise.iloc[:, i] < (col_mu - jump_fac_neg * col_sig))[0]) > 0:
                previous_pos = month_noise.iloc[:, i][np.where(month_noise.iloc[:, i] < (col_mu - jump_fac_neg * col_sig))[0]-1]
                if len(np.where(month_noise.iloc[:, i] < (col_mu - jump_fac_neg * col_sig))[0]) > 1:
                    month_noise.iloc[:, i][np.where(month_noise.iloc[:, i] < col_mu - jump_fac_neg * col_sig)[0]] = previous_pos 
                else:
                    month_noise.iloc[:, i][np.where(month_noise.iloc[:, i] < col_mu - jump_fac_neg * col_sig)[0]] = previous_pos 
        return month_noise
    
    def RemoveJumps(self, start, end, month_noise, jumps, N):
        #remove jumps - da: hh32 to hh44, c: hh66 to hh92   
        for i in range(start, end):
            for n in range(N):
                month_noise = self.ExtractJumps(month_noise, i, self.sims, jumps)
            #jumps = self.ExtractJumps(month_noise, i, self.sims, jumps)[1]
        return month_noise
    
    def GenerateNoiseOU(self, deseasonalised, sims, days):
        #Correlate each HH interval
        Corr = CorrelationClass()
        half_hourly_correlations = Corr.CholeskyMatrix(np.diff(deseasonalised, axis = 0))
        
        #must calibrate parameters - OLS, MLE, Jack-Knife
        #These would become an array for each HH interval
        OURC = OrnsteinUhlenbeckRegressorClass(0, 1, 365)
        theta = np.zeros(len(deseasonalised.T))
        mu = np.zeros(len(deseasonalised.T))
        sigma = np.zeros(len(deseasonalised.T))
        X0 = np.zeros(len(deseasonalised.T))
        for i in range(len(deseasonalised.T)):
            regression_parameters = OURC.ModifiedOLS(deseasonalised.iloc[:,i])
            theta[i] = regression_parameters[0]
            mu[i] = regression_parameters[1]
            sigma[i] = regression_parameters[2]
            X0[i] = deseasonalised.iloc[-1, i]#np.mean(deseasonalised.iloc[-30:-1, i])#
            
        #generate correlated noise process
        CSPC = CorrelatedStochasticProcessClass(1, days, days, half_hourly_correlations)
        noise_matrix = np.zeros((len(deseasonalised.T), sims, days))

        for i in range(len(deseasonalised.T)):
            noise_matrix[i, :, :] = CSPC.CorrelatedOUMatrix(X0[i], mu[i], sigma[i], theta[i], i, sims)
        return noise_matrix
    
    def GetNoise(self, spread_data):
        OURC = OrnsteinUhlenbeckRegressorClass(1, 2, 2)
        SPC = StochasticProcessClass(1, 2, 2)
        noise = np.zeros((96, 1000))
        for h in range(96):
            spread = spread_data.iloc[np.where(abs(spread_data.iloc[:, h]) < 10)[0], h]
            if len(spread) > 2:
                regression_parameters = OURC.MLE(spread)
                theta = regression_parameters[0]
                mu = regression_parameters[1]
                sigma = regression_parameters[2]
                X0 = spread.iloc[-1]
                
                noise[h, :] = SPC.OUMatrix(X0, mu, sigma, theta, self.sims)[:, 1]
        return noise
    
    def GetNoiseSell(self, spread):
        OURC = OrnsteinUhlenbeckRegressorClass(1, 2, 2)
        SPC = StochasticProcessClass(1, 2, 2)
        noise_sell = np.zeros((96, 1000))
        for h in range(96):
            sell = spread.iloc[np.where(spread.iloc[:, h] < 0)[0], h]
            if len(sell) < 2:
                sell = np.zeros(3)
                if len(spread.iloc[np.where(spread.iloc[:, h] < 0)[0], h]) == 0:
                    sell[1] = 0
                else:
                    sell[1] = np.asarray(spread.iloc[np.where(spread.iloc[:, h] < 0)[0], h])[0]
                    sell[0] = sell[1] - 0.5
                    sell[2] = sell[1] + 0.5
            
            col_mean = np.mean(sell)
            col_std = np.std(sell)
            outlier = np.where(sell < (col_mean - 1.5 * col_std))[0]
            if len(outlier) > 0:
                sell[sell.index[outlier]] = (col_mean - 1.5 * col_std)
            
            regression_parameters = OURC.MLE(sell)
            theta = regression_parameters[0]
            mu = regression_parameters[1]
            sigma = regression_parameters[2]
            X0 = spread.iloc[-1, h]
            
            noise_sell[h, :] = SPC.OUMatrix(X0, mu, sigma, theta, self.sims)[:, 1]
            adjust_loc = np.where(noise_sell[h, :] > 0)[0]
            if len(adjust_loc) > 0:
                noise_sell[h, adjust_loc] = noise_sell[h, adjust_loc] * -1
        return noise_sell
    
    def GetNoiseBuy(self, spread):
        OURC = OrnsteinUhlenbeckRegressorClass(1, 2, 2)
        SPC = StochasticProcessClass(1, 2, 2)
        noise_buy = np.zeros((96, 1000))
        for h in range(96):
            buy = spread.iloc[np.where(spread.iloc[:, h] > 0)[0], h]
            if len(buy) < 2:
                buy = np.zeros(3)
                if len(spread.iloc[np.where(spread.iloc[:, h] > 0)[0], h]) == 0:
                    buy[1] = 0
                else:
                    buy[1] = np.asarray(spread.iloc[np.where(spread.iloc[:, h] > 0)[0], h])[0]
                    buy[0] = buy[1] - 0.5
                    buy[2] = buy[1] + 0.5
            
            col_mean = np.mean(buy)
            col_std = np.std(buy)
            outlier = np.where(buy > (col_mean + 1.5 * col_std))[0]
            if len(outlier) > 0:
                if type(buy) == pd.core.series.Series:
                    buy[buy.index[outlier]] = (col_mean + 1.5 * col_std)
                else:
                    buy[outlier] = (col_mean + 1.5 * col_std)
            
            regression_parameters = OURC.MLE(buy)
            theta = regression_parameters[0]
            mu = regression_parameters[1]
            sigma = regression_parameters[2]
            X0 = spread.iloc[-1, h]
            
            noise_buy[h, :] = SPC.OUMatrix(X0, mu, sigma, theta, self.sims)[:, 1]
            adjust_loc = np.where(noise_buy[h, :] < 0)[0]
            if len(adjust_loc) > 0:
                noise_buy[h, adjust_loc] = noise_buy[h, adjust_loc] * -1
        return noise_buy
    
    def ExtractStochasticJumps(self, stochastic, combined_month_noise, i, sims, jumps):
        col_sig = np.std(combined_month_noise.iloc[:, i])
        col_mu = np.percentile(combined_month_noise.iloc[:, i], 50)
        
        if len(np.where(stochastic[i, :] > (col_mu + 5 * col_sig))[0]) > 0:
            #previous_pos = stochastic[i,:][np.where(stochastic[i, :] > (col_mu + 5 * col_sig))[0]-1]
            stochastic[i, :][np.where(stochastic[i, :] > (col_mu + 5 * col_sig))[0]] = (col_mu + 5 * col_sig)
                
        if len(np.where(stochastic[i, :] < (col_mu - 5 * col_sig))[0]) > 0:
            #previous_pos = stochastic[i,:][np.where(stochastic[i, :] < (col_mu - 5 * col_sig))[0]-1]
            stochastic[i, :][np.where(stochastic[i, :] < (col_mu - 5 * col_sig))[0]] = (col_mu - 5 * col_sig)
        return stochastic[i, :]
    
    def StochasticNoiseForecast(self, combined_noise, deterministic, m):
        month_noise = self.MonthNoiseData(combined_noise, deterministic, m)
        #recent_data = np.where(np.asarray([dt.datetime.strptime(month_noise.Date[d], '%d/%m/%Y').year for d in range(len(month_noise))]) == 2022)[0][0]
        #month_noise = month_noise.iloc[recent_data:, :]
        #month_noise.index = range(len(month_noise))
        
        jumps = np.zeros((96, self.sims))
        month_noise_da = self.RemoveJumps(30, 44, pd.DataFrame(month_noise.iloc[:, 1:49]), jumps, 3)
        month_noise_c = self.RemoveJumps(18, 40, pd.DataFrame(month_noise.iloc[:, 49:]), jumps, 3)
        combined_month_noise = pd.concat([month_noise_da, month_noise_c], axis=1, join="inner")

        # stochastic element
        stochastic = self.StochasticCleanHH(combined_month_noise, jumps)
        return stochastic
        
    def StochasticCleanHH(self, combined_month_noise, jumps):
        #stochastic element
        #stochastic = self.GenerateNoiseOU(combined_month_noise, self.sims, 2)[:, :, 1] 
        
# =============================================================================
#         for i in range(len(stochastic)):
#             stochastic[i, :] = self.ExtractStochasticJumps(stochastic, combined_month_noise, i, self.sims, jumps)
#         
#         nan = [np.percentile(stochastic[i, :], 50) for i in range(49, 96)]
#         nan_loc = np.where(np.isnan(np.asarray(nan)))[0]+1+48
#         if len(nan_loc > 0):
#             stochastic[nan_loc[0], :] = stochastic[nan_loc[0]-1, :]
# =============================================================================
        #spread = combined_month_noise.iloc[:, 48:] - combined_month_noise.iloc[:, :48]
        #combined_month_noise.iloc[:, 48:] = spread
        
        stochastic_mid = self.GetNoise(combined_month_noise)
        stochastic_sell = self.GetNoiseSell(combined_month_noise)
        stochastic_buy = self.GetNoiseBuy(combined_month_noise)
        stochastic = np.concatenate((stochastic_sell, stochastic_buy, stochastic_mid), axis = 1)
                
# =============================================================================
#         for i in range(int(len(stochastic)/2)):
#             stochastic[2*i+1, :] = stochastic[2*i, :]
# =============================================================================
        return stochastic
     
    def DeterministicForecast(self, yest_cashout, deterministic):
        month_deterministic = self.MonthDataDeterministic(deterministic)
        month_det = np.asarray(np.mean(month_deterministic.iloc[:, 1:], axis=0)[48:])
        
        d = datetime.date.today().timetuple().tm_yday
        
        model_det = np.asarray((deterministic.iloc[365*0 + d, 49:] + 2 * deterministic.iloc[365*1 + d, 49:] + 3 * deterministic.iloc[365*2 + d, 49:] + 4 * deterministic.iloc[365*3 + d, 49:])/10) 
        yest_det = yest_cashout
        da_det = ((np.asarray(np.mean(month_deterministic.iloc[:, 1:], axis=0)[0:48]) + np.asarray(deterministic.iloc[d-1, 1:49]))/2)
        c_det = (month_det + model_det + 2 * yest_det)/4
            
        det_sims = np.zeros((96, self.sims*3))
        det_sims[0:48, :] = (np.asarray(np.asarray(da_det)) * np.ones((self.sims*3, 48))).T
        det_sims[48:, :] = (np.asarray(np.asarray(c_det)) * np.ones((self.sims*3, 48))).T
        return det_sims
    
    def DeterministicForecastBacktest(self, date, yest_cashout, deterministic):
        month_deterministic = self.MonthDataDeterministic(deterministic)
        month_det = np.asarray(np.mean(month_deterministic.iloc[:, 1:], axis=0)[48:])
        
        d = dt.datetime.strptime(date, '%d/%m/%Y').timetuple().tm_yday
        
        model_det = np.asarray((deterministic.iloc[365*0 + d, 49:] + 2 * deterministic.iloc[365*1 + d, 49:])/3)
        #np.asarray((deterministic.iloc[365*0 + d, 49:] + 2 * deterministic.iloc[365*1 + d, 49:] + 3 * deterministic.iloc[365*2 + d, 49:] + 4 * deterministic.iloc[365*3 + d, 49:])/10) 
        yest_det = yest_cashout
        da_det = ((np.asarray(np.mean(month_deterministic.iloc[:, 1:], axis=0)[0:48]) + np.asarray(deterministic.iloc[d-1, 1:49]))/2)
        c_det = (month_det + model_det + 2 * yest_det)/4
            
        det_sims = np.zeros((96, self.sims*3))
        det_sims[0:48, :] = (np.asarray(np.asarray(da_det)) * np.ones((self.sims*3, 48))).T
        det_sims[48:, :] = (np.asarray(np.asarray(c_det)) * np.ones((self.sims*3, 48))).T
        return det_sims
     
    def CombineForecast(self, test_apx, det_sims, stochastic, regressions, high_corr):
        today_apx = (np.asarray(np.asarray(test_apx)) * np.ones((self.sims*3, 48))).T
        
        c_da_regression = regressions[0]
        c_returns_regression = regressions[1]
        c_vol_regression = regressions[2]
        c_ic_regression = regressions[3]
        
        high_da_corr_loc = high_corr[0]
        high_returns_corr_loc = high_corr[1]
        high_vol_corr_loc = high_corr[2]
        high_ic_corr_loc = high_corr[3]
        
        forecast = stochastic.copy()
        forecast[:48, :] = det_sims[:48, :] + stochastic[:48, :]
        forecast[48:, :] =  (det_sims[48:, :])
        
        if len(high_returns_corr_loc) > 0:
            det_regression = np.zeros(48)
            loc = np.asarray(list(set(high_da_corr_loc).intersection(high_returns_corr_loc)))
        else:
            loc = high_da_corr_loc
        det_regression[loc] = (c_da_regression[loc] + c_returns_regression[loc])/2
        
        if len(high_vol_corr_loc) > 0:
            loc_vol = np.asarray(list(set(loc).intersection(high_vol_corr_loc)))
        else: 
            loc_vol = loc
        det_regression[loc_vol] = (det_regression[loc_vol] + c_vol_regression[loc_vol])/2
        det_loc = np.where(det_regression != 0)[0]
        
        buy_loc = det_loc[np.where((det_regression[det_loc] - today_apx[det_loc, 0]) > 0)[0]]
        sell_loc = det_loc[np.where((det_regression[det_loc] - today_apx[det_loc, 0]) < 0)[0]]

        #buy
        if len(buy_loc) > 0:
            forecast[(48 + buy_loc), 1000:2000] = (forecast[(48 + buy_loc), 1000:2000] + \
                                                   (np.asarray(np.asarray(det_regression[buy_loc])) * \
                                                    np.ones((self.sims, len(buy_loc)))).T)/2
        #sell
        if len(sell_loc) > 0:
           forecast[(48 + sell_loc), :1000] = (forecast[(48 + sell_loc), :1000] + \
                                                  (np.asarray(np.asarray(det_regression[sell_loc])) * \
                                                   np.ones((self.sims, len(sell_loc)))).T)/2
 

# =============================================================================
#         forecast[(48 + det_loc), :] = (forecast[(48 + det_loc), :] + \
#                                                (np.asarray(np.asarray(det_regression[det_loc])) * \
#                                                 np.ones((self.sims*3, len(det_loc)))).T)/2
# =============================================================================
        
        buy_ic_loc = det_loc[np.where((c_ic_regression[det_loc] - today_apx[det_loc, 0]) > 0)[0]]
        sell_ic_loc = det_loc[np.where((c_ic_regression[det_loc] - today_apx[det_loc, 0]) < 0)[0]]
        
        #buy
        if len(buy_ic_loc) > 0:
            forecast[(48 + buy_ic_loc), 1000:2000] = (forecast[(48 + buy_ic_loc), 1000:2000] + \
                                                   (np.asarray(np.asarray(det_regression[buy_ic_loc])) * \
                                                    np.ones((self.sims, len(buy_ic_loc)))).T)/2
        #sell
        if len(sell_loc) > 0:
           forecast[(48 + sell_ic_loc), :1000] = (forecast[(48 + sell_ic_loc), :1000] + \
                                                  (np.asarray(np.asarray(det_regression[sell_ic_loc])) * \
                                                   np.ones((self.sims, len(sell_ic_loc)))).T)/2


# =============================================================================
#         forecast[(48 + high_ic_corr_loc), :] = (forecast[(48 + high_ic_corr_loc), :] + \
#                                                         (np.asarray(np.asarray(c_ic_regression[high_ic_corr_loc])) * \
#                                                          np.ones((self.sims*3, len(high_ic_corr_loc)))).T)/2
# =============================================================================

        forecast[48:, :] =  forecast[48:, :] + stochastic[48:, :] + today_apx
        return forecast        
        
    def RescaleForecast(self, yest_cashout, forecast):
        PLD = PullLiveDataClass()
        #avg_da = np.mean(Test_22.iloc[d-1, 1:48])
        #yest_cashout = self.ScrapeRecentTwoDayCashout()
        avg_da = np.mean(PLD.ScrapeDayAhead()[1])
        avg_c = np.mean(yest_cashout)
                       
        #forecast[0:48, :] = forecast[0:48, :]/np.mean(forecast[0:48, :]) * avg_da
        forecast[48:, :] = forecast[48:, :]/np.nanmean(forecast[48:, :]) * avg_da
        #forecast[48:, :] - (np.nanmean(forecast[48:, :]) - avg_c)
        return forecast
    
    def RescaleForecastTranslate(self, yest_cashout, forecast):
        PLD = PullLiveDataClass()
        #avg_da = np.mean(Test_22.iloc[d-1, 1:48])
        #yest_cashout = self.ScrapeRecentTwoDayCashout()
        avg_da = np.mean(PLD.ScrapeDayAhead()[1])
        avg_c = np.mean(yest_cashout)
                       
        #forecast[0:48, :] = forecast[0:48, :]/np.mean(forecast[0:48, :]) * avg_da
        forecast[48:, :] = forecast[48:, :] - (np.nanmean(forecast[48:, :]) - avg_da)
        return forecast

    def NominateHhtoH(self, sell_da):
        n = 0
        while n < 3:
            for i in range(len(sell_da)):
                if (sell_da[i]%2)==0 and len(np.where(sell_da == sell_da[i] + 1)[0]) == 0:
                    sell = sell_da.tolist()
                    sell.append(sell_da[i] + 1)
                    sell = np.sort(sell)
                    sell_da = np.asarray(sell)
                    
                if (sell_da[i]%2)==1 and len(np.where(sell_da == sell_da[i] - 1)[0]) == 0:
                    sell = sell_da.tolist()
                    sell.append(sell_da[i] - 1)
                    sell = np.sort(sell)
                    sell_da = np.asarray(sell)
            n = n + 1     
        return sell_da
        
    def Prioritise(self, sell_da, buy_da, most_likely):
        #prioritise
        common = np.asarray(list(set(sell_da).intersection(buy_da)))
        common = np.sort(common)
        for i in range(int(len(common)/2)):
            if np.mean(most_likely[common[2*i:2*i+2]]) == 0:
                buy_da = list(set(buy_da) - set(common[2*i:2*i+2]))
                buy_da = np.asarray(buy_da)
                
                sell_da = list(set(sell_da) - set(common[2*i:2*i+2]))
                sell_da = np.asarray(sell_da)                
            elif np.mean(most_likely[common[2*i:2*i+2]]) < 0:
                buy_da = list(set(buy_da) - set(common[2*i:2*i+2]))
                buy_da = np.asarray(buy_da)
            elif np.mean(most_likely[common[2*i:2*i+2]]) > 0:
                sell_da = list(set(sell_da) - set(common[2*i:2*i+2]))
                sell_da = np.asarray(sell_da)
        return sell_da, buy_da
             
    def MonthlyThresholds(self, data, m):
        train_data = data[0]
        test_data = data[1]
        
        NIV_month = self.GetThreshold(train_data, m)[1]
        
        lower_thresh = [np.percentile(NIV_month.iloc[:, i].astype(float), 20) for i in range(1, 49)]
        upper_thresh = [np.percentile(NIV_month.iloc[:, i].astype(float), 90) for i in range(1, 49)]
        
        lower_threshold = np.where((np.asarray(test_data) - np.asarray(lower_thresh)) < 0)[0]
        upper_threshold = np.where((np.asarray(test_data) - np.asarray(upper_thresh)) > 0)[0]
        return lower_threshold, upper_threshold
    
    def GetMonthSWT(self, swt):
        PLD = PullLiveDataClass()
        sol = swt[0]
        wind = swt[1]
        temp = swt[2]
        
        month_sol = self.MonthData(sol)
        month_wind = self.MonthData(wind)
        month_temp = self.MonthData(temp)
        
        current_sol = PLD.ScrapeSolarGen()
        current_wind = PLD.ScrapeWindGen()
        current_temp = PLD.ScrapeTempGen()
        return [month_sol, current_sol], [month_wind, current_wind], [month_temp, current_temp]
    
    def EFATightPriceTakeIII(self, factor, buffer, forecast, swt, train_apx, test_apx):           
        difference = forecast[48:, :] - (np.asarray(np.asarray(test_apx)) * np.ones((self.sims, 48))).T
        most_likely = np.percentile(difference, 50, axis = 1)
        profit_margin = test_apx + factor * most_likely

        execute_sell_da = np.where(most_likely < 0 - buffer)[0]
        execute_buy_da = np.where(most_likely > 0 + buffer)[0]
        volume_pnl = np.zeros(48)
        price_pnl = np.zeros(48)
        tight_hh = np.zeros(48)
        
        m = dt.datetime.today().date().month
        apx_month = self.GetThreshold(train_apx, m)[1]
        lower_thresh = [np.percentile(apx_month.iloc[:, i], 10) for i in range(1, 49)]
        upper_thresh = [np.percentile(apx_month.iloc[:, i], 90) for i in range(1, 49)]
        
        lower_threshold = np.where((np.asarray(test_apx) - np.asarray(lower_thresh)) < 0)[0]
        #buy da only
        execute_sell_da = list(set(execute_sell_da) - set(lower_threshold))
        execute_sell_da = np.asarray(execute_sell_da)
        
        upper_threshold = np.where((np.asarray(test_apx) - np.asarray(upper_thresh)) > 0)[0]
        #sell da only
        execute_buy_da = list(set(execute_buy_da) - set(upper_threshold))
        execute_buy_da = np.asarray(execute_buy_da)
        
        sol = swt[0]
        wind = swt[1]
        temp = swt[2]

        sol_low_thresh = self.MonthlyThresholds(sol, m)[0].tolist()
        wind_low_thresh = self.MonthlyThresholds(wind, m)[0].tolist()
        temp_low_thresh = self.MonthlyThresholds(temp, m)[0].tolist()
        
        low_gen = wind_low_thresh + temp_low_thresh
        #sol_low_thresh + wind_low_thresh + temp_low_thresh
        tight = low_gen
        #Take low gen as a proportion of demand
        #dynamic threshold for summer vs winter 
        #scrap solar 
        #consider monthly residual demand thresholds instead of swt thresholds 
        
        sol_high_thresh = self.MonthlyThresholds(sol, m)[1].tolist()
        wind_high_thresh = self.MonthlyThresholds(wind, m)[1].tolist()
        temp_high_thresh = self.MonthlyThresholds(temp, m)[1].tolist()
        
        high_gen = wind_high_thresh + temp_high_thresh
        #sol_high_thresh + wind_high_thresh + temp_high_thresh
        relaxed = high_gen
        tight_hh[tight] = 1
            
        if len(execute_sell_da) > 0:
            execute_sell_da = self.NominateHhtoH(execute_sell_da)
        if len(execute_buy_da) > 0:
            execute_buy_da = self.NominateHhtoH(execute_buy_da)
            
        execute_sell_da = self.Prioritise(execute_sell_da, execute_buy_da, most_likely)[0]
        execute_buy_da = self.Prioritise(execute_sell_da, execute_buy_da, most_likely)[1]
            
        if len(execute_sell_da) > 0:
            #realised_pnl[execute_sell_da] = -(actual_pnl[execute_sell_da])
            volume_pnl[execute_sell_da] = -1
            price_pnl[execute_sell_da] = profit_margin[execute_sell_da]
        if len(execute_buy_da) > 0:
            #realised_pnl[execute_buy_da] = (actual_pnl[execute_buy_da])
            volume_pnl[execute_buy_da] = 1
            price_pnl[execute_buy_da] = profit_margin[execute_buy_da]
                
        return volume_pnl, price_pnl, tight_hh, execute_sell_da, execute_buy_da, most_likely, profit_margin
    
    def Invert(self, execute_sell_da, execute_buy_da, most_likely, profit_margin):
        PLD = PullLiveDataClass()
        volume_pnl = np.zeros(48)
        price_pnl = np.zeros(48)
        yest_cashout = PLD.ScrapeRecentTwoDayCashout()[49:97]
        train_c = self.Cashout
        if np.nanstd(yest_cashout) > 0.5 * np.mean(np.std(train_c.iloc[:, 1:], axis = 1)):
            invert_execute = [execute_buy_da.copy(), execute_sell_da.copy()]
            execute_buy_da = invert_execute[1]
            execute_sell_da = np.asarray(list(set(invert_execute[0]) - set([35, 36, 37, 38])))
        
        if len(execute_sell_da) > 0:
            execute_sell_da = self.NominateHhtoH(execute_sell_da)
        if len(execute_buy_da) > 0:
            execute_buy_da = self.NominateHhtoH(execute_buy_da)
            
        execute_sell_da = self.Prioritise(execute_sell_da, execute_buy_da, most_likely)[0]
        execute_buy_da = self.Prioritise(execute_sell_da, execute_buy_da, most_likely)[1]
            
        if len(execute_sell_da) > 0:
            #realised_pnl[execute_sell_da] = -(actual_pnl[execute_sell_da])
            volume_pnl[execute_sell_da] = -1
            price_pnl[execute_sell_da] = profit_margin[execute_sell_da]
        if len(execute_buy_da) > 0:
            #realised_pnl[execute_buy_da] = (actual_pnl[execute_buy_da])
            volume_pnl[execute_buy_da] = 1
            price_pnl[execute_buy_da] = profit_margin[execute_buy_da]
        return volume_pnl, price_pnl
    
    def EFATightPriceTakeIII_2(self, factor, buffer, forecast, swt, train_apx, test_apx):           
        most_likely = forecast - (np.asarray(np.asarray(test_apx)))
        profit_margin = test_apx + factor * most_likely

        execute_sell_da = np.where(most_likely < 0 - buffer)[0]
        execute_buy_da = np.where(most_likely > 0 + buffer)[0]
        volume_pnl = np.zeros(48)
        price_pnl = np.zeros(48)
        tight_hh = np.zeros(48)
        
        m = dt.datetime.today().date().month
        apx_month = self.GetThreshold(train_apx, m)[1]
        lower_thresh = [np.percentile(apx_month.iloc[:, i], 10) for i in range(1, 49)]
        upper_thresh = [np.percentile(apx_month.iloc[:, i], 90) for i in range(1, 49)]
        
        lower_threshold = np.where((np.asarray(test_apx) - np.asarray(lower_thresh)) < 0)[0]
        #buy da only
        execute_sell_da = list(set(execute_sell_da) - set(lower_threshold))
        execute_sell_da = np.asarray(execute_sell_da)
        
        upper_threshold = np.where((np.asarray(test_apx) - np.asarray(upper_thresh)) > 0)[0]
        #sell da only
        execute_buy_da = list(set(execute_buy_da) - set(upper_threshold))
        execute_buy_da = np.asarray(execute_buy_da)
        
        sol = swt[0]
        wind = swt[1]
        temp = swt[2]

        sol_low_thresh = self.MonthlyThresholds(sol, m)[0].tolist()
        wind_low_thresh = self.MonthlyThresholds(wind, m)[0].tolist()
        temp_low_thresh = self.MonthlyThresholds(temp, m)[0].tolist()
        
        low_gen = wind_low_thresh + temp_low_thresh
        #sol_low_thresh + wind_low_thresh + temp_low_thresh
        tight = low_gen
        
        sol_high_thresh = self.MonthlyThresholds(sol, m)[1].tolist()
        wind_high_thresh = self.MonthlyThresholds(wind, m)[1].tolist()
        temp_high_thresh = self.MonthlyThresholds(temp, m)[1].tolist()
        
        high_gen = wind_high_thresh + temp_high_thresh
        #sol_high_thresh + wind_high_thresh + temp_high_thresh
        relaxed = high_gen
        
        if len(tight) > 0:
            #buy da only
            execute_sell_da = list(set(execute_sell_da) - set(tight))
            execute_sell_da = np.asarray(execute_sell_da)   
         
        tight_hh[tight] = 1
            
        if len(execute_sell_da) > 0:
            execute_sell_da = self.NominateHhtoH(execute_sell_da)
        if len(execute_buy_da) > 0:
            execute_buy_da = self.NominateHhtoH(execute_buy_da)
            
        execute_sell_da = self.Prioritise(execute_sell_da, execute_buy_da, most_likely)[0]
        execute_buy_da = self.Prioritise(execute_sell_da, execute_buy_da, most_likely)[1]
            
        if len(execute_sell_da) > 0:
            #realised_pnl[execute_sell_da] = -(actual_pnl[execute_sell_da])
            volume_pnl[execute_sell_da] = -1
            price_pnl[execute_sell_da] = profit_margin[execute_sell_da]
        if len(execute_buy_da) > 0:
            #realised_pnl[execute_buy_da] = (actual_pnl[execute_buy_da])
            volume_pnl[execute_buy_da] = 1
            price_pnl[execute_buy_da] = profit_margin[execute_buy_da]
                
        return volume_pnl, price_pnl, tight_hh
        
    def Trade(self, test_apx, train_apx, forecast, SWT):
        month_APX = self.MonthData(train_apx)

        jump_cond = np.std(test_apx) < 3 * \
            (np.mean(np.std(month_APX.iloc[:, 1:], axis=1)))

        if jump_cond:
            month_swt = self.GetMonthSWT(SWT)
            trade = self.EFATightPriceTakeIII(0.5, 2, forecast, month_swt, month_APX, test_apx)
            
        return trade
    
    def Trade_2(self, test_apx, train_apx, forecast, SWT):
        month_APX = self.MonthData(train_apx)

        jump_cond = np.std(test_apx) < 3 * \
            (np.mean(np.std(month_APX.iloc[:, 1:], axis=1)))

        if jump_cond:
            month_swt = self.GetMonthSWT(SWT)
            trade = self.EFATightPriceTakeIII_2(0.5, 0, forecast, month_swt, month_APX, test_apx)
            
        return trade
    
    def TradeInvert(self, execute_sell_da, execute_buy_da, most_likely, profit_margin):
        trade = self.Invert(execute_sell_da, execute_buy_da, most_likely, profit_margin)
            #EFAInvertTrade(0.5, 2, forecast, month_swt, month_APX, test_apx)
        return trade
          
    def ErrorCheck(self, forecast):
        if (forecast[0:48, :] < -50).any():
            for i in range(48):
                min_loc = np.where(forecast[i, :] < -50)
                forecast[i, min_loc] = forecast[i, min_loc]/10

        if (forecast[48:, :] < -100).any():
            for i in range(48):
                min_loc = np.where(forecast[i+48, :] < -100)
                forecast[i+48, min_loc] = -100

        if (forecast[48:, :] > 5000).any():
            for i in range(48):
                max_loc = np.where(forecast[i+48, :] > 5000)
                forecast[i+48, max_loc] = forecast[i+48, max_loc]/1000
                
        return forecast

    def AdjustForecast(self, yest_cashout, forecast, test_apx, translate_correction):
        if translate_correction:
            forecast[48:, :] = self.RescaleForecastTranslate(yest_cashout, forecast)[48:, :]
        else:
            forecast[48:, :] = self.RescaleForecast(yest_cashout, forecast)[48:, :]
            
        forecast[0:48, :] = forecast[0:48, :]/np.mean(forecast[0:48, :]) * np.mean(test_apx)
        
        f = forecast[0:48, :]
        correction = [(np.percentile(f, 50, axis=1)[i] - test_apx[i]) for i in range(48)]
        forecast[0:48, :] = np.asarray([f[i, :] - np.asarray(correction)[i] for i in range(48)])
        
        forecast = self.ErrorCheck(forecast)
        return forecast
    
    def GetForecastBacktest(self, date, yest_cashout, test_apx, test_ic, today_rd):
        #PLD = PullLiveDataClass()
        m = dt.datetime.strptime(date, '%d/%m/%Y').month

        # Historic price data
        recent = np.where(np.asarray([dt.datetime.strptime(self.Cashout.Date[d], '%d/%m/%Y').year for d in range(len(self.Cashout))]) == 2022)[0][0]
        train_loc = np.where(self.Cashout.Date == date)[0][0]
        
        train_c = self.Cashout.iloc[recent:train_loc, :]
        train_apx = self.DA_APX.iloc[recent:train_loc, :]
        train_ic = self.IC.iloc[recent:train_loc, :]

        # Noise
        decompose_data = self.Decompose(self.DA.iloc[recent:train_loc, :], train_c)
        deterministic = decompose_data[0]
        combined_noise = decompose_data[1]
        
        da_corr = self.DARegression(train_c, train_apx)
        returns_corr = self.CashoutReturnsRegression(train_c)
        vol_corr = self.CashoutVolRegression(train_c)
        ic_corr = self.ICRegression(train_c, train_ic)
        
        high_da_corr_loc = da_corr[0]
        da_gradient = da_corr[1]
        da_intercept_values = da_corr[2]
        
        high_returns_corr_loc = returns_corr[0]
        returns_gradient = returns_corr[1]
        returns_intercept_values = returns_corr[2]
        
        high_vol_corr_loc = vol_corr[0]
        vol_gradient = vol_corr[1]
        vol_intercept_values = returns_corr[2]
        
        high_ic_corr_loc = ic_corr[0]
        ic_gradient = ic_corr[1]
        ic_intercept_values = ic_corr[2]
        
        two_day_cashout = yest_cashout
        #self.ScrapeRecentTwoDayCashout()[0:48]
        scheduled_ic = test_ic
        
        train_da_reg = (da_gradient * self.DA.iloc[recent:train_loc, 1:] + da_intercept_values).iloc[2:, :]
        train_c_ret_reg = (returns_gradient * train_c.iloc[1:, 1:] + returns_intercept_values).iloc[1:, :]
        train_c_vol_reg = (vol_gradient * train_c.iloc[2:, 1:] + vol_intercept_values)
        train_ic_reg = (ic_gradient * self.DA.iloc[recent:train_loc, 1:] + ic_intercept_values).iloc[2:, :]
        train_reg = (train_da_reg + train_c_ret_reg + train_c_vol_reg + train_ic_reg)/4
        train_reg.index = range(len(train_reg))
        train_det = deterministic.copy()
        train_det.iloc[2:, 49:] = (np.asarray(train_det.iloc[2:, 49:]) + np.asarray(train_reg))/2
        
        spread_noise = pd.concat([self.DA.iloc[recent:train_loc, :], train_c.iloc[:, 1:]], axis = 1, join = 'inner')
        spread_noise.iloc[:, 49:] = (np.asarray(train_det.iloc[:, 49:]) + np.asarray(combined_noise.iloc[:, 49:])) - np.asarray(train_apx.iloc[:, 1:])
        #spread_noise.iloc[:, 48:] = spread_noise.iloc[:, 48:] - train_apx.iloc[:, 1:]
        stochastic = self.StochasticNoiseForecast(spread_noise, deterministic, m)
        #stochastic = self.StochasticNoiseForecast(combined_noise, deterministic, m)
# =============================================================================
#         c_stoch = stochastic[48:, :]
#         for s in range(self.sims*3):
#             loc = np.where(c_stoch[:, s] < -50)[0]
#             stochastic[48+loc, s] = -50#stochastic[48+loc, s]/5
#             loc_buy = np.where(c_stoch[:, s] > 100)[0]
#             stochastic[48+loc_buy, s] = 100#stochastic[48+loc_buy, s]/5
#         
# =============================================================================
        c_da_regression = np.asarray(da_gradient) * test_apx + da_intercept_values
        c_returns_regression = np.asarray(returns_gradient) * (np.max(np.asarray(yest_cashout) - np.asarray(two_day_cashout))) + returns_intercept_values
        c_vol_regression = np.asarray(vol_gradient) * np.std(yest_cashout) + vol_intercept_values
        c_ic_regression = np.asarray(ic_gradient) * scheduled_ic + ic_intercept_values
        
        regressions = [c_da_regression, c_returns_regression, c_vol_regression, c_ic_regression]
        high_corr = [high_da_corr_loc, high_returns_corr_loc, high_vol_corr_loc, high_ic_corr_loc]

        det_sims = self.DeterministicForecastBacktest(date, yest_cashout, deterministic)
        forecast = self.CombineForecast(test_apx, det_sims, stochastic, regressions, high_corr) 
        forecast[48:, :] = forecast[48:, :]/np.nanmean(forecast[48:, :]) * np.mean(test_apx)
# =============================================================================
#         if (today_rd < 7500).any():
#             low_rd = np.where(today_rd < 7500)[0]
#             for s in range(1000, 2000):
#                 loc_buy = np.where(c_stoch[:, s] > 0)[0]
#                 change_loc = np.asarray(list(set(low_rd).intersection(loc_buy)))
#                 forecast[48+change_loc, s] = forecast[48+change_loc, s] * 0.7
#                                  
# =============================================================================
        return forecast

    def ExecuteTrade(self, yest_cashout, test_apx, SWT):
        PLD = PullLiveDataClass()
        # Historic price data
        train_c = self.Cashout
        train_apx = self.DA_APX
        train_ic = self.IC

        # Noise
        decompose_data = self.Decompose(self.DA, train_c)
        deterministic = decompose_data[0]
        combined_noise = decompose_data[1]
        
        da_corr = self.DARegression(train_c, train_apx)
        returns_corr = self.CashoutReturnsRegression(train_c)
        vol_corr = self.CashoutVolRegression(train_c)
        ic_corr = self.ICRegression(train_c, train_ic)
        
        high_da_corr_loc = da_corr[0]
        da_gradient = da_corr[1]
        da_intercept_values = da_corr[2]
        
        high_returns_corr_loc = returns_corr[0]
        returns_gradient = returns_corr[1]
        returns_intercept_values = returns_corr[2]
        
        high_vol_corr_loc = vol_corr[0]
        vol_gradient = vol_corr[1]
        vol_intercept_values = returns_corr[2]
        
        high_ic_corr_loc = ic_corr[0]
        ic_gradient = ic_corr[1]
        ic_intercept_values = ic_corr[2]
        
        two_day_cashout = yest_cashout
        #self.ScrapeRecentTwoDayCashout()[0:48]
        scheduled_ic = PLD.ScrapeScheduledDAIC()
        
        stochastic = self.StochasticNoiseForecast(combined_noise, deterministic)
        c_da_regression = np.asarray(da_gradient) * test_apx + da_intercept_values
        c_returns_regression = np.asarray(returns_gradient) * (np.max(np.asarray(yest_cashout) - np.asarray(two_day_cashout))) + returns_intercept_values
        c_vol_regression = np.asarray(vol_gradient) * np.std(yest_cashout) + vol_intercept_values
        c_ic_regression = np.asarray(ic_gradient) * scheduled_ic + ic_intercept_values
        
        regressions = [c_da_regression, c_returns_regression, c_vol_regression, c_ic_regression]
        high_corr = [high_da_corr_loc, high_returns_corr_loc, high_vol_corr_loc, high_ic_corr_loc]

        det_sims = self.DeterministicForecast(yest_cashout, deterministic)
        forecast = self.CombineForecast(det_sims, stochastic, regressions, high_corr) 
        translate_correction = False
        forecast = self.AdjustForecast(yest_cashout, forecast, test_apx, translate_correction)

        trade = self.Trade(test_apx, train_apx, forecast, SWT)
        return trade, forecast

    def GetAdjustedTradePrice(self, trade):
        volume = trade[0][0]
        trade_price = trade[0][1]
        
        for i in range(24):
            if np.mean(volume[2*i:2*(i+1)] == 1):
                trade_price[2*i:2*(i+1)] = np.max(trade_price[2*i:2*(i+1)])
            if np.mean(volume[2*i:2*(i+1)] == -1):
                trade_price[2*i:2*(i+1)] = np.min(trade_price[2*i:2*(i+1)])
        return volume, trade_price
    
    def GetAdjustedTradePrice_2(self, trade):
        volume = trade[0]
        trade_price = trade[1]
        
        for i in range(24):
            if np.mean(volume[2*i:2*(i+1)] == 1):
                trade_price[2*i:2*(i+1)] = np.max(trade_price[2*i:2*(i+1)])
            if np.mean(volume[2*i:2*(i+1)] == -1):
                trade_price[2*i:2*(i+1)] = np.min(trade_price[2*i:2*(i+1)])
        return volume, trade_price

    def DeclareTrades(self, test_apx, cashout_forecast, volume, trade_price):
        today = (dt.datetime.today().date() + timedelta(days = 1)).strftime('%d/%m/%Y')
        trade_df = pd.DataFrame([[today]*48, range(1, 49), test_apx, cashout_forecast, volume, trade_price])
        trade_df = trade_df.T
        trade_df.columns = ["Date", "SP", "APX", "Imbalance Forecast", "Position", "Trade Price" ]
        return trade_df
    