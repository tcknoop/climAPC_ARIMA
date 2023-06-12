import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sc
from helpfuncs import * 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
     


params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
#%%
#TODO

#interpolation function
    #test for NANS,extreme jumps, examine time vector
    #incorporate into ARIMA,and examine function
    
#ARIMA(pdq)
#construct arima model and fit to data
#add seasonality (VARIMA?)

#predict ARIMA, estimates n further timesteps?

#%% load test data
#t_series = np.genfromtxt('data/NAO_index_test.txt', delimiter =' ')
t_series = np.genfromtxt('data/Pacific_warmpool_test.txt', delimiter =' ')
data = t_series[:,1]
time = t_series[:,0]
nan_index = [2,10,11,12]
data2 = data.copy()
data2[nan_index] = np.nan
#%%

def examine_data(data,time, d = 0):
    
    #TODO
    #plot to determine stationarity?
    
    if data.shape != time.shape:
        raise ValueError(f"Timeseries data and time vector should have identical dimensions, not {data.shape} and {time.shape} ")
    
    #differentiate d times
    if d > 0:
        data,time = difference(data,time, d = d)
    
    #plot data,ACF and PACF
    fig = plt.figure(figsize=(6,10))

    ax = fig.add_subplot(3,1,1)
    ax.plot(time,data)
    ax.set_xlabel('Time')
    ax.set_title(f'Timeseries d={d} times differentiated')
    ax.grid()

    ax = fig.add_subplot(3,1,2)
    plot_acf(data, ax = ax)
    ax.set_xlabel('Lag')
    ax.grid()
    ax = fig.add_subplot(3,1,3)
    plot_pacf(data, ax = ax)
    ax.set_xlabel('Lag')
    ax.grid()

    plt.tight_layout()
    return None

#examine_data(data, time, d = 2)

def interp_data(data, time , method):
    #TODO
    #different methods:
        #cut NANs
        #linear
        #seasonal?
    
    return None

def construct_ARIMA(data,time,
                    order = (1,1,1),#AR(q),I(d),MA(p)
                    season = None,
                    manual = True,
                    plot_result = True
                    ):
    #TODO add seasonal
    #construct ARIMA model
    model = ARIMA(data, order = order)
    
    #fit
    result = model.fit()
    print(result.summary())
    #plot best model?
    if plot_result:
        fig = plt.figure(figsize=(6,8))
    
        ax = fig.add_subplot(2,1,1)
        ax.plot(time,data, label = 'Data')
        ax.plot(time,result.fittedvalues, label = 'Fitted Model')
        ax.legend()
        
        ax = fig.add_subplot(2,1,2)
        ax.plot(time,result.resid)
        ax.set_title('Residuals')
    
    #TODO print RMSE, or AIC to compare
    #maybe parse on to automate finding best model
    return result

def forecast_ARIMA(model):
    #TODO forecast here
    print('nothing to see here yet')
    #plot with confidence interv
    return None


#res = construct_ARIMA(data, time, order = (7,1,1))
#%%











