import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sc
from helpfuncs import * 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
props = dict(boxstyle='square', facecolor='lightgrey',edgecolor = 'red', alpha=0.9)


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
order = (1,1,1)
#%%

def examine_data(data,time, d = 0, warnings = True):
    '''
    

    Parameters
    ----------
    data : 1-D array-like
        Input timeseries.
    time : 1-D array-like
        Input time vector, same length as data.
    d : int, optional
        Order of differencing to examine possible higher orders of d. The default is 0.
    warnings : bool, optional
        prints possible warnings into the first plot. The default is True.

    Raises
    ------
    ValueError
        For not matching dimension between data and time.

    Returns
    -------
    None.

    '''
    
    if data.shape != time.shape:
        raise ValueError(f"Timeseries data and time vector should have identical dimensions, not {data.shape} and {time.shape} ")
    
    warn = check_for_errors(data, time)
    if warnings == False:
        warn = None
    #differentiate d times
    if d > 0:
        data,time = difference(data,time, d = d)
    
    #plot data,ACF and PACF
    fig = plt.figure(figsize=(6,10))

    ax = fig.add_subplot(3,1,1)
    ax.plot(time,data)
    ax.set_xlabel('Time')
    ax.set_title(f'Timeseries d={d} times differentiated')
    ax.text(0.05, 0.95, warn, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    ax.grid()

    ax = fig.add_subplot(3,1,2)
    plot_acf(data, ax = ax)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.grid()
    
    ax = fig.add_subplot(3,1,3)
    plot_pacf(data, ax = ax)
    ax.set_ylabel('PACF')
    ax.set_xlabel('Lag')
    ax.grid()

    plt.tight_layout()
    return None

#examine_data(data, time, d = 2)

def fix_data(data, time , method):
    #TODO
    #different methods:
        #cut NANs
        #linear
        #seasonal?
    
    return data, time

def construct_ARIMA(data,time,
                    order = (1,1,1),#AR(q),I(d),MA(p)
                    season = (0,0,0,0),# (Q,D,P,s)
                    manual = True,
                    plot_result = True,
                    ret_aic = False
                    ):
    '''
    

    Parameters
    ----------
    data : 1-D array-like
        Input timeseries.
    time : 1-D array-like
        Input time vector, same length as data.
    order : tuple (int,int,int) , optional
        (p,d,q) Parameters in the ARIMA model. p for autoregressive model AR(p);
        d for the order of integration; q for the moving-average model MA(q). Default is (1,1,1).
    season: bool or tuple:
       expands the model to a SARIMA model. must include the seasonal parameters (Q,D,P,s);
       s indicates the number of timesteps per year (e.g. 12 for monthly timeseries).
    manual : bool, optional
        wether to manually determine model parameters or let the statsmodels library. The default is True.
    plot_result : bool, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    result : TYPE
        fitted ARIMA model object; can be parsed to the forecast function

    '''
    #TODO add seasonal
    #construct ARIMA model
    warn = check_for_errors(data, time)
    
    model = ARIMA(data, order = order,
                  seasonal_order = season,
                  missing = 'raise')
    
    #fit
    result = model.fit()
    print(result.summary())
    #plot best model?
    if plot_result:
        fig = plt.figure(figsize=(6,8))
    
        ax = fig.add_subplot(3,1,1)
        ax.plot(time,data, label = 'Data')
        ax.plot(time,result.fittedvalues, label = 'Fitted Model')
        ax.legend()
        
        ax = fig.add_subplot(3,1,2)
        ax.plot(time,result.resid)
        ax.set_title('Residuals')
        
        ax = fig.add_subplot(3,1,3)
        ax.plot(time,result.resid)
        ax.set_title('Autocorrelation of Residuals')
        ax.set_xlabel('Lags')
        ax.set_ylabel('ACF')
        
        plt.tight_layout()
    
    #TODO print RMSE, or AIC to compare
    #maybe parse on to automate finding best model
    return result

def forecast_ARIMA(result, steps, confidence = None, plot_results = True):
    pred = result.forecast(steps, alpha = confidence)

    plt.plot(data, label= 'data')
    plt.plot(len(data)+np.arange(pred.shape[0]),pred,lw = 3, label = 'forecast')
    plt.legend()
    return pred


#res = construct_ARIMA(data, time, order = (7,1,1))
#%% test






