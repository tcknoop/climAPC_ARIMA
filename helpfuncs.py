import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy import stats
import warnings
#%%



def difference(x, t = None, d = 1):
    '''
    Takes the difference of the timeseries x a total of d times. Functions as the "I" part of the ARIMA model.

    Parameters
    ----------
    x : 1-D array:
        Timeseries data; same dimension as t
    t : 1-D array, optional:
        Timevector, same dimension as x.
    d : int
        order of differencing; d=0 returns unchanged vectors.

    Returns
    -------
    x : 1-D array;
        Differenced timeseries; shortened by d indeces from the front.
    t : 1-D array;
        Differenced timevector; shortened by d indeces from the front.

    '''
    #take difference of data d times
    for i in range(d):
        x = np.ediff1d(x)
    
    if t is  None:
        return x
    else:
        return x,t[d:] #slice timevector to same dimensions 


def check_for_errors(array,time):
    '''
    Checks the Timeseries for Nans, missing data, uneven timesteps or extreme discontinuities.

    Parameters
    ----------
    data : 1-D array-like
        Input timeseries.
    time : 1-D array-like
        Input time vector, same length as data.

    Returns
    -------
    None.

    '''
    err_found = 0
    warning_message = None
    nan_count = np.isnan(array).sum() #count NaNs
    if nan_count > 0:
        warning_message = f"{nan_count} NaN Value(s) detected in timeseries;\ninterpolation or deletion of missing values advised."
        warnings.warn(warning_message,stacklevel = 2 )
        err_found =+ 1
        
    diff = difference(time)
    if np.amax(diff)>np.mean(diff)+0.001: #check for inconsistency in timestep; allow for float error
        warning_message = 'Irregular timesteps found; filling missing timesteps with NaN\nor Interpolation to regular timesteps advised'
        warnings.warn(warning_message,stacklevel = 2 )
        err_found =+ 1
    
    #check if any indices lie way outside the standard deviation -> discontinuities
    z_score = (array - np.mean(array)) / np.std(array)
    is_outlier = np.abs(z_score) > 3
    outlier_indices = np.where(is_outlier)[0].tolist()
    if outlier_indices:
        warning_message = f'Outlier datapoint(s) found at {outlier_indices},\nmaybe delete or set to NaN?'
        warnings.warn(warning_message,stacklevel = 2 )
        err_found =+ 1

    if err_found == 0:
        print('no apparent errors found')
        return None
    if err_found >= 1:
        return warning_message

#%%
