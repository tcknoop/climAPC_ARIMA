import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
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


def check_for_errors(array,time,out_threshold = 4, set_nan = True):
    '''
    Checks the Timeseries for Nans, missing data, uneven timesteps or extreme discontinuities.
    Sets outliers to NaN by default.

    Parameters
    ----------
    data : 1-D array-like
        Input timeseries.
    time : 1-D array-like
        Input time vector, same length as data.
    out_threshold: int
        theshold; number of multiple standard deviations before a datapoint is marked as discontinuity
    set_nan: bool
        replaces marked datapoints as NaN.

    Returns
    -------
    Corrected data and time arrays. 
    warn: str
        Warning message

    '''
    err_found = 0
    warning_message = None
    nan_count = np.isnan(array).sum() #count NaNs
    if nan_count > 0:
        warning_message = f"{nan_count} NaN Value(s) detected in timeseries;\nconstructARIMA can deal with NaN."
        warnings.warn(warning_message,stacklevel = 2 )
        err_found =+ 1
        
    diff = difference(time)
    if np.amax(diff)>np.mean(diff)+0.001: #check for inconsistency in timestep; allow for float error
        warning_message = 'Irregular timesteps found; filling missing timesteps with NaN\nor Interpolation to regular timesteps advised'
        warnings.warn(warning_message,stacklevel = 2 )
        err_found =+ 1
    
    #check if any indices lie way outside the standard deviation -> discontinuities
    z_score = (array - np.mean(array)) / np.std(array)
    is_outlier = np.abs(z_score) > out_threshold
    outlier_indices = np.where(is_outlier)[0].tolist()
    
    if outlier_indices:
        if len(outlier_indices) > 5:
            warning_message = f'Outlier datapoint(s) found at {outlier_indices[:5]}...'
        else:    
            warning_message = f'Outlier datapoint(s) found at {outlier_indices}'
        
        err_found =+ 1
        if set_nan:
            array[outlier_indices] = np.nan
            
            warning_message = warning_message + ',\nset to NaN'
    
        warnings.warn(warning_message,stacklevel = 2 )
    
    if err_found == 0:
        print('no apparent errors found in timeseries')
        return array, time, None
    if err_found >= 1:
        return array, time, warning_message

def fix_data(data, time, method):
    '''
    
    fixes the timeseries eihter by cutting NaN datapoints (not recommended), or
    by interpolation scipy.interpolate.PchipInterpolator

    '''

    if method == 'interp':#scipy cubic spline
        nan_ind = np.isnan(data)
        d_known = data[~nan_ind]
        t_known = time[~nan_ind]
        
        cs = PchipInterpolator(t_known, d_known)
        data = cs(time)
        
    if method == 'cut':
        time = time[~np.isnan(data)]
        data = data[~np.isnan(data)]
        #cut timeseries will work, but ARIMA assumes equal timesteps 
        #-> weigthing of individual datapoints will be off
  
    return data, time
#%%
