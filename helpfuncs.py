import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy import stats
import warnings
#%%

def check_interp(array,time):
    nan_count = np.isnan(array).sum()
    if nan_count > 0:
        warning_message = f"{nan_count} NaN Value(s) detected in timeseries; interpolation or deletion of missing values advised."
        warnings.warn(warning_message,stacklevel = 2 )
    #TODO #maybe include in     
    #check timestep continuity
    
    #check Value errors over std?; over intg std?
    
    return None

def difference(x,t,d):
    
    for i in range(d):
        x = np.ediff1d(x)
    
    return x,t[d:]

#%%
