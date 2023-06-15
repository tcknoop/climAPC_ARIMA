# climAPC_ARIMA
Contains functions to help fit a (S)ARIMA model to timeseries data, evaluate the model and to forecast/predict the timeseries. 

## Theory 
An ARIMA model consists of three components with their corresponding parameters:
  - AR(p): Autoregression
  - I(d): Integrated
  - MA(q): Moving average

AR and MA both make assumptions about the relation of the next timestep to the previous datapoints. In the AR model the order p indicates on how many previous values the current value $X_t$ depends: 
```math
X_t = \sum_{i=1}^{p}\psi X_{t-i} + \epsilon_t
```
with $\psi_{i-p}$ being the parameters and $\epsilon_t$ as white noise.

If the raw observations are non-stationary, replacing the raw data by the difference $Z_t$ of their values to help with stationarity (which the other two models need). This is done d times:
```math
Z_t = X_{t+1} - X_t
```



### References 
Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting: principles and practice, 2nd edition, OTexts: Melbourne, Australia. OTexts.com/fpp2. Accessed on 15.06.2023.
