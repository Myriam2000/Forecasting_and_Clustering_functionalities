#general
import numpy as np
import pandas as pd
# holt winters  
from statsmodels.tsa.holtwinters import ExponentialSmoothing
#confidence interval
from sklearn.utils import resample
#ARIMA
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
#SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')


def HW(ts_train, d, m=12, type='add'):  # Holt’s Winters Seasonal
    model = ExponentialSmoothing(ts_train, trend = type,seasonal=type, seasonal_periods=m, initialization_method="estimated")
    HW_fit = model.fit()
    # Predict the next values
    pred = HW_fit.forecast(d)
    simulations = HW_fit.simulate(d, repetitions=50, error="mul")
    # Calculate confidence intervals
    lower = pd.DataFrame(np.percentile(simulations.values, 0, axis=1), index=simulations.index)
    upper = pd.DataFrame(np.percentile(simulations.values, 100, axis=1), index=simulations.index)
    return pred, HW_fit.fittedvalues, lower[0], upper[0]

def arma (ts_train, d):
    model_arima = ARIMA(ts_train, order = (2,0,2))
    ARIMA_fit = model_arima.fit()
    forecast = ARIMA_fit.get_forecast(d)
    pred = forecast.predicted_mean
    yhat_conf_int = forecast.conf_int(alpha=0.05)
    return pred, ARIMA_fit.fittedvalues, yhat_conf_int['lower Anzahl'], yhat_conf_int['upper Anzahl']

def set_arima_orders (ts):
    stepwise_fit = auto_arima(ts, start_p=2, start_q=2, suppress_warnings=True)
    stepwise_fit = str(stepwise_fit)
    return stepwise_fit[6:13]

def arima (ts_train, d, param): 
    model_arima = ARIMA(ts_train, order = (int(param[1]),int(param[3]),int(param[5])))
    ARIMA_fit = model_arima.fit()
    forecast = ARIMA_fit.get_forecast(d)
    pred = forecast.predicted_mean
    yhat_conf_int = forecast.conf_int(alpha=0.05)
    return pred, ARIMA_fit.fittedvalues, yhat_conf_int['lower Anzahl'], yhat_conf_int['upper Anzahl']

def set_sarima_orders (ts, m=12):
    stepwise_fit = auto_arima(ts, start_p=2, start_q=2, start_P=0, start_Q=0, seasonal=True,  m=m, d=1, D=1, suppress_warnings=True)
    stepwise_fit = str(stepwise_fit)
    return stepwise_fit[6:20]

def sarima(ts_train, d, param, m):
    SARIMA_fit = SARIMAX(ts_train, order = (int(param[1]),int(param[3]),int(param[5])), seasonal_order=(int(param[8]), int(param[10]), int(param[12]), m)).fit() 
    forecast = SARIMA_fit.get_prediction(start=len(ts_train), end=len(ts_train)+(d-1))
    pred = forecast.predicted_mean
    conf_int = forecast.conf_int()
    return pred, SARIMA_fit.fittedvalues, conf_int.iloc[:, 0], conf_int.iloc[:, 1]
    
def output(ts_train, pred, fit, lower, upper, input_name, method): # add forecast to the time series and save the dataframe
    forecast_output = ts_train.copy()
    forecast_output = pd.DataFrame(forecast_output._append(pred))
    forecast_output.rename(columns = {0:'data'}, inplace = True)
    fitting = pd.concat([fit, pd.DataFrame(index=pred.index)], sort=False)
    lower = pd.concat([pd.DataFrame(index=ts_train.index), lower], sort=False)
    upper = pd.concat([pd.DataFrame(index=ts_train.index), upper], sort=False)
    forecast_output['fitting'] = fitting
    forecast_output['lower'] = lower
    forecast_output['upper'] = upper
    forecast_output.to_csv('../output/{}_forecast_output_{}.csv'.format(input_name, method))
    print("The file has been successfully saved.")
    return forecast_output


def use_forecasting_method(ts_train, ts, d, input_name, method='HW', m=12): # main function for clustering
    if method == 'ARMA':
        pred, fit, lower, upper = arma(ts_train, d)

    elif method == 'HW':
        pred, fit, lower, upper = HW(ts_train, d)

    elif method == 'ARIMA':
        param = set_arima_orders(ts)        
        pred, fit, lower, upper = arima(ts_train, d, param)

    elif method == 'SARIMA':
        param = set_sarima_orders(ts, m)  
        pred, fit, lower, upper = sarima(ts_train, d, param, m)
    
    forecast_output = output(ts_train, pred, fit, lower, upper, input_name, method)
    
    return forecast_output

