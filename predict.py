from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import numpy as np
import matplotlib.pyplot as plt
import pylab

import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
import plotly

from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor

def LRM(x,y):
    xvals = np.array(x[:-1],dtype=np.dtype('Float64'))
    yvals = np.array(y[:-1],dtype=np.dtype('Float64'))
    coefs = pylab.polyfit(xvals,yvals,1)
    estvals = coefs[0]*xvals[-1] + coefs[1]
    return [(xvals,yvals),(np.array(xvals[-1]),np.array(estvals))]

def PR(x,y,n):
    xvals = np.array(x[:-1],dtype=np.dtype('Float64'))
    yvals = np.array(y[:-1],dtype=np.dtype('Float64'))
    coefs = pylab.polyfit(xvals,yvals,n)
    estvals = np.array([0.0 for i in x])
    for i in range(n+1):
        estvals = estvals + (coefs[i]*( np.power(xvals[-1],(n-i)) ))
    return [(xvals,yvals),(np.array(xvals[-1]),np.array(estvals))]

def SRMX(x,y):
    data = np.array(y[:-2],dtype = np.dtype('Float64'))
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
    model_fit = model.fit(disp=False)
    yhat = model_fit.predict(len(data)-1,len(data))
    return [(x,y),(np.array(x[-2:]),np.array(yhat))]

def ARM(x,y):
    data =  np.array(y[-1],dtype = np.dtype('Float64'))
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit(disp=False)
    yhat = y[0] + model_fit.predict(len(data)-2, len(data)+1)
    print(x[-1])
    return [(x,y),(np.array(x[-1]),np.array(yhat))]

def HWES(x,y):
    data = np.array(y[:-2],dtype = np.dtype('Float64'))
    model = ExponentialSmoothing(data)  
    model_fit = model.fit()
    yhat = model_fit.predict(len(data)-1, len(data))
    print(x[-1])
    return [(x,y),(np.array(x[-2:]),np.array(yhat))]