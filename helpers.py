from flask import redirect, render_template, request, session, url_for
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
#plotly.tools.set_credentials_file(username='kmohan962', api_key='tVpcTsGDBQ1SnAb28tZB')

def apology(top="", bottom=""):
    """Renders message as an apology to user."""
    def escape(s):
        """
        Escape special characters.

        https://github.com/jacebrowning/memegen#special-characters
        """
        for old, new in [("-", "--"), (" ", "-"), ("_", "__"), ("?", "~q"),
            ("%", "~p"), ("#", "~h"), ("/", "~s"), ("\"", "''")]:
            s = s.replace(old, new)
        return s
    return render_template("apology.html", top=escape(top), bottom=escape(bottom))

def plot(x,y,name,*args):
    predicted = go.Scatter(x=x,y=y,name="predicted")
    if args:
        actual = go.Scatter(x =args[0],y=args[1],name="actual")
        data = [predicted,actual]
    else:
        data = [predicted]
    layout= go.Layout(title=name, xaxis={'title':'years'}, yaxis={'title':'electricity consumption'})
    figure =  go.Figure(data=data,layout=layout)
    return plotly.offline.plot(figure,output_type="div")

def linearRegressionMod(x,y):
    xvals = np.array(x,dtype=np.dtype('Float64'))
    yvals = np.array(y,dtype=np.dtype('Float64'))
    coefs = pylab.polyfit(xvals,yvals,1)
    estvals = coefs[0]*xvals + coefs[1]
    return [(xvals,yvals),(xvals,estvals)]

def polynomialRegression(x,y,n):
    xvals = np.array(x,dtype=np.dtype('Float64'))
    yvals = np.array(y,dtype=np.dtype('Float64'))
    coefs = pylab.polyfit(xvals,yvals,n)
    estvals = np.array([0.0 for i in x])
    for i in range(n+1):
        estvals = estvals + (coefs[i]*( np.power(xvals,(n-i)) ))
    return [(xvals,yvals),(xvals,estvals)]

def sarimax(x,y):
    data = np.array(y,dtype = np.dtype('Float64'))
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
    model_fit = model.fit(disp=False)
    yhat = model_fit.predict(0,len(data)-1)
    return [(x,y),(x,yhat)]

def arima(x,y):
    data =  np.array(y,dtype = np.dtype('Float64'))
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit(disp=False)
    yhat = y[0] + model_fit.predict(1, len(data)-1)
    return [(x,y),(x,yhat)]

def hwes(x,y):
    data = np.array(y,dtype = np.dtype('Float64'))
    model = ExponentialSmoothing(data)  
    model_fit = model.fit()
    yhat = model_fit.predict(0, len(data)-1)
    return [(x,y),(x,yhat)]

def Random_Forest_Regressor(x,y):
    data =  np.array(y,dtype = np.dtype('Float64'))
    model = RandomForestRegressor(n_estimators=20, random_state=0)
    model_fit = model.fit(x,y)
    yhat = y[0] + model_fit.predict(1,len(data)-1)
    return [(x,y),(x,yhat)] 

def xgbregression(x,y):    
    data =  np.array(y,dtype = np.dtype('Float64'))
    model = XGBClassifier()
    model_fit = model.fit(x,y)
    yhat = y[0] + model_fit.predict(1,len(data)-1)
    return [(x,y),(x,yhat)] 


def rmserr(y,y_):
    return sqrt(mean_squared_error(y, y_))

def get_errors(l):
    errors = []
    for i in l:
        errors.append( rmserr(i[0][1],i[1][1]) )
    return errors

