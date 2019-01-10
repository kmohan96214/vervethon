from flask import redirect, render_template, request, session, url_for

import pprint
import numpy as np

import matplotlib.pyplot as plt
import pylab

import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
import plotly

plotly.tools.set_credentials_file(username='kmohan962', api_key='tVpcTsGDBQ1SnAb28tZB')

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

def plot(x,y):
    trace = go.Scatter(x=x,y=y)
    data = [trace]
    layout= go.Layout(title="elcsmp", xaxis={'title':'years'}, yaxis={'title':'electricity consumption'})
    figure =  go.Figure(data=data,layout=layout)
    return py.iplot(figure)

def linearRegressionMod(x,y):
    print(x,y)
    xvals = np.array(x,dtype=np.dtype('Float64'))
    yvals = np.array(y,dtype=np.dtype('Float64'))
    coefs = pylab.polyfit(xvals,yvals,1)
    estvals = coefs[0]*xvals + coefs[1]
    return [(xvals,yvals),(xvals,estvals)]