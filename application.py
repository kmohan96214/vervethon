from flask import Flask, redirect, render_template, request, url_for

from readData import *
from helpers import *

app = Flask(__name__)

availableCountries = readcsv()

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/",methods=['POST'])
def search():
    if request.method == "POST":
        country = request.form["country"]
        if(country.lower() not in availableCountries):
            return apology("Country data unavailable")
        years,csmp = readCountry(country)
        """
        data models
        """
        """
        plots
        """
        linear = linearRegressionMod(years,csmp)
        poly2 = polynomialRegression(years,csmp,2)
        poly3 = polynomialRegression(years,csmp,3)
        poly4 = polynomialRegression(years,csmp,4)
        poly8 = polynomialRegression(years,csmp,8)
        sarimaxmod = sarimax(years,csmp)
        hwesmod = hwes(years,csmp)

        charts = []
        charts.append(plot(years,csmp,"data"))
        charts.append(plot(linear[1][0],linear[1][1],"linear fit" , years ,csmp))
        charts.append(plot(poly2[1][0],poly2[1][1], "polynomial fit degree 2" , years ,csmp))
        charts.append(plot(poly3[1][0],poly3[1][1], "polynomial fit degree 3" , years ,csmp))
        charts.append(plot(poly4[1][0],poly4[1][1], "polynomial fit degree 4" , years ,csmp))
        charts.append(plot(poly8[1][0],poly8[1][1],"polynomial fit degree 8" , years ,csmp))
        charts.append(plot(sarimaxmod[1][0],sarimaxmod[1][1],"sarimax", years ,csmp))
        charts.append(plot(hwesmod[1][0],hwesmod[1][1],"hwes" ,years ,csmp))

        return render_template("plot.html",charts = charts,country=country)        

    