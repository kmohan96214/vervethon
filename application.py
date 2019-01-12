from flask import Flask, redirect, render_template, request, url_for
import math

from readData import *
from helpers import *
from predict import *

app = Flask(__name__)

availableCountries = readcsv()

@app.route("/")
def index():
    return render_template('index.html',countries = sorted(availableCountries))

@app.route("/predictions/<string:country>",methods=['GET'])
def predictions(country):
    if request.method == "GET":
        if(country.lower() not in availableCountries):
            return apology("Country data unavailable")
        try:
            years,csmp = readCountry(country)
        except :
            return apology("Country data unavailable")
        try:
            linear = LRM(years,csmp)
            poly2 = PR(years,csmp,2)
            poly3 = PR(years,csmp,3)
            poly4 = PR(years,csmp,4)
            poly8 = PR(years,csmp,8)
            sarimaxmod = SRMX(years,csmp)
            hwesmod = HWES(years,csmp)
            #rndfrtmod = Random_Forest_Regressor(years,csmp)
            #xgbmod = xgbregression(years,csmp)
        except :
            return apology("error")

        charts = []
        charts.append(plot(years,csmp,"Actual data"))
        charts.append(plot(poly8[1][0],poly8[1][1],"polynomial fit degree 8" , years ,csmp))
        charts.append(plot(sarimaxmod[1][0],sarimaxmod[1][1],"sarimax", years ,csmp))
        charts.append(plot(hwesmod[1][0],hwesmod[1][1],"hwes" ,years ,csmp))
        charts.append(plot(poly2[1][0],poly2[1][1], "polynomial fit degree 2" , years ,csmp))
        charts.append(plot(poly3[1][0],poly3[1][1], "polynomial fit degree 3" , years ,csmp))
        charts.append(plot(poly4[1][0],poly4[1][1], "polynomial fit degree 4" , years ,csmp))
        charts.append(plot(linear[1][0],linear[1][1],"linear fit" , years ,csmp))
        
        errors1 = []
        errors1.append(0.0)
        errors1.append(abs(poly8[1][1] -csmp[-1])[0])
        errors1.append(abs(sarimaxmod[1][1] -csmp[-1])[1] )
        errors1.append(abs(hwesmod[1][1]-csmp[-1])[1] )
        errors1.append(abs(poly2[1][1]-csmp[-1])[0] )
        errors1.append(abs(poly3[1][1]-csmp[-1])[0])
        errors1.append(abs(poly4[1][1]-csmp[-1])[0])
        errors1.append(abs(linear[1][1]-csmp[-1]))

        return render_template("predictions.html",charts = charts,country=country.upper(),errors=errors1)


@app.route("/search",methods=['POST'])
def search():
    if request.method == "POST":
        country = request.form["country"]
        if(country.lower() not in availableCountries):
            return apology("Country data unavailable")
        try:
            years,csmp = readCountry(country)
        except :
            return apology("Country data unavailable")
    
        try:
            linear = linearRegressionMod(years,csmp)
            poly2 = polynomialRegression(years,csmp,2)
            poly3 = polynomialRegression(years,csmp,3)
            poly4 = polynomialRegression(years,csmp,4)
            poly8 = polynomialRegression(years,csmp,8)
            sarimaxmod = sarimax(years,csmp)
            hwesmod = hwes(years,csmp)
            #rndfrtmod = Random_Forest_Regressor(years,csmp)
            #xgbmod = xgbregression(years,csmp)
        except :
            return apology("error")
        
        errors = get_errors([linear,poly2,poly3,poly4,poly8,sarimaxmod,hwesmod])
        least_error = 10000000007
        
        least_error = min(errors)
        j = errors.index(least_error)

        charts = []
        charts.append(plot(years,csmp,"Actual data"))
        charts.append(plot(linear[1][0],linear[1][1],"linear fit" , years ,csmp))
        charts.append(plot(poly2[1][0],poly2[1][1], "polynomial fit degree 2" , years ,csmp))
        charts.append(plot(poly3[1][0],poly3[1][1], "polynomial fit degree 3" , years ,csmp))
        charts.append(plot(poly4[1][0],poly4[1][1], "polynomial fit degree 4" , years ,csmp))
        charts.append(plot(poly8[1][0],poly8[1][1],"polynomial fit degree 8" , years ,csmp))
        charts.append(plot(sarimaxmod[1][0],sarimaxmod[1][1],"sarimax", years ,csmp))
        charts.append(plot(hwesmod[1][0],hwesmod[1][1],"hwes" ,years ,csmp))

        charts[j+1],charts[1] = charts[1],charts[j+1]
        errors[j],errors[0] = errors[0],errors[j]

        return render_template("plot.html",charts = charts,country=country.upper(),errors=errors)