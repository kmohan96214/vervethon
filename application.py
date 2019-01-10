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
        chart = plot(linear[1][0],linear[1][1])
        charts = [chart]*8
        return render_template("plot.html",charts = charts,country=country)
        

    