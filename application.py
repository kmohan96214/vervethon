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
        country = request.args.get("country")
        if(country not in availableCountries):
            return apology("Country data unavailable")
        years,csmp = readCountry(country)
        """
        data models
        """
        """
        plot
        """
        

        

    