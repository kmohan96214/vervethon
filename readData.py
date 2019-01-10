import pandas as pd
import numpy as np

def readcsv():
    datacnt = pd.read_csv('countrydata.csv',sep=',')
    availableData = set()
    for i in range(len(datacnt)):
        availableData.add(datacnt.iloc[i]['Country Name'])
    return availableData

def readCountry(country):
    datacnt = pd.read_csv('countrydata.csv',sep=',')
    years = []
    csmp = []
    for i in range(len(datacnt)):
        if datacnt.iloc[i]["Country Name"].lower() == country.lower():
            l = list(datacnt.columns.values)
            for j in l:
                if( type(datacnt.iloc[i][j]).__name__ != 'str' and  not np.isnan(datacnt.iloc[i][j]) ):
                    years.append(j)
                    csmp.append(datacnt.iloc[i][j])
    return years,csmp