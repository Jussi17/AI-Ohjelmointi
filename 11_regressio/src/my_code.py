# M11 Fysiikan laboratoriotyön mittauspöytäkirja on talletettu tiedostoon
# measurements.csv. Kappaleen massa on 0.75 kg. Tehtävänäsi on
# tarvittaessa esikäsitellä data ja määrittää mitatun aineen
# ominaislämpökapasiteetti. Ohjelma tulostaa ominaislämpökapasiteetin
# lukuarvon yksikössä kJ/(kg*K) (Noin 0.45)

import sys
import time
import pandas
from scipy import stats
import numpy as np
from sklearn import linear_model
import os

basedir = os.path.dirname(__file__) 
inputfile = os.path.join(basedir, 'measurements.csv')
m = 0.75  # mass of the object

data = pandas.read_csv(inputfile)

# Esikäsittely: poistetaan puuttuvat arvot
data = data.dropna()

T = data['T'].values
E = data['E'].values

# Lineaarinen regressio
slope, intercept, r_value, p_value, std_err = stats.linregress(T, E)

c = abs(slope)

# Estimated specific heat capacity
print(c)