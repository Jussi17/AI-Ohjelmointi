# M12 Konenäkölaitteisto mittaa kuulantyönnön harjoituksissa kuulan x- ja
# y-koordinaattia 50 ms välein. y−koordinaatin 0-taso on kentän pinnan
# tasolla. Yhden työntön esikäsitellyt mittaustulokset on talletettu
# tiedostoon mittaus.csv. Tehtävänäsi on ennustaa kuulan
# laskeutumispaikan x-koordinaatti, eli työntön pituus. Ilmanvastus
# jätetään huomioimatta. Ohjelma tulostaa kuulan laskeutumispaikan
# x-koordinaatin (metreissä, ilman yksikköä). (noin 14 m)


import sys
import time
import pandas
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import os

basedir = os.path.dirname(__file__)
inputfile = os.path.join(basedir, 'mittaus.csv')

data = pandas.read_csv(inputfile)
data = data.dropna()

X = data['x'].values.reshape(-1, 1)
y = data['y'].values

# Polynominen regressio (2. asteen paraabeli)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = linear_model.LinearRegression()
model.fit(X_poly, y)

c = model.intercept_
b = model.coef_[1]
a = model.coef_[2]

discriminant = b**2 - 4*a*c
x1 = (-b + np.sqrt(discriminant)) / (2*a)
x2 = (-b - np.sqrt(discriminant)) / (2*a)

# Valitaan suurempi x (laskeutumispaikka, ei lähtöpiste)
x_landing = max(x1, x2)

# Tulostetaan laskeutumispaikan x-koordinaatti
print(x_landing)