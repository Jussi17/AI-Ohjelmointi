# M19 Annetun aineiston1
# viimeinen sarake sisältää tiedon kotiloiden
# vuosirenkaiden lukumäärästä, jonka perusteella voidaan laskea kotilon
# ikä. Aineiston ensimmäinen sarake on simpukan sukupuoli, joka voi
# olla F/I/M. Nämä korvataan arvoilla -1/0/1 esimerkiksi käyttämällä
# pandas.DataFrame.replace-metodia. Tehtävänä on ennustaa
# mittausaineistosta kotilon vuosirenkaiden lukumäärä oikein ±3
# kappaleen tarkkuudella vähintään 70% mittausaineiston tapauksissa.
# Käytä viimeisen tason aktivaatiofunktiona ’linear’-funktiota ja tee
# verkko jossa on vain yksi ulostulo.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import os

Y_COLUMN = 8

def splitXY(d):
    return d.drop(Y_COLUMN, axis=1), d[Y_COLUMN]

basedir = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(basedir, 'traindata.csv'), header=None)

# Korvaa sukupuolitiedot numeerisilla arvoilla
data = data.replace({'F': -1, 'I': 0, 'M': 1})
data = shuffle(data, random_state=42)

# Jaa piirteet ja tavoitteet
X, Y = splitXY(data)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Määritellään neuroverkko
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=0)

# Ennustetaan testidatalla
predY = np.round(model.predict(X_test)).reshape((X_test.shape[0], ))

# Lasketaan ±3 tarkkuuden sisällä olevien ennusteiden määrä
accepted_n = (np.abs(predY - Y_test) <= 3).sum()
print('Correct predictions:', accepted_n, '/', Y_test.shape[0])

# Tallennetaan malli
model.save('kotilo.keras')


