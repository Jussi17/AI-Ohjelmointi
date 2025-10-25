# M15 Kirjoita ohjelma, joka lukee opetusdatan tiedostoista teach data.npy
# ja teach class.npy. Jälkimmäinen tiedosto sisältää kunkin
# datapisteen luokan. Jaa data opetus- ja testijoukkoon ja opeta sillä
# valitsemasi luokittelija. Luokittele tiedoston data in.npy aineisto.
# Talleta luokittelun tulos tiedostoon data classified.npy. Tämä
# tiedosto on samaa muotoa kuin teach class.npy. Vaatimuksena on
# vähintään 92% tarkkuus luokittelussa.
# Valitse sopiva luokittelumenetelmä ja lisää tarvittavat import-lauseet
# tiedoston alkuun. Älä normalisoi datoja.


import sys
import time
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

basedir = os.path.dirname(__file__)
real_X = np.load(os.path.join(basedir, 'data_in.npy'))
X = np.load(os.path.join(basedir, 'teach_data.npy'))
Y = np.load(os.path.join(basedir, 'teach_class.npy'))

################################################
# Your code below this line
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=500, 
    max_depth=25, 
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print('Accuracy:', accuracy)

# Your code above this line
################################################

print('Compute real predictions')
real_X = np.load('data_in.npy')

print('real_X -', np.shape(real_X))
pred = model.predict(real_X)
print('pred -', np.shape(pred))
np.save('data_classified.npy', pred)